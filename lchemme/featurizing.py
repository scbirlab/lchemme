"""Utilities for featurizing."""

from typing import Callable, Dict, Iterable, Mapping, Optional, Union

from collections import defaultdict
from functools import partial
import os
import sys

from carabiner import print_err
from carabiner.cast import cast
from carabiner.itertools import tenumerate
from carabiner.mpl import grid
from datasets import Dataset
from numpy import ndarray
import numpy as np
from pandas import DataFrame
from schemist.tables import converter
import torch
# from transformers import AutoModelForSeq2SeqLM, PreTrainedModel, PreTrainedTokenizerBase
from umap import UMAP

from .io import read_dataset
from .pretraining import _load_tokenizer

_FEATURE_METHODS = (
    'start', 'end', 'sum', 
    'mean', 'median', 'var', 'max',
)


def _check_columns(columns, dataset):
    if columns is None:
        columns = []
    else:
        missing_from_data = [col for col in columns 
                             if col not in dataset.column_names]
        if len(missing_from_data) > 0:
            raise KeyError(f"Columns missing from dataset: {', '.join(missing_from_data)}")
    return columns

def _return_embedding(
    last_hidden, 
    method: str = 'mean'
):
    """Aggregate LLM embedding along sequence dimension.

    Examples
    ========
    >>> import torch
    >>> emb = _return_embedding(torch.randn(2, 4, 8), method="mean")
    >>> emb.shape == (2, 8)  # pooled to (B, H)
    True
    >>> emb_var = _return_embedding(torch.ones(1, 3, 4), method="var")
    >>> torch.all(torch.isfinite(emb_var))  # no NaNs / infs
    tensor(True)
    
    """

    if method == 'start':
        vals = last_hidden[:,0,:]
    elif method == 'end':
        vals = last_hidden[:,-1,:]
    elif method == 'flat':
        raise NotImplementedError(f"No embedding method {method}")
        # a = torch.nan_to_num(torch.flatten(last_hidden, start_dim=1),
        #                      posinf=0., neginf=0.)
        # nan_count = torch.sum(torch.isnan(a)).numpy()
        # assert nan_count == 0, f"There are {nan_count} NaNs in embedding."
        # vals = a
    elif method == 'sum':
        vals = torch.sum(last_hidden, dim=1)
    elif method == 'mean':
        vals = torch.mean(last_hidden, dim=1)
    elif method == 'median':
        vals = torch.median(last_hidden, dim=1).values
    elif method == 'var':
        vals = torch.var(last_hidden, dim=1)
    elif method == 'max':
        vals = torch.max(last_hidden, dim=1).values
    elif isinstance(method, int) and method >= 0 and method < last_hidden.shape[1]:
        vals = last_hidden[:,method,:]
    else:
        raise NotImplementedError(f"No embedding method {method}")

    return vals.detach()


def _tokenize_for_embedding(
    ds: Mapping[str, Iterable],
    tokenizer, 
    column: str = 'smiles'
) -> Dict[str, ndarray]:

    """Tokenize strings before embedding with BART.

    Examples
    ========
    >>> from transformers import AutoTokenizer
    >>> tok = AutoTokenizer.from_pretrained("sshleifer/bart-tiny-random")
    >>> toks = _tokenize_for_embedding({"smiles": ["CC"]}, tok)
    >>> set(toks) == {"input_ids", "attention_mask"}  # correct keys
    True

    """

    smiles = cast(ds[column], to=list)
    smiles = [s if s is not None else '<unk>' for s in smiles]

    try:
        tokenized = tokenizer(smiles, 
                              return_tensors='pt', 
                              padding=True)
    except (ValueError, TypeError):
        tokenized = tokenizer(['<unk>'] * len(smiles), 
                              return_tensors='pt', 
                              padding=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {key: tokenized[key].to(device) for key in ['input_ids', 'attention_mask']}

    
def _embed_smiles(
    ds: Mapping,
    tokenizer,
    model, #: PreTrainedModel, 
    column: str = 'smiles',
    method: Union[str, Iterable[str]] = 'mean'
) -> Dict[str, ndarray]:

    method = cast(method, to=list)
    ds_tokenized = _tokenize_for_embedding(ds, tokenizer=tokenizer, column=column)
    
    tokenized_inputs = {key: ds_tokenized[key] for key in ['input_ids', 'attention_mask']}
    model.eval()
    with torch.no_grad():
        outputs = model(
            **tokenized_inputs, 
            decoder_input_ids=tokenized_inputs['input_ids'],
            output_hidden_states=True,
            return_dict=True,
        )
    
    enc_last = outputs.encoder_hidden_states[-1]
    dec_last = outputs.decoder_hidden_states[-1]
        
    if len(method) == 1:
        results = {
            "embedding": np.concatenate([
                _return_embedding(enc_last, method=method[0]),
                _return_embedding(dec_last, method=method[0]),
            ], axis=-1)
        }
    else:
        results = {
            f"embedding_{str(meth)}": np.concatenate([
                _return_embedding(enc_last, method=meth),
                _return_embedding(dec_last, method=meth),
            ], axis=-1) for meth in method
        }

    results.update({"smiles": ds[column]})

    return results


def plot_embedding(
    embedding: Dataset,
    embedding_columns: Iterable[str],
    column: str = 'smiles',
    descriptor: Optional[Union[str, Iterable[str]]] = None,
    sample_size: int = 20_000,
    additional_columns: Optional[Union[str, Iterable[str]]] = None
):

    """

    """
    
    if descriptor is None:
        descriptor = ['mwt', 'clogp', 'tpsa', 'max_charge', 'min_charge']
    else:
        descriptor = cast(descriptor, to=list)

    additional_columns = _check_columns(additional_columns, embedding)

    print_err(f"Plotting UMAP embedding with maximum {sample_size} rows, coloring by", 
              ", ".join(descriptor + additional_columns))

    fig, axes = grid(
        ncol=len(descriptor + additional_columns) + 1, 
        nrow=len(embedding_columns), 
        aspect_ratio=1.1,
        sharex='row', sharey='row', 
        squeeze=False,
    )

    desc_dfs = {}
    df = embedding

    if df.num_rows > sample_size:
        df = df.shuffle(seed=1).take(sample_size) 

    df0 = df.to_dict()
    smiles_data = df0[column]
    valid_smiles_filter = [s != ""  and s is not None and "." not in s for s in smiles_data]
    valid_smiles = [s for s, _filter in zip(smiles_data, valid_smiles_filter) if _filter]
    descriptor_data = {
        col: [v for v, _filter in zip(df0[col], valid_smiles_filter) if _filter] 
        for col in additional_columns
    }

    desc_df0 = DataFrame(dict(smiles=valid_smiles, 
                              points=np.nan * np.ones((len(valid_smiles), )),
                              **descriptor_data))
    errors, desc_df0 = converter(desc_df0, 
                                 output_representation=descriptor)
    
    for row, method in zip(axes, embedding_columns):

        df = DataFrame(df0[method], 
                       index=smiles_data)
        df = df[valid_smiles_filter]
        reducer = UMAP()
        reducer.fit(df)
        reduced = reducer.transform(df) 
        
        desc_df = desc_df0.copy().assign(
            dim1=reduced[:, 0], 
            dim2=reduced[:, 1],
        )
        desc_dfs[method.replace('embedding_', '')] = desc_df.sort_values(by=['dim1', 'dim2', 'smiles'])
    
        for ax, desc in zip(row, descriptor + additional_columns + ['points']):
            colors, log_colors = desc_df[desc].values, False
            try:
                isnan_colors = np.isnan(colors)
            except TypeError as e:
                print_err(f"Some values from column '{desc}' were incompatible for testing for NaN.")
                raise e
            else:
                all_values_gt0 = np.all(colors[~isnan_colors] >= 0.)
                some_not_nan = len(colors[~isnan_colors]) > 0
                if all_values_gt0 and some_not_nan and desc not in ('min_charge', 'max_charge'):
                    min_not_eq_max = colors[~isnan_colors].min() != colors[~isnan_colors].max()
                    if min_not_eq_max:
                        colors, log_colors = colors, True
            if log_colors:
                colors = np.where(colors <= 0., np.nan, colors)
                isnan_colors = np.isnan(colors)
            
            ax.scatter(
                'dim1', 'dim2', 
                c='lightgrey',
                data=desc_df[isnan_colors],
                s=1., 
                zorder=-5,
            )
            sc = ax.scatter(
                'dim1', 'dim2', 
                c=colors[~isnan_colors],
                data=desc_df[~isnan_colors],
                s=1., 
                cmap='magma', 
                norm="log" if log_colors else None,
                zorder=10,
            )
            try:
                cb = fig.colorbar(sc, ax=ax)
            except ValueError as e:
                print_err(e)
            ax.set(title=f"{method.replace('embedding_', '')} : {desc}")
            ax.set_axis_off()

    return (fig, axes), desc_dfs


def _embedding_routine(df: Union[str, Iterable[str], DataFrame], 
                       tokenizer, #: Union[str, PreTrainedTokenizerBase], 
                       model, #: Union[str, PreTrainedModel], 
                       column: Optional[str] = None, 
                       method: Union[str, Iterable[str]] = 'mean', 
                       batch_size: int = 128,
                       plot: Optional[str] = None,
                       additional_columns: Optional[Union[str, Iterable[str]]] = None) -> Dataset:

    from transformers import AutoModelForSeq2SeqLM

    method = cast(method, to=list)
    method = [m.casefold() for m in method]

    if column is None:
        column = 'smiles'
        df = DataFrame({column: cast(df, to=list)})

    tokenizer = _load_tokenizer(tokenizer)
    if isinstance(model, str):
        print_err(f"Loading model from {model}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model)
    if torch.cuda.is_available():
        model = model.to('cuda')
    model = torch.compile(
        model, 
        fullgraph=True,
        mode="max-autotune",
    )

    ds = read_dataset(df, column=column, meta_columns=additional_columns)
    additional_columns = _check_columns(additional_columns, ds)
    smiles_values = []
    for item in ds.iter(batch_size=batch_size):
        smiles_values += item[column]

    embeddings = ds.map(
        _embed_smiles, 
        fn_kwargs={
            "column": column, 
            "tokenizer": tokenizer,
            "model": model, 
            "method": method,
        },
        batched=True, 
        batch_size=batch_size, 
        desc="Calculating embedding",
    )
    embedding_columns = [
        col for col in embeddings.column_names 
        if col.startswith('embedding')
    ]

    if plot is not None and isinstance(plot, str):
        (fig, axes), desc_dfs = plot_embedding(embeddings, embedding_columns, 
                                               column=column, 
                                               additional_columns=additional_columns)
        print_err(f"Saving embedding plot as {plot}...")
        fig.savefig(f"{plot}.png", dpi=300, bbox_inches='tight')
        for method, desc_df in desc_dfs.items():
            desc_df.to_csv(f"{plot}-{method}.csv", index=False)

    return embedding_columns, embeddings


def _flatten_embedding(
    embedding: Mapping[str, Iterable], 
    embedding_columns: Iterable[str]
) -> Dict[str, np.ndarray]:

    output = {
        f"{col_name}_{i}": col_data for col_name in embedding_columns for i, col_data in enumerate(np.asarray(embedding[col_name]).T)
    }
    
    return output


def embed_smiles_files(
    filename: str, 
    tokenizer, #: PreTrainedTokenizerBase, 
    model, #: PreTrainedModel, 
    output: Optional[str] = None,
    column: str = 'smiles', 
    method: Union[str, Iterable[str]] = 'mean', 
    batch_size: int = 128,
    plot: Optional[str] = None,
    additional_columns: Optional[Union[str, Iterable[str]]] = None
) -> str:

    """

    """

    if output is None:
        output = sys.stdout

    embedding_columns, embeddings = _embedding_routine(
        df=filename, 
        column=column, 
        tokenizer=tokenizer, 
        model=model, 
        method=method, 
        batch_size=batch_size,
        plot=plot,
        additional_columns=additional_columns,
    )

    embeddings = (
        embeddings#.to_iterable_dataset()
        .map(
            _flatten_embedding,
            fn_kwargs={
                "embedding_columns": embedding_columns,
            },
            remove_columns=embedding_columns,
            batched=True, 
            batch_size=batch_size,
            desc="Flattening embedding",
        )
    )

    print_err(f"Writing featurized {column} to {output}")
    try:
        for i, chunk in tenumerate(embeddings.iter(batch_size=batch_size)):
            df = DataFrame(chunk)
            df.to_csv(output, 
                      index=False, 
                      header=i==0,
                      mode='w' if i == 0 else 'a')
    except BrokenPipeError:
        pass

    return filename
    


    