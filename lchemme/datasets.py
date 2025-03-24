"""Utilities for making large language chemical datasets."""

from typing import Dict, Iterable, List, Mapping, Optional, Union

from carabiner import print_err
from datasets import Dataset, DatasetDict
from pandas import DataFrame
from schemist.converting import convert_string_representation
from transformers import PreTrainedTokenizerFast
from rdkit.Chem import MolFromSmiles, MolToSmiles

def _prepare_data(x: Mapping[str, Iterable[str]],                   
                  column: str,
                  permuted_column: str,
                  tokenizer: PreTrainedTokenizerFast) -> Dict[str, List[str]]:

    inputs = tokenizer(x[permuted_column], 
                       return_special_tokens_mask=True, 
                       padding=True,
                       text_target=x[column])

    if len(inputs['input_ids']) != len(inputs['labels']):
        for i, (_input, _label) in enumerate(zip(x[column], x[permuted_column])):
            print_err(f"{i} : {_input} | {_label}")
        raise ValueError(f"Inputs and labels have differing lengths.")

    return inputs


def _random_smiles(column: str,
                   tokenizer: PreTrainedTokenizerFast,
                   permuted_column: Optional[str] = None,
                   unknown_token: str = '<unk>'):

    if permuted_column is None:
        permuted_column = f"permuted_{column}"

    def _cleaner(x: str):
        if x is not None and x != "":
            return x
        else:
            return unknown_token

    def random_smiles(x: Mapping[str, Iterable[str]]) -> Dict[str, List[str]]:

        results = convert_string_representation(x[column], 
                                                output_representation=['smiles', 'permuted_smiles'])

        x[column] = list(map(_cleaner, results['smiles']))
        x[permuted_column] = list(map(_cleaner, results['permuted_smiles']))

        return _prepare_data(x, 
                             column=column, 
                             permuted_column=permuted_column, 
                             tokenizer=tokenizer)

    return random_smiles


def _load_from_csv(_class, _df):
    return _class.from_csv(_df) 


def _load_from_dict(_class, _df):
    return _class.from_dict(_df) 


def dataset_from_dataframe(df: Union[DataFrame, str], 
                           df_test: Optional[Union[DataFrame, str]] = None,
                           column: str = 'smiles',
                           batch_size: int = 1000,
                           filename: Optional[str] = None) -> Union[Dataset, DatasetDict]:

    """Load a dataset from a Pandas DataFrame and prepare for pretraining.
    
    """

    if df_test is not None:
        ds_class = DatasetDict
        if isinstance(df, str):
            df = dict(train=df, 
                      test=df_test)
            loader = _load_from_csv
        else:
            df = dict(train=df.to_dict("list"), 
                      test=df_test.to_dict("list"))
            loader = _load_from_dict
    else:
        ds_class = Dataset
        if isinstance(df, str):
            loader = _load_from_csv
        else:
            df = df.to_dict("list")
            loader = _load_from_dict
    
    ds = (loader(ds_class, df)
          .select_columns(column)
          .map(_random_smiles(column), 
               batched=True, 
               batch_size=batch_size))

    if filename is not None:
        ds.save_to_disk(filename)
    
    return ds