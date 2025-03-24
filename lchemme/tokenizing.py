"""Utilities for tokenizing chemical datasets."""

from typing import Dict, Iterable, Optional, Union

from functools import partial

from carabiner import print_err
from pandas import DataFrame
from tokenizers import Tokenizer, decoders, models, processors, trainers
# from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast
from tqdm.auto import trange

from .io import read_dataset

_MODELS = dict(wordpiece=(partial(models.WordPiece, max_input_chars_per_word=500), 
                          trainers.WordPieceTrainer, decoders.WordPiece),
               bpe=(models.BPE, trainers.BpeTrainer, decoders.BPEDecoder))


def _get_training_corpus(ds, 
                         column: str,
                         batch_size: int = 2000) -> Iterable[list]:

    for batch in (ds
                  .filter(lambda x: x[column] is not None and x[column] != "")
                  .iter(batch_size=batch_size)):
        
        yield batch[column]


def tokenize(df: Union[str, DataFrame],
             column: str,
             checkpoint: str,
             vocab_size: int = 512,
             filename: Optional[str] = None,
             hub_repo: Optional[str] = None,
             batch_size: Optional[int] = None,
             tokenizer : str = 'WordPiece'):

    """Tokenize the column of a Pandas DataFrame.

    Currently only supports WordPiece model and BERT processing.

    Parameters
    ----------
    df : str, pd.DataFrame
        Path to CSV, or a Pandas DataFrame containing a column to tokenize.
    column : str
        Name of column to tokenize.
    checkpoint : str
        Model checkpoint name to use as basis for tokenizer.
    vocab_size : int, optional
        Size of tokenizer vocabulary. Default: 512.
    filename : str, optional
        Filename to save tokenizer to. Default: do not save.
    hub_repo : str, optional
        Not implemented. Repository to push to.
    
    """

    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if isinstance(df, str):
        print_err(f"Loading dataset from {df}")
    ds = read_dataset(df, column)
    n_rows = ds.num_rows
    ds = ds.to_iterable_dataset()

    print_err(f"Loading tokenizer parameters from {checkpoint}")
    tokenizer_structure = AutoTokenizer.from_pretrained(checkpoint)

    print_err(f"Setting up tokenization with {tokenizer}")
    _model, _trainer, _decoder = _MODELS[tokenizer.casefold()]
    
    # wordpiece_model = models.WordPiece(unk_token=tokenizer_structure.special_tokens_map['unk_token'],
    #                                    max_input_chars_per_word=500)
    wordpiece_model = _model(unk_token=tokenizer_structure.special_tokens_map['unk_token'])
    slow_tokenizer = Tokenizer(wordpiece_model)

    special_tokens = [item for item in list(tokenizer_structure.special_tokens_map_extended.values())
                      if isinstance(item, str)]
    slow_tokenizer.add_special_tokens(special_tokens)

    post_process_args = {key: tokenizer_structure.special_tokens_map[f'{key}_token'] 
                        for key in ('sep', 'cls')}
    post_process_args = {key: (val, slow_tokenizer.token_to_id(val))
                        for key, val in post_process_args.items()}
    slow_tokenizer.post_processor = processors.BertProcessing(**post_process_args)
    slow_tokenizer.decoder = _decoder()

    trainer = _trainer(vocab_size=vocab_size, 
                       special_tokens=special_tokens,
                       show_progress=True)
    print_err(f"Training tokenizer with vocab size {vocab_size}")
    slow_tokenizer.train_from_iterator(_get_training_corpus(ds, column), 
                                       trainer=trainer,
                                       length=n_rows)

    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=slow_tokenizer,
                                             **tokenizer_structure.special_tokens_map)

    if filename is not None:
        print_err(f"Saving pretrained tokenizer as {filename}")
        fast_tokenizer.save_pretrained(filename)

    if hub_repo is not None:
        print_err("WARNING: Pushing to Hub is not yet implemented. Skipping.")

    return fast_tokenizer