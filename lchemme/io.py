"""Utilities for reading and writing."""

from typing import Iterable, Optional, Union

from carabiner import print_err
from carabiner.cast import cast

from datasets import load_dataset, Dataset, DatasetDict, IterableDataset
from pandas import DataFrame

def read_dataset(
    df: Union[str, DataFrame],
    column: Optional[str] = None,
    meta_columns: Optional[Union[str, Iterable[str]]] = None,
    split: Optional[float] = None,
    in_memory: bool = False
) -> Union[Dataset, DatasetDict]:

    """Read a CSV path or load a Pandas DataFrame as a Huggingface Dataset.
    
    """

    if in_memory:
        ds_class = Dataset
    else:
        ds_class = Dataset #IterableDataset

    if column is not None:
        columns = [column]
    else:
        columns = []

    if meta_columns is not None:
        meta_columns = cast(meta_columns, to=list)
        columns += meta_columns

    if isinstance(df, str):
        print_err(f"Loading dataset from {df}")
        if not df.endswith('.csv') and not df.endswith('.csv.gz'):
            ds = load_dataset(df)
        else:
            ds = ds_class.from_csv(df)
    elif isinstance(df, DataFrame):
        missing_cols = [col for col in columns if col not in df]
        if len(missing_cols) > 0:
            raise KeyError("Columns missing from input data: " + ", ".join(missing_cols))
        ds = ds_class.from_dict(df[columns].todict("list"))
    else:
        raise TypeError("Input to read_dataset must be a str or a DataFrame.")

    if len(columns) > 0:
        ds = ds.select_columns(columns)
    
    if split is not None:
        if not isinstance(split, float):
            raise TypeError(f"Split fraction must be a float: {split}")
        print_err(f"Splitting dataset into {split * 100.} % testing.")
        ds = ds.train_test_split(test_size=split)
    
    return ds#.take(10_000)

    
