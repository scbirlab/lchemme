"""Utilities for reading and writing."""

from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Union

import os

from carabiner import print_err
from carabiner.cast import cast

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset
    from pandas import DataFrame
else:
    Dataset, DatasetDict, IterableDataset, DataFrame = Any, Any, Any, Any
from numpy.typing import ArrayLike


DataLike = Union[
    str, 
    DataFrame, 
    Mapping[str, ArrayLike], 
    Dataset, 
    IterableDataset
]

_HF_PREFIX = "hf://"


def _resolve_hf_hub_dataset(
    ref: str, 
    cache: str,
) -> Dataset:
    from datasets import load_dataset

    hf_ref_full = ref.split("hf://")[-1]
    hf_ref = hf_ref_full.split("@")[0] if "@" in ref else hf_ref_full
    if ":" in hf_ref_full:
        ds_config, ds_split = hf_ref_full.split("@")[-1].split(":")[:2]
    else:
        ds_config, ds_split = hf_ref_full.split("@")[-1], "train"
    return load_dataset(hf_ref, ds_config, split=ds_split, cache_dir=cache)


def _load_from_csv(
    filename: str,
    cache: Optional[str] = None
) -> Dataset:
    from datasets import Dataset
    return Dataset.from_csv(
        filename, 
        cache_dir=cache, 
        sep="," if filename.endswith((".csv", ".csv.gz")) else "\t",
    )


def _load_from_dataframe(
    dataframe: Union[DataFrame, Mapping[str, ArrayLike]],
    cache: Optional[str] = None
) -> Dataset:

    from datasets.fingerprint import Hasher
    from pandas import DataFrame

    if cache is None:
        cache = "cache"
        print_err(f"Defaulting to cache: {cache}")
    if not isinstance(dataframe, DataFrame) and isinstance(dataframe, Mapping):
        dataframe = DataFrame(dataframe)

    hash_name = Hasher.hash(dataframe)
    df_temp_file = os.path.join(cache, "df-load", f"{hash_name}.csv")
    df_temp_dir = os.path.dirname(df_temp_file)
    if not os.path.exists(df_temp_dir):
        os.makedirs(df_temp_dir)

    if not os.path.exists(df_temp_file):
        print_err(f"Caching dataframe at {df_temp_file}")
        dataframe.to_csv(df_temp_file, index=False)

    print_err(f"Reloading dataframe from {df_temp_file}")
    return _load_from_csv(
        df_temp_file, 
        cache=cache,
    )

    
def _resolve_data(
    data: DataLike, 
    cache: Optional[str] = None
) -> Union[Dataset, IterableDataset]:
    from datasets import Dataset, IterableDataset
    from pandas import DataFrame

    if isinstance(data, (Dataset, IterableDataset)):
        dataset = data
    elif isinstance(data, (DataFrame, Mapping)):
        dataset = _load_from_dataframe(
            data, 
            cache=cache,
        )
    elif isinstance(data, str):
        if data.startswith(_HF_PREFIX):
            dataset = _resolve_hf_hub_dataset(
                data,
                cache=cache,
            )
        else:
            dataset = _load_from_csv(
                data, 
                cache=cache,
            )
    else:
        raise ValueError(
            """
            Data must be a string, Dataset, dictionary, or Pandas DataFrame.
            """
        )
    return dataset


def read_dataset(
    df: DataLike,
    column: Optional[str] = None,
    meta_columns: Optional[Union[str, Iterable[str]]] = None,
    split: Optional[float] = None,
    cache: Optional[str] = None
) -> Union[Dataset, DatasetDict]:

    """Read a CSV path or load a Pandas DataFrame as a Huggingface Dataset.
    
    """

    ds = _resolve_data(
        data=df, 
        cache=cache,
    )

    if column is not None:
        columns = [column]
    else:
        columns = []

    if meta_columns is not None:
        meta_columns = cast(meta_columns, to=list)
        columns += meta_columns

    if len(columns) > 0:
        ds = ds.select_columns(columns)
    
    if split is not None:
        if not isinstance(split, float):
            raise TypeError(f"Split fraction must be a float: {split}")
        print_err(f"Splitting dataset into {split * 100.} % testing.")
        ds = ds.train_test_split(test_size=split)
    
    return ds#.take(10_000)

    
