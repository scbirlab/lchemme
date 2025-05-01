"""Utilities for pre-training chemical language models."""

from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Union

from math import ceil
import multiprocessing
import os

from carabiner import print_err
from datasets import Dataset, DatasetDict

if TYPE_CHECKING:
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        PreTrainedTokenizerFast
    )
else:
    AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerFast = Any, Any, Any

from .datasets import _random_smiles


_DEFAULT_BATCH_SIZE = 128
_DEFAULT_SEED = 0


def _load_train_test(
    train: Union[str, Dataset, DatasetDict],
    column: Optional[str],
    test: Optional[Union[str, Dataset, float]] = None
):

    from .io import read_dataset

    if isinstance(train, str):
        print_err(f"Loading data from {train}")
        train = read_dataset(train, column)

    if isinstance(test, str):
        print_err(f"Loading data from {test}")
        test = read_dataset(test, column)

    if isinstance(train, DatasetDict) and all(key in train for key in ('train', 'test')):
        ds_train, ds_test = (train[key] for key in ('train', 'test'))
        if test is not None:
            print_err("WARNING: Dataset contains train/test splits; ignoring supplied separate test data.")
    elif isinstance(train, Dataset):
        if isinstance(test, float) and test > 0.:
            train = train.train_test_split(test_size=test)
            ds_train, ds_test = (train[key] for key in ('train', 'test'))
        elif isinstance(test, Dataset) or test is None:
            ds_train, ds_test = train, test
        elif test == 0.:
            ds_train, ds_test = train, None
        elif test is not None:
            raise ValueError("Supplied test data must be a float for splitting or a HuggingFace Dataset.")
    else:
        raise ValueError("Supplied test data must be a path to CSV, a Pandas DataFrame, or a HuggingFace Dataset.")

    max_test_examples = 100_000
    if ds_test is not None:
        test_n = ds_test.num_rows
        if test_n > max_test_examples:
            print_err(f"Test data contains {test_n} rows; downsampling to {max_test_examples}.")
            ds_test = ds_test.shuffle(seed=1).take(max_test_examples)

    return ds_train, ds_test


def _load_tokenizer(
    tokenizer: Optional[Union[PreTrainedTokenizerFast, str]] = None,
    checkpoint: Optional[str] = None
) -> PreTrainedTokenizerFast:
    
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if tokenizer is None:
        print_err(f"Loading tokenizer from {checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    elif isinstance(tokenizer, str):
        print_err(f"Loading tokenizer from {tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    elif not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError("Supplied tokenizer must be a path to a saved tokenizer or PreTrainedTokenizerFast object")

    return tokenizer


def chinchilla_token_number(model) -> int:
    ##  20 tokens per parameter
    return model.num_parameters() * 20


def _get_token_ids(tokenizer: PreTrainedTokenizerFast) -> Dict[str, int]:

    token_ids = [f"{special_token.split('_token')[0]}_token_id" 
                 for special_token in tokenizer.special_tokens_map]
    token_ids = {token_id: getattr(tokenizer, token_id) 
                 for token_id in token_ids}

    return token_ids


def dataset_token_number(
    dataset: Dataset, 
    tokenizer: PreTrainedTokenizerFast,
    nrows: Optional[int] = None
) -> int:

    max_rows = 100_000
    
    try:
        total_rows = dataset.num_rows
    except Exception:
        if nrows is not None:
            total_rows = nrows
        else:
            raise ValueError("Dataset is Iterable and number of rows not provided.")

    sample_fraction = max_rows / total_rows
    dataset_sample = dataset.shuffle(seed=1).take(min(max_rows,total_rows))

    token_ids = _get_token_ids(tokenizer)

    return ceil(
        sum(
            len([
                _id for _id in item['input_ids'] 
                if _id not in token_ids.values()
            ]) for item in dataset_sample
        ) 
        / sample_fraction
    )


def ideal_epochs(
    model, 
    dataset: Dataset, 
    tokenizer: PreTrainedTokenizerFast,
    nrows: Optional[int] = None
) -> int:

    """Chinchilla estimate of the ideal number of training epochs.

    Examples
    ========
    >>> from datasets import Dataset
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    >>> from .datasets import _random_smiles
    >>> tok = AutoTokenizer.from_pretrained("sshleifer/bart-tiny-random")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/bart-tiny-random")
    >>> fn = _random_smiles(column="smiles", tokenizer=tok)
    >>> dummy_ds = Dataset.from_dict({"smiles": ["C"] * 10}).map(fn, batched=True)
    >>> ideal_epochs(model, dummy_ds, tok, nrows=10) > 0
    True
    
    """

    n_tokens_to_train = chinchilla_token_number(model)
    tokens_per_ds = dataset_token_number(dataset, tokenizer, nrows)
    
    return n_tokens_to_train / tokens_per_ds


def max_epochs_per_time(max_time: float, 
                        n_rows: int) -> int:

    """Number of epochs that can be run in a period of time.
    
    """

    mins_per_row = (3. * 60.) / (13158 * 256)
    n_rows_in_time = int(max_time // mins_per_row)
    return n_rows / n_rows_in_time


def _get_steps_per_epoch(
    n_rows: int, 
    batch_size: int
) -> int:

    """Wrapper around ceiling divide.

    Examples
    ========
    >>> from datasets import Dataset
    >>> dummy_ds = Dataset.from_dict({"smiles": ["C"] * 10})
    >>> _get_steps_per_epoch(10, batch_size=4)  # ceiling divide
    3

    """


    return ceil(n_rows / batch_size)


def _load_new_dataset(
    train: Union[str, Dataset, DatasetDict],
    column: Optional[str],
    tokenizer: PreTrainedTokenizerFast,
    test: Optional[Union[str, Dataset, float]] = None, 
    batch_size: int = _DEFAULT_BATCH_SIZE,
    seed: int = _DEFAULT_SEED,
):
    ds_train, ds_test = _load_train_test(train, column, test)
    permutation_function = _random_smiles(column, tokenizer)

    def _filter_fn(x):
        return x[column] is not None and x[column] != ""
    ds_train = (
        ds_train
        .filter(_filter_fn)
        .to_iterable_dataset(
            num_shards=256,
        )
        .shuffle(
            seed=seed, 
            buffer_size=1024,
        )
        .map(
            permutation_function, 
            batched=True, 
            batch_size=batch_size,
        )
    )
    if ds_test is not None:
        ds_test = (
            ds_test
            .filter(_filter_fn)
            .to_iterable_dataset(
                num_shards=256,
            )
            .map(
                permutation_function, 
                batched=True, 
                batch_size=batch_size,
            )
        )
    # print(ds_train, ds_test)
    return ds_train, ds_test

def pretrain(
    checkpoint: str,
    output: str,
    train: Optional[Union[str, Dataset, DatasetDict]] = None,
    column: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
    test: Optional[Union[str, Dataset, float]] = None, 
    reinitialize_before_training: bool = True,
    resume_training: bool = False,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    n_epochs: Optional[float] = None,
    max_time: Optional[float] = None,
    early_stopping: Optional[int] = None,
    plot_filename: Optional[str] = None,
    max_learning_rate: float = 1e-4,
    warmup_steps: int = 10_000,
    weight_decay: float = .01,
    seed: int = _DEFAULT_SEED,
    no_compile: bool = False
) -> AutoModelForSeq2SeqLM:

    """Pretrain a BART model on SMILES canonicalization.
    
    """

    from datasets import Dataset, IterableDataset
    import torch
    # torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch._dynamo import config as dynamo_config
    dynamo_config.suppress_errors = True
    dynamo_config.capture_scalar_outputs = True
    from transformers import (
        AutoConfig, 
        AutoModelForSeq2SeqLM,
        DataCollatorForSeq2Seq, 
        EarlyStoppingCallback,
        TrainerCallback,
        Seq2SeqTrainer, 
        Seq2SeqTrainingArguments,
        PreTrainedTokenizerFast,
    )
    from .callbacks import PlotterCallback, SaveDatasetStateCallback, _plot_history, dataset_state_dict_loader

    tokenizer = _load_tokenizer(tokenizer, checkpoint)
    ds_train, ds_test = _load_new_dataset(
        train=train,
        column=column,
        test=test, 
        tokenizer=tokenizer,
    )
    # print_err(ds_train)
    if resume_training:
        ds_train, n_training_rows = dataset_state_dict_loader(checkpoint, ds_train)
    else:
        if isinstance(ds_train, Dataset):
            n_training_rows = ds_train.num_rows
        elif isinstance(ds_train, IterableDataset):
            for n_training_rows, _ in enumerate(ds_train):
                pass
        else:
            raise TypeError("Training data should be a Hugging Face dataset.")
        print_err(f"Dataset contains ~{dataset_token_number(ds_train, tokenizer, nrows=n_training_rows)} tokens")

    if reinitialize_before_training and not resume_training:
        print_err(f"Loading model architecture from {checkpoint} with vocab size {tokenizer.vocab_size}")
        config = AutoConfig.from_pretrained(
            checkpoint,
            vocab_size=tokenizer.vocab_size,
            **_get_token_ids(tokenizer),
        )
        model = AutoModelForSeq2SeqLM.from_config(config=config)
    else:
        print_err(f"Loading pretrained model from {checkpoint}")
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    print_err(f"Loaded model from {checkpoint}; it has {model.num_parameters()} parameters")

    if not resume_training:
        ideal_n_epochs = ideal_epochs(
            model, 
            ds_train, 
            tokenizer, 
            nrows=n_training_rows,
        )
        print_err(f"Ideal number of epochs (based on Chinchilla estimate) is {ideal_n_epochs}")
    else:
        ideal_n_epochs = 1.

    if max_time is not None:
        n_epochs = max_epochs_per_time(max_time, n_training_rows)
        print_err(f"Maximum time is {max_time} minutes")
    elif n_epochs is None or n_epochs == 0:
        n_epochs = ideal_n_epochs
        
    print_err(f"Training will have {n_epochs:.2f} epochs")

    ncpus = min(4, multiprocessing.cpu_count() - 1)  # TODO: This doesn't seem to work well on Crick Slurm cluster
    
    
    steps_per_epoch = _get_steps_per_epoch(n_training_rows, batch_size)
    max_training_steps = ceil(n_epochs * n_training_rows / batch_size)
    print_err(f"Each epoch has {steps_per_epoch} steps, with the entire training taking {max_training_steps} steps")

    eval_steps = min(10_000, steps_per_epoch) if ds_test is not None else None
    save_steps = (50 * eval_steps) if ds_test is not None else min(50_000, steps_per_epoch)
    training_args = Seq2SeqTrainingArguments(
        torch_compile=not no_compile,
        tf32=torch.cuda.is_available(),
        # bf16=True,
        # optim="adamw_bnb_8bit", 
        output_dir=output,
        resume_from_checkpoint=checkpoint if resume_training else None,
        save_strategy="steps",
        save_steps=save_steps,
        overwrite_output_dir=True,
        max_steps=max_training_steps,
        logging_strategy="steps",
        logging_steps=min(500, steps_per_epoch),
        learning_rate=max_learning_rate,
        warmup_steps=warmup_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, # if ds_test is not None else None,
        weight_decay=weight_decay,
        report_to=["tensorboard"],
        log_level="error",
        logging_dir=os.path.join(output, "logs"),
        eval_strategy="steps" if ds_test is not None else "no",
        eval_steps=eval_steps,
        eval_on_start=True,
        load_best_model_at_end=ds_test is not None,
        dataloader_num_workers=ncpus,
    )

    tokenizer = _load_tokenizer(tokenizer, checkpoint)  # reload to stop warnings about forking
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model,
        padding=True,
    )
    
    tokenizer = _load_tokenizer(tokenizer, checkpoint)  # reload to stop warnings about forking
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.add_callback(SaveDatasetStateCallback(trainer, num_rows=n_training_rows))
    if ds_test is not None and early_stopping is not None:
        print_err(f"Setting early stopping patience to {early_stopping} evaluation steps")
        trainer.add_callback(
            EarlyStoppingCallback(early_stopping_patience=early_stopping)
        )
    else:
        print_err("No early stopping.")
    if plot_filename is not None:
        trainer.add_callback(PlotterCallback(plot_filename))

    trainer.train()
    trainer.save_model(output)

    if plot_filename is not None:
        print_err(f"Saving training results in files with prefix {plot_filename}")
        _plot_history(trainer.state, plot_filename)
        
    return model