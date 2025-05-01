"""Hugging Face training callbacks."""

from glob import glob
import os

from carabiner import print_err
from carabiner.mpl import grid, colorblind_palette
from datasets import Dataset, IterableDataset
import torch
from transformers import TrainerCallback


def _plot_history(
    trainer_state, 
    filename: str,
) -> None:

    from numpy import nanmean
    from pandas import DataFrame

    data_to_plot = (
        DataFrame(trainer_state.log_history)
        .groupby('step')
        .agg("mean")
        .reset_index()
    )
    data_to_plot.to_csv(
        filename + '.csv',
        index=False,
    )   

    fig, ax = grid()
    for _y in ('eval_loss', 'loss'):
        if _y in data_to_plot:
            ax.plot(
                'step', _y, 
                data=data_to_plot, 
                label=_y,
            )
            ax.scatter(
                'step', _y, 
                data=data_to_plot,
                s=1.,
            )
    ax.legend()
    ax.set(xlabel='Training step', ylabel='Loss', yscale='log')
    fig.savefig(filename + '.png', dpi=600, bbox_inches='tight')

    return None


class PlotterCallback(TrainerCallback):

    """Save a PNG of training progress when logging.

    """

    def __init__(self, filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename

    def on_log(self, args, state, control, **kwargs):
        _plot_history(state, self.filename)

    def on_save(self, args, state, control, **kwargs):
        self.on_log(args, state, control, **kwargs)


class SaveDatasetStateCallback(TrainerCallback):

    """Checkpoint the state of a Huggingface Dataset object for 
    resuming training.

    """

    _training_data_path = lambda d: os.path.join(d, "training-data.hf")
    _eval_data_path = lambda d: os.path.join(d, "eval-data.hf")
    _training_state_path = lambda d: os.path.join(d, "dataset_state_{}.pt")

    def __init__(self, trainer, num_rows: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = trainer
        self.num_rows = num_rows

    def on_save(self, args, state, control, **kwargs):
        train_ds = self.trainer.train_dataset
        eval_ds = self.trainer.eval_dataset
        train_out = self.__class__._training_data_path(args.output_dir)
        if isinstance(train_ds, Dataset):
            print_err(f"INFO: Saving training data at {args.output_dir}")
            train_ds.save_to_disk(train_out)
        elif isinstance(train_ds, IterableDataset):
            print_err(f"INFO: Saving training data state at {args.output_dir}")
            # if state.global_step < state.max_steps and args.process_index == 0:
            state_dict = {
                "seen": state.global_step * args.per_device_train_batch_size * args.world_size,
                "cursor": train_ds.state_dict(),
                "num_rows": self.num_rows,
            }
            torch.save(
                state_dict,
                self.__class__._training_state_path(args.output_dir).format(state.global_step)
            )
        else:
            print_err(f"WARNING: Not saving training dataset")

        if eval_ds is not None:
            print_err(f"INFO: Saving eval data at {args.output_dir}")
            eval_out = self.__class__._eval_data_path(args.output_dir)
            if isinstance(eval_ds, Dataset):
                train_ds.save_to_disk(train_out)
            else:
                print_err(f"WARNING: Not saving eval dataset")

# load (called by trainer when you pass --resume or resume_from_checkpoint)
def dataset_state_dict_loader(checkpoint, dataset):
    num_rows = None
    if checkpoint is not None:
        print_err(f"INFO: Loading dataset state from '{checkpoint}'")
        dataset_checkpoints = glob(
            SaveDatasetStateCallback._training_state_path(checkpoint)
            .format("*")
        )
        if len(dataset_checkpoints) > 0:
            latest = max(
                dataset_checkpoints,
                key=lambda p: int(p.split("_")[-1].split(".pt")[0])
            )
            if isinstance(dataset, IterableDataset):
                state = torch.load(latest)
                num_rows = state["num_rows"]
                seen = state["seen"]
                print_err(
                    f"INFO: dataset state at {latest} had reached row {seen} / {num_rows}."
                )
                for i, _ in enumerate(dataset):
                    if i == seen:
                        break
            else:
                print_err(
                    f"""
                    WARNING: Training dataset is not IterableDataset, was {type(dataset)}. 
                    Cannot load state from '{checkpoint}'.
                    """
                )
        else:
            print_err(f"WARNING: Could not load dataset state from '{checkpoint}'.")
    return dataset, num_rows
        
