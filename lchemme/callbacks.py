"""Hugging Face training callbacks."""

from glob import glob
import os

from carabiner.mpl import grid, colorblind_palette
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
        .agg(nanmean)
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

    Not yet implemented.

    """
    # TODO: Wait for sateful dataloader support https://github.com/huggingface/transformers/pull/34205
    def __init__(self, filename, dataset_attr: str = "train_dataset", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename
        self.dataset_attr = dataset_attr

    def on_save(self, args, state, control, **kwargs):
        ds = getattr(kwargs["trainer"], self.dataset_attr)
        torch.save(
            ds.state_dict(),
            os.path.join(args.output_dir, f"dataset_state_{state.global_step}.pt")
        )

    # load (called by trainer when you pass --resume or resume_from_checkpoint)
    def on_load(self, args, state, **kwargs):
        if args.resume_from_checkpoint is not None:
            ckpt_dir = args.resume_from_checkpoint
            latest = max(
                (p for p in glob(os.path.join(ckpt_dir, "dataset_state_*.pt"))),
                key=lambda p: int(p.stem.split("_")[-1])
            )
            ds = getattr(kwargs["trainer"], self.dataset_attr)
            ds.load_state_dict(torch.load(latest))
