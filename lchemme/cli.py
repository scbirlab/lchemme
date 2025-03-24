"""Command-line interface for lchemmy."""

from typing import Any, Dict, List, Optional

from argparse import ArgumentParser, FileType, Namespace
from collections import Counter
from functools import partial
import os
import sys

from carabiner import cast, pprint_dict, print_err, upper_and_lower
from carabiner.cliutils import clicommand, CLIOption, CLICommand, CLIApp
from carabiner.pd import get_formats

from .tokenizing import _MODELS, tokenize
from .featurizing import _FEATURE_METHODS

__version__ = '0.0.1'

def _option_parser(x: Optional[List[str]]) -> Dict[str, Any]:

    options = {}

    try:
        for opt in x:

            try:
                key, value = opt.split('=')
            except ValueError:
                raise ValueError(f"Option {opt} is misformatted. It should be in the format keyword=value.")
            
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass

            options[key] = value

    except TypeError:
        
        pass

    return options


def _sum_tally(tallies: Counter, 
               message: str = "Error counts",
               use_length: bool = False):

    total_tally = Counter()
    
    for tally in tallies:

        if use_length:
            total_tally.update({key: len(value) for key, value in tally.items()})
        else:
            total_tally.update(tally)

    pprint_dict(total_tally, message=message)

    return total_tally


@clicommand(message="Tokenizing with the following parameters")
def _tokenize(args: Namespace) -> None:

    tokenizer = tokenize(args.input,
                         column=args.column,
                         checkpoint=args.model,
                         vocab_size=args.vocab_size,
                         filename=args.output,
                         tokenizer=args.piecer,
                         hub_repo=args.repo)
    
    return None


@clicommand(message="Training with the following parameters")
def _pretrain(args: Namespace) -> None:

    from .pretraining import pretrain

    try:
        test = float(args.test)
    except (TypeError, ValueError):
        test = args.test
    
    model = pretrain(args.input,
                     column=args.column,
                     checkpoint=args.model,
                     output=args.output,
                     tokenizer=args.tokenizer,
                     reinitialize_before_training=not args.resume,
                     test=test, 
                     batch_size=args.batch_size,
                     n_epochs=args.epochs,
                     early_stopping=args.early_stopping,
                     max_time=args.max_time,
                     plot_filename=args.plot,
                     max_learning_rate=1e-4,
                     warmup_steps=10_000,
                     weight_decay=.01)

    return None


@clicommand(message="Featurizing with the following parameters")
def _featurize(args: Namespace) -> None:

    from .featurizing import embed_smiles_files

    if args.tokenizer is None:
        tokenizer = args.model
    else:
        tokenizer = args.tokenizer

    output_filename = embed_smiles_files(
        filename=args.input, 
        tokenizer=tokenizer, 
        model=args.model, 
        output=args.output,
        column=args.column, 
        method=args.method, 
        batch_size=args.batch_size,
        plot=args.plot,
        additional_columns=args.extras,
    )
    
    return None


def main() -> None:

    input_stream = CLIOption('input', 
                             default=sys.stdin,
                             type=FileType('r'), 
                             nargs='?',
                             help='Input columnar Excel, CSV or TSV file. Default: STDIN.')
    inputs = CLIOption('input', 
                       type=str, 
                       help='Input columnar Excel, CSV or TSV file.')
    test_input = CLIOption('--test', '-2',
                           type=str, 
                           default=None,
                           help='Test / validation data. Either columnar Excel, CSV or TSV file, or '
                                'a number indicating the random proportion of input table to take as test data. '
                                'Default: Don\'t use test data.')
    piecer = CLIOption('--piecer', 
                        type=str, 
                        default='WordPiece',
                        choices=upper_and_lower(_MODELS),
                        help='Tokenization strategy. WordPiece requires more memory, but results in faster model pretraining.')

    epochs = CLIOption('--epochs', '-n',
                        type=float, 
                        default=None,
                        help='Number of epochs to use for training. Default: if --epochs or '
                             '--max-time unspecified, use Chinchilla optimum.')
    batch_size = CLIOption('--batch-size', '-b',
                           type=int, 
                           default=256,
                           help='Number of observations per batch.')
    feature_method = CLIOption('--method',
                               type=str, 
                               nargs="*",
                               default="mean",
                               choices=upper_and_lower(_FEATURE_METHODS),
                               help='Method of emebdding generation.')
    extras = CLIOption('--extras',
                       type=str, 
                       nargs="*",
                       default=None,
                       help='Extra values to plot. Must be column names from input.')
    max_time = CLIOption('--max-time', '-t',
                         type=float, 
                         default=None,
                         help='Maximumum running time, to estimate the maximum number of training epochs. '
                              'Default: if --epochs unspecified, use Chinchilla optimum.')
    early_stopping = CLIOption('--early-stopping', '-y',
                               type=int, 
                               default=None,
                               help='Number of evaluation steps to wait for improved test loss before stopping. '
                                    'Only used if test data are present through --test.'
                                    'Default: Don\'t stop early.')
    resume = CLIOption(
        '--resume',
        action='store_true',
        help='Whether to continue training, or reinitialize model parameters (the default).',
    )
    tokenizer_ = CLIOption('--tokenizer', '-k', 
                           type=str, 
                           default=None,
                           help='Path to saved Huggingface tokenizer or Huggingface checkpoint containing a tokenizer. '
                                'Default: use model tokenizer.')
    column = CLIOption('--column', '-c', 
                       default='smiles',
                       type=str,
                       help='Column to use as input string representation. ')
    vocab_size = CLIOption('--vocab-size', '-z', 
                           default=512,
                           type=int,
                           help='Vocab size of language model.')
    repo = CLIOption('--repo', 
                     type=str,
                     default=None,
                     help='Not implemented. Huggingface Hub repository to upload to. Default: Don\'t upload.')  
    output_stream = CLIOption('--output', '-o', 
                              type=FileType('w'),
                              default=sys.stdout,
                              help='Output file. Default: STDOUT')    
    output = CLIOption('--output', '-o', 
                       type=str,
                       required=True,
                       help='Output file path.')  
    plot_ = CLIOption('--plot', '-g', 
                       type=str,
                       default=None,
                       help='If provided, save plots with this prefix. Default: don\'t plot.')                   
    checkpoint = CLIOption('--model', '-m', 
                           type=str,
                           required=True,
                           help='Model checkpoint to use. Provide either a built-in model name or a reference from https://huggingface.co/models')

    tokenize = CLICommand('tokenize', 
                          description='Tokenize the data inputs.',
                          main=_tokenize,
                          options=[inputs, output, column, checkpoint, repo, vocab_size, piecer])
    pretrain = CLICommand(
        'pretrain', 
        description='Pre-train a large language model using self-supervised learning.',
        main=_pretrain,
        options=[
            inputs, 
            test_input, 
            output, 
            column, 
            checkpoint, 
            # repo, 
            tokenizer_, 
            plot_, 
            epochs, 
            early_stopping, 
            batch_size, 
            max_time, 
            resume,
            ],
        )
    featurize = CLICommand('featurize', 
                           description='Get vector embeddings of a chemical dataset using a pre-trained large language model.',
                           main=_featurize,
                           options=[inputs, output_stream, column, feature_method, checkpoint, tokenizer_,
                                    batch_size, plot_, extras])

    app = CLIApp(
        "lchemmy",
        version=__version__,
        description="Training and applying large chemistry models.",
        commands=[tokenize, pretrain, featurize],
    )

    app.run()

    return None


if __name__ == "__main__":

    main()