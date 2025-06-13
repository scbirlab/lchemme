# 🚄 lchemme

![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/scbirlab/lchemme/python-publish.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lchemme)
![PyPI](https://img.shields.io/pypi/v/lchemme)

Pretraining large chemistry models for embedding.

- [Installation](#installation)
- [Command-line usage](#command-line-usage)
    - [Example](#example)
    - [Other commands](#other-commands)
- [Python API](#python-api)
- [Documentation](#documentation)

## Installation

### The easy way

Install the pre-compiled version from PyPI:

```bash
pip install lchemme
```

### From source

Clone the repository, then `cd` into it. Then run:

```bash
pip install -e .
```

## Command-line usage

**lchemme**  provides command-line utlities to pre-train BART models.

To get a list of commands (tools), do

```bash
$ lchemme --help
usage: lchemme [-h] [--version] {tokenize,pretrain,featurize} ...

Training and applying large chemistry models.

options:
  -h, --help            show this help message and exit
  --version, -v         show program's version number and exit

Sub-commands:
  {tokenize,pretrain,featurize}
                        Use these commands to specify the tool you want to use.
    tokenize            Tokenize the data inputs.
    pretrain            Pre-train a large language model using self-supervised learning.
    featurize           Get vector embeddings of a chemical dataset using a pre-trained large language model.
```

And to get help for a specific command, do

```bash
$ lchemme <command> --help
```

### Tokenizing

The first step is to build a tokenizer for your dataset. **LChemME** works with BART models,
and pulls their architecture from the [Hugging Face Hub](https://huggingface.co/models) or 
a local directory. Training data can also be pulled from [Hugging Face Hub](https://huggingface.co/datasets)
with the `hf://` prefix, or it can be loaded from local CSV files with a column containing
SMILES strings.

```bash
lchemme tokenize \
    --train hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train \
    --column smiles \
    --model facebook/bart-base \
    --output my-model
```

This should be relatively fast, but could take several hours for millions of rows.

In principle, existing tokenizers trained on natural language could work, but they have
much larger vocabularies which are largely unused in SMILES.

### Model pretraining

**LChemME** performs semi-supervised pretraining on a SMILES canonicalization task.
This requires an understanding of chemical connectivity and atom precedence rules,
forcing the model to build an internal representation of the chemical graph.

```bash
lchemme pretrain \
    --train hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train \
    --column smiles \
    --test hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test \
    --model facebook/bart-base \
    --tokenizer my-model \
    --epochs 0.5 \
    --output my-model \
    --plot my-model/training-log
```

If you want to continue training, you can do so with the `--resume` flag.

```bash
lchemme pretrain \
    --train hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train \
    --column smiles \
    --test hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test \
    --model my-model \
    --epochs 0.5 \
    --output my-model \
    --plot my-model/training-log \
    --resume
```

The dataset state can only be restored if the `--model` was trained with **LChemME**
and the dataset configuration is identical, i.e. `--train`, `--column` are the same.

### Featurizing

With a trained model, you can generate embeddings of your chemical datasets,
optionally with UMAP plots colored by chemical properties.

```bash
lchemme featurize \
    --train hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train \
    --column smiles \
    --model my-model \
    --batch-size 16 \
    --method mean \
    --plot umap \
> featurized.csv
```

You can specify one or several aggregation functions with `--method`. **LChemME**
aggregates the sequence dimension of the encoder and decoder, then concatenates
them.

If you want to use additional columns containing numerical values to color the UMAP 
plots, provide the column names under `--extras`.

## Documentation

(Full API documentation to come at [ReadTheDocs](https://lchemme.readthedocs.org).)