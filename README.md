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
lchemme --help
```

And to get help for a specific command, do

```bash
lchemme <command> --help
```

## Documentation

(Full API documentation to come at [ReadTheDocs](https://lchemme.readthedocs.org).)