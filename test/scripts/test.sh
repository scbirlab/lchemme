#!/usr/bin/env bash

set -xeuo pipefail

TRAIN_DATA="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train"
TEST_DATA="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test"
COLUMN=smiles
MODEL="facebook/bart-base"
OUTPUT_DIR="test/outputs"
EPOCHS=.05

export TOKENIZERS_PARALLELISM=false

lchemme tokenize \
    --train $TRAIN_DATA \
    --column smiles \
    --model $MODEL \
    --output $OUTPUT_DIR/tokenizer

lchemme pretrain \
    --train $TRAIN_DATA \
    --column $COLUMN \
    --test $TEST_DATA \
    --model $MODEL \
    --tokenizer $OUTPUT_DIR/tokenizer \
    --epochs $EPOCHS \
    --output $OUTPUT_DIR/model \
    --plot $OUTPUT_DIR/training-log

lchemme pretrain \
    --train $TRAIN_DATA \
    --column $COLUMN \
    --test $TEST_DATA \
    --resume \
    --model $OUTPUT_DIR/model \
    --epochs $EPOCHS \
    --output $OUTPUT_DIR/model \
    --plot $OUTPUT_DIR/training-log

lchemme featurize \
    --train $TEST_DATA \
    --column $COLUMN \
    --batch-size 16 \
    --method max mean median start sum var \
    --model $OUTPUT_DIR/model \
    --plot $OUTPUT_DIR/umap \
> $OUTPUT_DIR/featurized.csv