#!/usr/bin/env bash

set -e
set -x

TRAIN_DATA="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train"
TEST_DATA="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test"
COLUMN=smiles
MODEL="sshleifer/bart-tiny-random"
OUTPUT_DIR="test/outputs"
EPOCHS=.2

lchemme tokenize  \
    --column smiles \
    --model $MODEL \
    --output $OUTPUT_DIR/tokenizer

lchemme pretrain $TRAIN_DATA \
    --column $COLUMN \
    --test $TEST_DATA \
    --model $MODEL \
    --tokenizer OUTPUT_DIR/tokenizer \
    --epochs $EPOCHS \
    --output $OUTPUT_DIR/model \
    --plot $OUTPUT_DIR/training-log

lchemme pretrain $TRAIN_DATA \
    --column $COLUMN \
    --test $TEST_DATA \
    --model $MODEL/model \
    --resume \
    --epochs $EPOCHS \
    --output $OUTPUT_DIR/model \
    --plot $OUTPUT_DIR/training-log

lchemme featurize $TEST_DATA \
    --column $COLUMN \
    --batch-size 64 \
    --method max mean median start sum var \
    --model $OUTPUT_DIR/model \
    --plot $OUTPUT_DIR/umap