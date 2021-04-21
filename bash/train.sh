#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00

set -x 

export PYTHONIOENCODING=utf-8
CROSS_VAL_ROOT=/home/clark.3664/projects/ling5802_final_project/data/cross_val


#TRAIN=/home/clark.3664/projects/ling5802_final_project/data/maltese_data_20210418.train.txt
EMBEDDINGS=/home/clark.3664/projects/ling5802_final_project/data/maltesewiki_vectorsunkl4
HIDDEN_LAYERS=1
BATCH_SIZE=32
EPOCHS=200
MODEL_NAME=bs${BATCH_SIZE}_e${EPOCHS}


for split in split1 split2; do
    echo "=== TRAINING MODEL FOR $split ==="
    train=$CROSS_VAL_ROOT/$split/train.txt
    dest=$CROSS_VAL_ROOT/$split/$MODEL_NAME
    python /home/clark.3664/projects/ling5802_final_project/python/train_noun_class_prediction.py \
      --train $train \
      --embeddings $EMBEDDINGS \
      --hidden_layers $HIDDEN_LAYERS \
      --batch_size $BATCH_SIZE \
      --epochs $EPOCHS \
      --model_dest $dest \
      #--no_etymology \
      #--no_lstm \
      #--no_semantics \
      #--fine_tune_semantics \

