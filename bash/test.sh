#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=1

set -x 

export PYTHONIOENCODING=utf-8
#TEST=/home/clark.3664/projects/ling5802_final_project/data/maltese_etym_data.test.txt
TEST=/home/clark.3664/projects/ling5802_final_project/data/maltese_data_20210418.test.txt
#MODEL_SRC=/home/clark.3664/projects/ling5802_final_project/models/bs32_e200
#MODEL_SRC=/home/clark.3664/projects/ling5802_final_project/models/bs32_e200_noLstm
#MODEL_SRC=/home/clark.3664/projects/ling5802_final_project/models/bs32_e200_noEtym
#MODEL_SRC=/home/clark.3664/projects/ling5802_final_project/models/bs32_e200_noSemantics
#MODEL_SRC=/home/clark.3664/projects/ling5802_final_project/models/bs32_e200_fineTuneSemantics
#MODEL_SRC=/home/clark.3664/projects/ling5802_final_project/models/bs32_e200_lstmOnly
#MODEL_SRC=/home/clark.3664/projects/ling5802_final_project/models/bs32_e200_etymOnly
MODEL_SRC=/home/clark.3664/projects/ling5802_final_project/models/bs32_e200_semanticsOnly
PRED_DEST=/home/clark.3664/projects/ling5802_final_project/predictions.txt

python /home/clark.3664/projects/ling5802_final_project/python/test_noun_class_prediction.py \
  --test $TEST \
  --model_src $MODEL_SRC \
  --pred_dest $PRED_DEST

