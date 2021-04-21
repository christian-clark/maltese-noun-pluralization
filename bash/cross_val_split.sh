FULL_DATASET=/home/clark.3664/projects/ling5802_final_project/data/maltese_data_20210418.all.txt
NUM_SPLITS=10
TRAIN_SIZE=2063

for i in $(seq $NUM_SPLITS); do
    dir=split${i}
    mkdir $dir
    shuf $FULL_DATASET | split -l $TRAIN_SIZE - $dir/
    mv $dir/aa $dir/train.txt
    mv $dir/ab $dir/test.txt
done

