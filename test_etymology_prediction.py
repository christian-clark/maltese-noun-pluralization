from train_plural_type_prediction import get_data, get_char_sequence_matrix
from test_plural_type_prediction import (
    zero_one_accuracy, per_class_metrics, matrix2preds)
import argparse
import io
import json
import numpy as np
from tensorflow import keras


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Test noun class prediction model')
    parser.add_argument('--test', type=str, required=True,
                        help='evaluation corpus')
    parser.add_argument('--model_src', type=str,
                        help='directory name for loading the trained model')
    parser.add_argument('--pred_dest', type=str,
                        help='output file for model predictions')
    return parser


def main():
    parser = get_argparser()
    args = parser.parse_args()

    json_data = json.load(open('{}/data.json'.format(args.model_src)))
    char2id = json_data['char2id']
    max_lstm_seq_length = json_data['max_lstm_seq_length']
    model = keras.models.load_model('{}/tf_model'.format(args.model_src))

    words, etymologies, _ = get_data(args.test)
    char_sequence_matrix = get_char_sequence_matrix(
                               words, char2id, max_lstm_seq_length)
    ys = np.array(etymologies)
    pred_matrix = model(char_sequence_matrix)
    preds = matrix2preds(pred_matrix)

    if args.pred_dest:
        f = io.open(args.pred_dest, 'w', encoding='utf8')
        f.write('word\tprediction\tactual\n')
        for i, word in enumerate(words):
            pred = preds[i]
            actual = etymologies[i]
            f.write('{}\t{}\t{}\n'.format(word, pred, actual))
        f.close()

    accuracy = zero_one_accuracy(preds, etymologies)
    print('0/1 accuracy:', accuracy)
    # class 0: Non-Semitic (mostly Romance) etymology
    # class 1: Semitic etymology
    metrics0, metrics1 = per_class_metrics(preds, etymologies)
    print('class 0 (sound plural) metrics:')
    print('\tprecision:', metrics0['precision'])
    print('\trecall:', metrics0['recall'])
    print('\tf-score:', metrics0['fscore'])
    print('class 1 (broken plural) metrics:')
    print('\tprecision:', metrics1['precision'])
    print('\trecall:', metrics1['recall'])
    print('\tf-score:', metrics1['fscore'])


if __name__ == '__main__':
    main()
