from train_plural_type_prediction import get_data, get_semantic_id_matrix
from train_plural_type_prediction_lstm import \
    get_expanded_char_sequence_matrix
from test_plural_type_prediction import (
    matrix2preds, zero_one_accuracy, per_class_metrics
)
import argparse
import io
import json
import math
import numpy as np
from sklearn.metrics import mean_squared_error
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
    word2id = json_data['word2id']
    max_word_length = json_data['max_word_length']
    model = keras.models.load_model('{}/tf_model'.format(args.model_src))

    words, etymologies, plural_types = get_data(args.test)
    semantic_id_matrix = get_semantic_id_matrix(words, word2id)
    # +1 to leave room for etymology
    char_sequence_matrix = get_expanded_char_sequence_matrix(
                               words, etymologies, char2id, 
                               max_word_length+1
                           )
    ys = np.array(plural_types)
    pred_matrix = model([char_sequence_matrix, semantic_id_matrix])
    preds = matrix2preds(pred_matrix)

    if args.pred_dest:
        f = io.open(args.pred_dest, 'w', encoding='utf8')
        f.write('word\tprediction\tactual\n')
        for i, word in enumerate(words):
            pred = preds[i]
            actual = plural_types[i]
            f.write('{}\t{}\t{}\n'.format(word, pred, actual))
        f.close()

    # MSE from predicting sound plural every time
    baseline_mse = mean_squared_error([0]*len(plural_types), plural_types)
    mse = mean_squared_error(pred_matrix, plural_types)
    accuracy = zero_one_accuracy(preds, plural_types)
    print('baseline mean squared error:', baseline_mse)
    print('mean squared error:', mse)
    print('0/1 accuracy:', accuracy)

    # class 0: sound plural
    # class 1: broken plural
    metrics0, metrics1 = per_class_metrics(preds, plural_types)
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

