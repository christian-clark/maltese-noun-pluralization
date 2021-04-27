from train_plural_type_prediction import get_data, get_semantic_id_matrix, \
    get_char_sequence_matrix
import argparse
import io
import json
import numpy as np
from tensorflow import keras


def matrix2preds(matrix):
    preds = list()
    for val in matrix:
        if val >= 0.5:
            preds.append(1)
        else:
            preds.append(0)
    return preds


def zero_one_accuracy(preds, actuals):
    assert len(preds) == len(actuals)
    correct = 0
    total = 0
    for i, pred in enumerate(preds):
        if pred == actuals[i]:
            correct += 1
        total += 1
    return correct/total


def per_class_metrics(preds, actuals):
    assert len(preds) == len(actuals)
    t0, t1, f0, f1 = [0]*4
    for i, pred in enumerate(preds):
        actual = actuals[i]
        if actual == 0:
            if pred == 0:
                # actual 0, pred 0
                t0 += 1
            else:
                # actual 0, pred 1 
                f1 += 1
        else:
            if pred == 0:
                # actual 1, pred 0
                f0 += 1
            else:
                # actual 1, pred 1
                t1 += 1

    metrics0, metrics1 = dict(), dict()
    prec0 = t0/(t0 + f0)
    rec0 = t0/(t0 + f1)
    fsc0 = 2*prec0*rec0 / (prec0 + rec0)
    metrics0['precision'] = prec0
    metrics0['recall'] = rec0
    metrics0['fscore'] = fsc0

    prec1 = t1/(t1 + f1)
    rec1 = t1/(t1 + f0)
    fsc1 = 2*prec1*rec1 / (prec1 + rec1)
    metrics1['precision'] = prec1
    metrics1['recall'] = rec1
    metrics1['fscore'] = fsc1

    return metrics0, metrics1
                


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
    max_lstm_seq_length = json_data['max_lstm_seq_length']
    model = keras.models.load_model('{}/tf_model'.format(args.model_src))


    words, etymologies, noun_classes = get_data(args.test)
    etymologies = np.reshape(etymologies, (-1, 1))
    semantic_id_matrix = get_semantic_id_matrix(words, word2id)
    char_sequence_matrix = get_char_sequence_matrix(
                               words, char2id, max_lstm_seq_length)
    ys = np.array(noun_classes)
    pred_matrix = model([etymologies, semantic_id_matrix, char_sequence_matrix])
    preds = matrix2preds(pred_matrix)

    if args.pred_dest:
        f = io.open(args.pred_dest, 'w', encoding='utf8')
        f.write('word\tprediction\tactual\n')
        for i, word in enumerate(words):
            pred = preds[i]
            actual = noun_classes[i]
            f.write('{}\t{}\t{}\n'.format(word, pred, actual))
        f.close()

    accuracy = zero_one_accuracy(preds, noun_classes)
    print('0/1 accuracy:', accuracy)
    # class 0: sound plural
    # class 1: broken plural
    metrics0, metrics1 = per_class_metrics(preds, noun_classes)
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
