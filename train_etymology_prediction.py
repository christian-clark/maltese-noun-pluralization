from train_plural_type_prediction import (
    get_data, get_char2id_dict, get_char_sequence_matrix,
    HIDDEN_LAYER_SIZE, HIDDEN_LAYER_ACTIVATION, LSTM_EMBEDDING_DIM, LSTM_DIM
)
import argparse
import io
import json
import numpy as np
import tensorflow as tf
import unidecode
from os import makedirs


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Train etymology prediction model')
    parser.add_argument('--train', type=str, required=True,
                        help='training corpus')
    parser.add_argument('--hidden_layers', type=int, default=1,
                        help='number of hidden layers')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--model_dest', type=str,
                        help='directory name for saving the trained model')
    return parser


def main():
    parser = get_argparser()
    args = parser.parse_args()

    words, etymologies, _ = get_data(args.train)
    assert len(words) == len(etymologies)
    max_lstm_seq_length = max(len(w) for w in words)

    lstm_input = tf.keras.layers.Input(shape=(max_lstm_seq_length,))
    char2id = get_char2id_dict(words)
    char_embedding = tf.keras.layers.Embedding(
                         input_dim=len(char2id),
                         output_dim=LSTM_EMBEDDING_DIM)(lstm_input)
    lstm_out = tf.keras.layers.LSTM(units=LSTM_DIM)(char_embedding)

    curr = lstm_out

    for i in range(args.hidden_layers):
        hidden = tf.keras.layers.Dense(
            units=HIDDEN_LAYER_SIZE,
            activation=HIDDEN_LAYER_ACTIVATION
        )
        curr = hidden(curr)

    prediction_layer = tf.keras.layers.Dense(
        units=1,
        activation='sigmoid'
    )
    prediction = prediction_layer(curr)
    model = tf.keras.Model(
        inputs=lstm_input,
        outputs=prediction
    )
    model.summary()

    char_sequence_matrix = get_char_sequence_matrix(
                               words, char2id, max_lstm_seq_length)
    ys = np.array(etymologies)
    model.compile(optimizer='adam', loss='binary_crossentropy',
    	      metrics=['accuracy'])
    model.fit(
        x=char_sequence_matrix,
        y=ys,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    makedirs(args.model_dest, exist_ok=True)
    model.save('{}/tf_model'.format(args.model_dest))
    json_data = dict()
    json_data['char2id'] = char2id
    json_data['max_lstm_seq_length'] = max_lstm_seq_length
    json.dump(json_data, open('{}/data.json'.format(args.model_dest), 'w'))


if __name__ == '__main__':
    main()
