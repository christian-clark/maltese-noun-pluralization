from train_plural_type_prediction import (
    get_data, get_pretrained_embeddings, get_semantic_id_matrix,
    get_char_sequence_matrix
)
import argparse
import io
import json
import numpy as np
import tensorflow as tf
import unidecode
from os import makedirs


HIDDEN_LAYER_SIZE = 128
HIDDEN_LAYER_ACTIVATION = 'sigmoid'
LSTM_EMBEDDING_DIM = 64
LSTM_DIM = 64
PAD = '<PAD>'
UNK = '<UNK>'
SEMITIC = '<SEM>'
NONSEMITIC = '<NONSEM>'


def get_expanded_char2id_dict(words):
    '''this char2id dict includes special characters for semitic and
    non-semitic etymology'''
    chars = set(char for word in words for char in word)
    chars.add(SEMITIC)
    chars.add(NONSEMITIC)
    char2id = dict()
    for char in chars:
        char2id[char] = len(char2id)
    char2id[PAD] = len(char2id)
    return char2id


def get_expanded_char_sequence_matrix(words, etyms, char2id, sequence_length):
    matrix = np.full((len(words), sequence_length), char2id[PAD])
    for i, word in enumerate(words):
        # truncate words that exceed sequence_length
        # save last character in sequence for etymology
        for j, char in enumerate(word[:sequence_length-1]):
            if char in char2id:
                matrix[i, j] = char2id[char]
            else:
                simplified = unidecode.unidecode(char)
                matrix[i, j] = char2id[simplified]
        etym = etyms[i]
        if etym == 0:
            matrix[i, -1] = char2id[NONSEMITIC]
        else:
            matrix[i, -1] = char2id[SEMITIC]
    return matrix


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Train LSTM prediction model for Maltese plural type')
    parser.add_argument('--train', type=str, required=True,
                        help='training corpus')
    parser.add_argument('--embeddings', type=str, required=True,
                        help='pretrained word embeddings')
    parser.add_argument('--hidden_layers', type=int, default=1,
                        help='number of hidden layers for the'\
                              + ' feedforward network')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    #parser.add_argument('--bilstm', action='store_true', default=False)
    parser.add_argument('--model_dest', type=str,
                        help='directory name for saving the trained model')
    return parser


def main():
    parser = get_argparser()
    args = parser.parse_args()

    words, etymologies, plural_types = get_data(args.train)
    word_count = len(words)
    assert word_count == len(etymologies) and word_count == len(plural_types)
    max_word_length = max(len(w) for w in words) 

    semantic_embedding_weights, word2id = \
        get_pretrained_embeddings(args.embeddings)
    semantic_embedding_input_dim = semantic_embedding_weights.shape[0]
    semantic_embedding_output_dim = semantic_embedding_weights.shape[1]

    ################
    # PREPARE NETWORK
    ################

    semantic_word_id = tf.keras.layers.Input(shape=(1,))
    semantic_embedding = tf.keras.layers.Embedding(
        input_dim=semantic_embedding_input_dim,
        output_dim=semantic_embedding_output_dim,
        trainable=False,
        weights=[semantic_embedding_weights]
    )(semantic_word_id)
    semantic_embedding = tf.reshape(
        semantic_embedding, (-1, semantic_embedding_output_dim)
    )

    # squeeze semantic embeddings down to dimensionality required by 
    # LSTM
    semantic_projection = tf.keras.layers.Dense(
                            units=LSTM_EMBEDDING_DIM,
                            activation='sigmoid'
                          )(semantic_embedding)

    # the +1 is for the etymology, which is represented as a special
    # character after the sequence of characters from the word
    char_embedding_input = tf.keras.layers.Input(shape=(max_word_length+1,))
    char2id = get_expanded_char2id_dict(words)
    char_embedding = tf.keras.layers.Embedding(
                         input_dim=len(char2id),
                         output_dim=LSTM_EMBEDDING_DIM)(char_embedding_input)
    #if args.bilstm:
    #    lstm_layer = tf.keras.layers.LSTM(units=LSTM_DIM) 
    #    lstm_out = tf.keras.layers.Bidirectional(lstm_layer)(char_embedding)

    semantic_projection = tf.reshape(
        semantic_projection, 
        (tf.shape(char_embedding)[0], 1, semantic_projection.shape[1])
    )
    lstm_input = tf.keras.layers.Concatenate(axis=1)(
        [char_embedding, semantic_projection],
    )
    lstm_out = tf.keras.layers.LSTM(units=LSTM_DIM)(lstm_input)

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
        inputs=[char_embedding_input, semantic_word_id],
        outputs=prediction
    )
    model.summary()

    ################
    # TRAIN NETWORK
    ################

    semantic_id_matrix = get_semantic_id_matrix(words, word2id)
    # this includes the designated etymology characters
    char_sequence_matrix = get_expanded_char_sequence_matrix(
                                words, etymologies, char2id,
                                max_word_length+1
                           )

    print('csm shape:', char_sequence_matrix.shape)
    
    ys = np.array(plural_types)
    model.compile(optimizer='adam', loss='binary_crossentropy',
    	      metrics=['accuracy'])
    model.fit(
        x=[char_sequence_matrix, semantic_id_matrix],
        y=ys, 
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    makedirs(args.model_dest, exist_ok=True)
    model.save('{}/tf_model'.format(args.model_dest))
    json_data = dict()
    json_data['char2id'] = char2id
    json_data['word2id'] = word2id
    json_data['max_word_length'] = max_word_length
    json.dump(json_data, open('{}/data.json'.format(args.model_dest), 'w'))

if __name__ == '__main__':
    main()

