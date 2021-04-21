import argparse
import io
import json
import numpy as np
import tensorflow as tf
from os import makedirs

# TODO might want to make this a command-line argument
HIDDEN_LAYER_SIZE = 128
HIDDEN_LAYER_ACTIVATION = 'sigmoid'
LSTM_EMBEDDING_DIM = 64
LSTM_DIM = 64
# TODO possible to avoid using PAD?
PAD = '<PAD>'
UNK = '<UNK>'


# https://keras.io/guides/making_new_layers_and_models_via_subclassing/
class Ablate(tf.keras.layers.Layer):
    def __init__(self, units=1):
        super(Ablate, self).__init__()
        self.units = units

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.zeros((batch_size, self.units))


def get_data(filename):
    words, etymologies, noun_classes = list(), list(), list()
    #with open(filename) as f:
    with io.open(filename, encoding='utf8') as f:
        for l in f:
            word, noun_class, etym = l.strip().split()
            words.append(word)
            noun_classes.append(int(noun_class))
            etymologies.append(int(etym))
    return words, etymologies, noun_classes


def get_pretrained_embeddings(filename):
    f = open(filename)
    toks = f.readline().strip().split()
    word_count = int(toks[0])
    feature_count = int(toks[1])
    embeddings = list()
    word2id = dict()
    l = f.readline()
    while l:
        toks = l.strip().split()
        word = toks[0]
        feats = toks[1:]
        assert len(feats) == feature_count
        feats = [float(fe) for fe in feats]
        embeddings.append(feats)
        word2id[word] = len(word2id)
        l = f.readline()
    assert len(word2id) == word_count
    return np.array(embeddings), word2id


def get_char2id_dict(words):
    chars = set(char for word in words for char in word)
    char2id = dict()
    for char in chars:
        char2id[char] = len(char2id)
    char2id[PAD] = len(char2id)
    return char2id


def get_semantic_id_matrix(words, word2id):
    ids = list()
    for w in words:
        if w in word2id:
            ids.append([word2id[w]])
        else:
            ids.append([word2id[UNK]])
    return np.array(ids)


def get_char_sequence_matrix(words, char2id, sequence_length):
    matrix = np.full((len(words), sequence_length), char2id[PAD])
    for i, word in enumerate(words):
        # truncate words that exceed sequence_length
        for j, char in enumerate(word[:sequence_length]):
            matrix[i, j] = char2id[char]
    return matrix


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Train noun class prediction model')
    parser.add_argument('--train', type=str, required=True,
                        help='training corpus')
    parser.add_argument('--embeddings', type=str, required=True,
                        help='pretrained word embeddings')
    parser.add_argument('--hidden_layers', type=int, default=1,
                        help='number of hidden layers')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--no_etymology', action='store_true', default=False)
    parser.add_argument('--no_semantics', action='store_true', default=False)
    parser.add_argument('--no_lstm', action='store_true', default=False)
    parser.add_argument('--fine_tune_semantics', action='store_true', default=False)
    parser.add_argument('--model_dest', type=str,
                        help='directory name for saving the trained model')
    return parser


def main():
    parser = get_argparser()
    args = parser.parse_args()

    assert (not args.no_etymology) or (not args.no_semantics) \
        or (not args.no_lstm)

    words, etymologies, noun_classes = get_data(args.train)
    word_count = len(words)
    assert word_count == len(etymologies) and word_count == len(noun_classes)
    max_lstm_seq_length = max(len(w) for w in words)

    semantic_embedding_weights, word2id = \
        get_pretrained_embeddings(args.embeddings)
    semantic_embedding_input_dim = semantic_embedding_weights.shape[0]
    semantic_embedding_output_dim = semantic_embedding_weights.shape[1]

    ################
    # PREPARE NETWORK
    ################

    # last 2 dimensions are for word embedding ID and etymology
    #input_dim = max_lstm_seq_length + 2
    #network_input = tf.keras.layers.Input(shape=(input_dim,))

    ### Etymology
    etymology_input = tf.keras.layers.Input(shape=(1,))
    #etymology_feature = network_input[:, -2]
    if args.no_etymology:
        etymology_feature = Ablate()(etymology_input)

    else:
        etymology_feature = etymology_input

    #etymology_feature = tf.reshape(etymology_feature, (-1, 1))

    ### Semantic (word-level) embeddings
    #semantic_word_id = network_input[:, -1]
    semantic_word_id = tf.keras.layers.Input(shape=(1,))
    if args.no_semantics:
        semantic_embedding = Ablate(
            units=semantic_embedding_output_dim
        )(semantic_word_id)

    else:
        semantic_embedding = tf.keras.layers.Embedding(
            input_dim=semantic_embedding_input_dim,
            output_dim=semantic_embedding_output_dim,
            trainable=args.fine_tune_semantics,
            weights=[semantic_embedding_weights]
        )(semantic_word_id)
        semantic_embedding = tf.reshape(
            semantic_embedding, (-1, semantic_embedding_output_dim)
        )

    ### LSTM (char-level) embeddings
    lstm_input = tf.keras.layers.Input(shape=(max_lstm_seq_length,))
    #lstm_input = network_input[:, :-2]
    char2id = get_char2id_dict(words)
    if args.no_lstm:
        lstm_out = Ablate(units=LSTM_DIM)(lstm_input)

    else:
        char_embedding = tf.keras.layers.Embedding(
                             input_dim=len(char2id),
                             output_dim=LSTM_EMBEDDING_DIM)(lstm_input)
        lstm_out = tf.keras.layers.LSTM(units=LSTM_DIM)(char_embedding)

    ### Etymology, semantics, and LSTM embeddings go into feedforward network
    feedforward_input = tf.keras.layers.Concatenate()(
        [lstm_out, semantic_embedding, etymology_feature]
    )

    curr = feedforward_input

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
        inputs=[etymology_input, semantic_word_id, lstm_input],
        outputs=prediction
    )
    model.summary()

    ################
    # TRAIN NETWORK
    ################

    semantic_id_matrix = get_semantic_id_matrix(words, word2id)
    char_sequence_matrix = get_char_sequence_matrix(
                               words, char2id, max_lstm_seq_length)
    etymologies = np.reshape(etymologies, (-1, 1))
    ys = np.array(noun_classes)
    model.compile(optimizer='adam', loss='binary_crossentropy',
    	      metrics=['accuracy'])
    model.fit(
        x=[etymologies, semantic_id_matrix, char_sequence_matrix],
        y=ys, 
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    makedirs(args.model_dest, exist_ok=True)
    model.save('{}/tf_model'.format(args.model_dest))
    json_data = dict()
    json_data['char2id'] = char2id
    json_data['word2id'] = word2id
    json_data['max_lstm_seq_length'] = max_lstm_seq_length
    json.dump(json_data, open('{}/data.json'.format(args.model_dest), 'w'))


if __name__ == '__main__':
    main()

