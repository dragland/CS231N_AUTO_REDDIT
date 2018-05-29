import numpy as np

START_TOKEN = '<START>'
PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'
END_TOKEN = '<END>'
# important that PAD_TOKEN have index 0
SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, UNKNOWN_TOKEN, END_TOKEN]

def load_embedding_matrix():
    embeddings_index = {}
    embedding_size = 50
    with open('glove.6B.50d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding

    num_words = len(embeddings_index) + len(SPECIAL_TOKENS)
    embedding_matrix = np.zeros((num_words, embedding_size))
    words_by_id = {}
    ids_by_word = {}
    start_i = len(SPECIAL_TOKENS)
    for i, (word, embedding) in enumerate(embeddings_index.items()):
        word_id = start_i = i
        words_by_id[word_id] = word
        ids_by_word[word] = word_id
        embedding_matrix[word_id] = embedding

    # give special tokens a random word vector
    for i, token in enumerate(SPECIAL_TOKENS):
        word_id = i
        words_by_id[word_id] = token
        ids_by_word[token] = word_id
        embedding_matrix[i] = np.random.randn(embedding_size)

    return embedding_matrix, words_by_id, ids_by_word