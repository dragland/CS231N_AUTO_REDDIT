import numpy as np
from keras.preprocessing.text import text_to_word_sequence
import json
from collections import defaultdict

START_TOKEN = '<START>'
PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'
END_TOKEN = '<END>'
# important that PAD_TOKEN have index 0
SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, UNKNOWN_TOKEN, END_TOKEN]

def load_vocab(json_path):
    # maintain array so that ordering is consistent across runs
    # and words get mapped to same id
    # use set for performance reasons
    word_counts = defaultdict(int)
    with open(json_path) as f:
        data = json.load(f)
        for post in data['posts']:
            words = text_to_word_sequence(post['title'])
            for w in words:
                word_counts[w] += 1

    words_by_id = {}
    ids_by_word = {}

    i = 0
    for token in SPECIAL_TOKENS:
        words_by_id[i] = token
        ids_by_word[token] = i
        i += 1

    for word, count in word_counts.items():
        if count >= 5:
            words_by_id[i] = word
            ids_by_word[word] = i
            i += 1

    return words_by_id, ids_by_word

def load_limited_embedding_matrix(json_path, embedding_size):
    glove_path = 'glove.6B.{}d.txt'.format(embedding_size)
    glove_index = {}
    with open(glove_path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            glove_index[word] = embedding

    # maintain array so that ordering is consistent across runs
    # and words get mapped to same id
    # use set for performance reasons
    unique_words_set = set()
    unique_words = []
    word_counts = defaultdict(int)
    total_num_words = 0
    with open(json_path) as f:
        data = json.load(f)
        for post in data['posts']:
            words = text_to_word_sequence(post['title'])
            total_num_words += len(words)
            for w in words:
                if w not in unique_words_set:
                    unique_words_set.add(w)
                    unique_words.append(w)
                word_counts[w] += 1

    print('total num words:', total_num_words)
    num_words = len(unique_words) + len(SPECIAL_TOKENS)
    embedding_matrix = np.zeros((num_words, embedding_size))
    words_by_id = {}
    ids_by_word = {}

    # give special tokens a random word vector
    next_word_id = 0
    for token in SPECIAL_TOKENS:
        words_by_id[next_word_id] = token
        ids_by_word[token] = next_word_id
        embedding_matrix[next_word_id] = np.random.randn(embedding_size)
        next_word_id += 1

    for word in unique_words:
        if word in glove_index:
            embedding = glove_index[word]
        elif word_counts[word] >= 20:
            embedding = np.random.randn(embedding_size)
        else:
            # skip it, let it map to unknown token
            continue

        words_by_id[next_word_id] = word
        ids_by_word[word] = next_word_id
        embedding_matrix[next_word_id] = embedding

        next_word_id += 1

    embedding_matrix = embedding_matrix[:next_word_id]

    print('total vocab size', len(embedding_matrix))
    return embedding_matrix, words_by_id, ids_by_word

def load_embedding_matrix():
    embedding_size = 50

    num_glove_words = 0
    with open('glove.6B.50d.txt') as f:
        for i, line in enumerate(f):
            num_glove_words += 1

    num_words = num_glove_words + len(SPECIAL_TOKENS)
    embedding_matrix = np.zeros((num_words, embedding_size))
    words_by_id = {}
    ids_by_word = {}

    # give special tokens a random word vector
    for i, token in enumerate(SPECIAL_TOKENS):
        word_id = i
        words_by_id[word_id] = token
        ids_by_word[token] = word_id
        embedding_matrix[i] = np.random.randn(embedding_size)

    start_i = len(SPECIAL_TOKENS)
    with open('glove.6B.50d.txt') as f:
        for i, line in enumerate(f):
            word_id = start_i + i
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')

            words_by_id[word_id] = word
            ids_by_word[word] = word_id
            embedding_matrix[word_id] = embedding

    return embedding_matrix, words_by_id, ids_by_word
