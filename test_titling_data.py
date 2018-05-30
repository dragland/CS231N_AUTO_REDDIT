import vocab
import numpy as np
from titling_data import *
import json

NUM_SUBREDDITS = 20
START_TOKEN = '<START>'
PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'
END_TOKEN = '<END>'

embedding_matrix, words_by_id, id_by_words = vocab.load_embedding_matrix()
with open('train.json') as f:
    data = json.load(f)
    posts = data['posts']
    for post in posts[:5]:
        X_img, X_subreddit, X_title_indices, y = model_input_output_from_post(post, id_by_words, 15)
        print('Post title: ', post['title'])

        sequence = []
        for i in X_title_indices:
            word = words_by_id[i]
            sequence.append(word)
        print('Inputs: ', sequence)

        target = []
        for prob in y:
            i = np.argmax(prob)
            word = words_by_id[i]
            target.append(word)

        print('Target:', target)

        print()
