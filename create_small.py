#!//usr/bin/python
import json
import numpy as np

with open('train.json') as f:
    NUM_EXAMPLES = 500
    data = json.load(f)
    posts = data['posts']
    indices = np.random.choice(len(posts), NUM_EXAMPLES)
    small_posts = []
    for i in indices:
      small_posts.append(posts[i])
    data['posts'] = small_posts
    with open('small_train.json', 'w') as f:
        json.dump(data, f)
