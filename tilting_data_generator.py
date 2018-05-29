def data_generator(json, num_subreddits, word_indices_map):
    X_train = []
    y_train = []
    with open(json_path) as f:
        data = json.load(f)
        for post in data['posts']:
            path = post['path']
            subreddit = post['subreddit']
            img = np.array(Image.open(path))

            subreddit_one_hot = np.zeros(num_subreddits)
            subreddit_one_hot[subreddit] = 1

            title_indices = []
            y = []
            for word in post['title'].split():
                index = word_indices_map[word]
                title_indices.append(index)

                target = np.zeros(len(word_indices_map))
                target[index] = 1
                y.append(target)
            title_indices = np.array(title_indices)
            y = np.array(y)

            title_indices = np.array(title_indices)
