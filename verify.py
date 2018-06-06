import json
from PIL import Image

def get_data(json_path):
    X_train = []
    y_train = []
    with open(json_path) as f:
        data = json.load(f)
        for post in data['posts']:
            path = post['path']
            label = post['subreddit']

            img = np.array(Image.open(path))
            X_train.append(img)

            one_hot = np.zeros(NUM_CLASSES)
            one_hot[label] = 1
            y_train.append(one_hot)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train

def validate(json_path):
    with open(json_path) as f:
        data = json.load(f)
        for post in data['posts']:
            path = post['path']
            
            try:
                img = Image.open(path)
            except Exception as e:
                print('post is bad:', post)

validate('model.json')
