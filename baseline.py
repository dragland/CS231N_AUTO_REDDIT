#!//usr/bin/python
#Davy Ragland | dragland@stanford.edu
#Adrien Truong | aqtruong@stanford.edu 
#CS231N_REDDIT_NET | 2018

#*********************************** SETUP *************************************
import sys
import argparse
import PIL
from PIL import Image, ImageOps
import json
import numpy as np
from collections import Counter

NUM_CLASSES=20
k_default = 5
train_path_default = "train.json"
train_small_path_default = "small_train.json"

#*********************************** HELPERS ***********************************
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

def get_subreddit_indices_map(path):
    with open(path) as f:
        data = json.load(f)
        return data['subreddit_indices_map']

def get_subreddit_for_index(index):
    subreddit_indices_map = get_subreddit_indices_map(train_path_default)
    reverse_map = {}
    for k, v in subreddit_indices_map.items():
        reverse_map[v] = k
    return reverse_map[index]

def evaluate(config):
    img_path = config.i
    X_train, y_train = get_data(config.train_path)
    X_val, y_val = get_data('validation.json')
    size = X_train.shape[1], X_train.shape[2]
    num = X_val.shape[0]
    acc = 0
    for i in range(0, num):
    	d = []
    	img = X_val[i]
    	for j in range(0, X_train.shape[0]):
    		dist = np.sum(np.abs(img - X_train[j]))
    		d.append(dist)
    	idx = np.argpartition(d, config.k)[:config.k]
    	b = []
    	for j in range(0, config.k):
    		label = np.where(y_train[idx[j]] != 0)[0][0]
    		b.append(label)
    	label_i = Counter(b).most_common(1)[0][0]
    	label = get_subreddit_for_index(label_i)

    	if label_i == np.where(y_val[i] != 0)[0][0]:
    		acc += 1
    	print(str(i) + ": Predicting " + label + " acc:" + str(acc))
    print("validation accuracy of " + str(float(100 * acc) / num) + "%...")

def predict(config):
    img_path = config.i
    X_train, y_train = get_data(config.train_path)
    size = X_train.shape[1], X_train.shape[2]
    img = Image.open(img_path).convert('RGB')
    new = ImageOps.fit(img, size, Image.ANTIALIAS)
    img = np.array(new)
    d = []
    for i in range(0, X_train.shape[0]):
    	dist = np.sum(np.abs(img - X_train[i]))
    	d.append(dist)
    idx = np.argpartition(d, config.k)[:config.k]
    b = []
    for i in range(0, config.k):
    	label = np.where(y_train[idx[i]] != 0)[0][0]
    	b.append(label)
    label_i = Counter(b).most_common(1)[0][0]
    label = get_subreddit_for_index(label_i)
    print('Predicting {}'.format(label))

#************************************ MAIN *************************************
if __name__ == "__main__":
	print(sys.version)
	print("Executing program:")
	parser = argparse.ArgumentParser()
	parser.add_argument("-s", action="store_true", help="use small training set")
	parser.add_argument("-k", type=int, help="k value for baseline")
	parser.add_argument("-e", action="store_true", help="evaluate baseline")
	parser.add_argument("-i", type=str, help='path of img to predict')
	config = parser.parse_args()

	if len(sys.argv) <= 1:
	    print('Invalid mode! Aborting...')
	    print("example usage: ")
	    print("./baseline.py -k=5 -e")
	    print("./baseline.py -k=5 -i=example.jpg")

	else:
		config.train_path = train_path_default
		if config.s:
			config.train_path = train_small_path_default
		if config.k == None:
			config.k = k_default
		print(config)

		if config.e:
			evaluate(config)
		if config.i:
			predict(config)