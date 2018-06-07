#!//usr/bin/python
#Davy Ragland | dragland@stanford.edu
#Adrien Truong | aqtruong@stanford.edu 
#CS231N_REDDIT_NET | 2018

#*********************************** SETUP *************************************
import credentials
import os
import sys
import argparse
import praw
import urllib2
import PIL
from PIL import Image, ImageOps
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import json
import random

reddits = ["art", "streetwear",
		   "womensstreetwear", "OldSchoolCool",
		   "cosplay", "malefashion",
		   "sneakers", "cats",
		   "flowers", "gaming",
		   "AccidentalRenaissance", "AccidentalWesAnderson",
		   "ExpectationVsReality", "PerfectTiming",
		   "woahdude", "CrappyDesign",
		   "CozyPlaces", "comics",
		   "natureisfuckinglit", 'aww']
dataset_path = "datasets/"
model_path = "model.json"
train_path = "train.json"
validation_path = "validation.json"
test_path = "test.json"
training_size_default = 50
augment_size_default = 4
training_resolution_default = 1000
indices_by_prefixed_subreddit = {}

def get_index_for_subreddit_name_prefixed(name):
	if name not in indices_by_prefixed_subreddit:
		indices_by_prefixed_subreddit[name] = len(indices_by_prefixed_subreddit)
	return indices_by_prefixed_subreddit[name]

#*********************************** HELPERS ***********************************
def download(training_size):
	print("Downloading training data...")
	posts = []
	reddit = praw.Reddit(client_id=credentials.CLIENT_ID, client_secret=credentials.CLIENT_SECRET, user_agent="CS231N_REDDIT_NET")
	for subreddit in reddits:
		print(subreddit + "......................................................")
		i = 0
		for submission in reddit.subreddit(subreddit).top("all", limit=None):
			filetype = submission.url.split(".")[-1]
			if filetype == "jpg" or filetype == "png" or filetype == "JPG" or filetype == "PNG":
				post = {}
				post["id"] = submission.id
				post["title"] = submission.title
				post["subreddit"] = get_index_for_subreddit_name_prefixed(submission.subreddit_name_prefixed)
				post["url"] = submission.url
				post["score"] = str(submission.score)
				post["path"] = dataset_path + subreddit + str(i) + "." + filetype
				post["created"] = submission.created
				
				print(str(i) + ": " + post["url"] + " [" + post["score"] + "]")
				req = urllib2.Request(post["url"])
				req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0")
				try:
					f = urllib2.urlopen(req)
					urltype = f.geturl().split(".")[-1]
					if urltype == "jpg" or urltype == "png" or urltype == "JPG" or urltype == "PNG":
						with open(post["path"], "wb") as output:
						  output.write(f.read())
						posts.append(post)
						i += 1
				except Exception as e:
				    print("URLError = " + str(e.reason))
			if i == training_size:
				break
	model = {
		'posts': posts,
		'subreddit_indices_map': indices_by_prefixed_subreddit
	}
	with open(model_path, "w") as outfile:  
		json.dump(model, outfile)

def preprocess(training_resolution):
	print("pre-processing training data...")
	size = training_resolution, training_resolution
	for pic in os.listdir(dataset_path):
		path = dataset_path + pic
		img = Image.open(path).convert('RGB')
		new = ImageOps.fit(img, size, Image.ANTIALIAS)
		print(pic + str(new.size))
		new.save(path)

def augment():
	print("augmenting training data...")
	datagen = ImageDataGenerator(
	        rotation_range=40,
	        width_shift_range=0.2,
	        height_shift_range=0.2,
	        shear_range=0.2,
	        zoom_range=0.2,
	        horizontal_flip=True,
	        fill_mode='nearest')
	with open(train_path) as f:
		model = json.load(f)
		posts = model["posts"]
		key = model["subreddit_indices_map"]
		for i in range(len(posts)):
			curr = posts[i]
			img = load_img(curr["path"])
			x = img_to_array(img)
			x = x.reshape((1,) + x.shape)
			i = 0
			for new in datagen.flow(x, batch_size=1, save_to_dir='augment', save_prefix=curr["path"].split("/")[-1].split(".")[0], save_format='jpg'):
				i += 1
				if i == augment_size_default:
					break
			for f in os.listdir('augment'):
				copy = curr.copy()
				os.rename(os.path.join('augment', f), os.path.join('datasets', f))
				copy["path"] = os.path.join('datasets', f)
				posts.append(copy)
			print(curr["path"])
		random.shuffle(posts)
		print(len(posts))
		train = {
			'posts': posts,
			'subreddit_indices_map': key
		}
		with open(train_path, "w") as outfile:  
			json.dump(train, outfile)

def cleanup():
	print("cleaning up training data...")
	if os.path.exists(model_path):
		os.remove(model_path)
	if os.path.exists(train_path):
		os.remove(train_path)
	if os.path.exists(validation_path):
		os.remove(validation_path)
	if os.path.exists(test_path):
		os.remove(test_path)
	for pic in os.listdir(dataset_path):
		os.remove(dataset_path + pic)	

def split():
	print("splitting up training data for validation...")
	with open(model_path) as f:
		model = json.load(f)
		posts = model["posts"]
		key = model["subreddit_indices_map"]
		length = len(posts)
		random.shuffle(posts)
		train = {
			'posts': posts[:int(length * .8)],
			'subreddit_indices_map': key
		}
		print("training size: " + str(len(train["posts"])))
		validate = {
			'posts': posts[int(length * .8): int(length * .9)],
			'subreddit_indices_map': key
		}
		print("validation size: " + str(len(validate["posts"])))
		test = {
			'posts': posts[int(length * .9):],
			'subreddit_indices_map': key
		}
		print("test size: " + str(len(test["posts"])))
		with open(train_path, "w") as outfile:
			json.dump(train, outfile)
		with open(validation_path, "w") as outfile:
			json.dump(validate, outfile)
		with open(test_path, "w") as outfile:
			json.dump(test, outfile)

#************************************ MAIN *************************************
if __name__ == "__main__":
	print("Executing program:")
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", action="store_true", help="cleanup training directory")
	parser.add_argument("-d", type=int, help="number of training examples per class")
	parser.add_argument("-p", type=int, help="preprocessing resolution of training example")
	parser.add_argument("-a", action="store_true", help="augment training examples")
	parser.add_argument("-s", action="store_true", help="split training data for validation")
	args = parser.parse_args()
	if len(sys.argv) <= 1:
		download(training_size_default)
		preprocess(training_resolution_default)
		split()
	else:
		if args.c:
			cleanup()
		if args.d:
			download(args.d)
		if args.p:
			preprocess(args.p)
		if args.a:
			augment()
		if args.s:
			split()