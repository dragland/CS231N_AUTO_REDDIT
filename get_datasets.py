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
import json

reddits = ["art", "streetwear", "womensstreetwear", "OldSchoolCool", "cosplay", "malefashionadvice", "sneakers", "cats", "flowers", "gaming"]
dataset_path = "datasets/"
model_path = "model.json"
training_size_default = 50
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
				
				print(str(i) + ": " + post["url"] + " [" + post["score"] + "]")
				req = urllib2.Request(post["url"])
				req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0")
				try:
					f = urllib2.urlopen(req)
					with open(post["path"], "wb") as output:
					  output.write(f.read())
					posts.append(post)
					i += 1
				except urllib2.URLError, e:
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
		img = Image.open(dataset_path + pic).convert('RGB')
		new = ImageOps.fit(img, size, Image.ANTIALIAS)
		print(pic + str(new.size))
		path = dataset_path + os.path.splitext(pic)[0] + '.jpg'
		os.remove(dataset_path + pic) # to remove old png files, if converting
		new.save(path, 'JPEG')

def cleanup():
	if os.path.exists(model_path):
		os.remove(model_path)
	print("cleaning up training data...")
	for pic in os.listdir(dataset_path):
		os.remove(dataset_path + pic)	

#************************************ MAIN *************************************
if __name__ == "__main__":
	print("Executing program:")
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", action="store_true", help="cleanup training directory")
	parser.add_argument("-d", type=int, help="number of training examples per class")
	parser.add_argument("-p", type=int, help="preprocessing resolution of training example")
	args = parser.parse_args()
	if len(sys.argv) <= 1:
		download(training_size_default)
		preprocess(training_resolution_default)
	else:
		if args.c:
			cleanup()
		if args.d:
			download(args.d)
		if args.p:
			preprocess(args.p)