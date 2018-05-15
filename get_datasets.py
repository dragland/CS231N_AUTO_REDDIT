#!//usr/bin/python
#Davy Ragland | dragland@stanford.edu
#Adrien Truong | aqtruong@stanford.edu 
#CS231N_REDDIT_NET | 2018

#*********************************** SETUP *************************************
import praw
import urllib2
import credentials
import PIL
from PIL import Image, ImageOps
import os
import sys
import argparse

reddits = ["OldSchoolCool", "streetwear", "cosplay", "womensstreetwear", "malefashionadvice", "gaming", "sneakers"]
dataset_path = "datasets/"
training_size_deafault = 50
training_resolution_deafault = 1000

#*********************************** HELPERS ***********************************
def download(training_size):
	print("Downloading training data...")
	reddit = praw.Reddit(client_id=credentials.CLIENT_ID, client_secret=credentials.CLIENT_SECRET, user_agent="CS231N_REDDIT_NET")
	for subreddit in reddits:
		print(subreddit + "......................................................")
		i = 0
		for submission in reddit.subreddit(subreddit).top("all"):
			filetype = submission.url.split(".")[-1]
			if filetype == "jpg" or filetype == "png" or filetype == "JPG" or filetype == "PNG":
				print(str(i) + ": " + submission.url + " [" + str(submission.score) + "]")
				path = dataset_path + subreddit + str(i) + "." + filetype
				req = urllib2.Request(submission.url)
				req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0")
				f = urllib2.urlopen(req)
				with open(path, "wb") as output:
				  output.write(f.read())
				i += 1
			if i == training_size:
				break

def preprocess(training_resolution):
	print("pre-processing training data...")
	size = training_resolution, training_resolution
	for pic in os.listdir(dataset_path):
		img = Image.open(dataset_path + pic).convert('RGB')
		new = ImageOps.fit(img, size, Image.ANTIALIAS)
		print(pic + str(new.size))
		new.save(dataset_path + pic)

def cleanup():
	print("cleaning up training data...")
	for pic in os.listdir(dataset_path):
		os.remove(dataset_path + pic)	

#************************************ MAIN *************************************
if __name__ == "__main__":
	print("Executing program:")
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", type=int, help="number of training examples per class")
	parser.add_argument("-p", type=int, help="preprocessing resolution of training example")
	parser.add_argument("-c", action="store_true", help="cleanup training directory")
	args = parser.parse_args()
	if len(sys.argv) <= 1:
		download(training_size_deafault)
		preprocess(training_resolution_deafault)
	else:
		if args.d:
			download(args.d)
		if args.p:
			preprocess(args.p)
		if args.c:
			cleanup()