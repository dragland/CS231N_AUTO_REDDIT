#!//usr/bin/python
#Davy Ragland | dragland@stanford.edu
#Adrien Truong | aqtruong@stanford.edu 
#CS231N_REDDIT_NET | 2018

#*********************************** SETUP *************************************
import praw
import urllib2
import credentials

#************************************ MAIN *************************************
print("Executing program:")
reddit = praw.Reddit(client_id=credentials.CLIENT_ID, client_secret=credentials.CLIENT_SECRET, user_agent="CS231N_REDDIT_NET")

reddits = ["OldSchoolCool", "streetwear", "cosplay", "womensstreetwear", "malefashionadvice", "gaming"]
dataset_path = "datasets/"

for subreddit in reddits:
	print(subreddit + "......................................................")
	i = 0
	for submission in reddit.subreddit(subreddit).top("all", limit=50):
		filetype = submission.url.split(".")[-1]
		if filetype == "jpg" or filetype == "png" or filetype == "JPG" or filetype == "PNG":
			print(str(i) + ": " + submission.url + " [" + str(submission.score) + "]")
			path = dataset_path + subreddit + str(i) + "." + filetype
			f = urllib2.urlopen(submission.url)
			with open(path, "wb") as output:
			  output.write(f.read())
			i += 1