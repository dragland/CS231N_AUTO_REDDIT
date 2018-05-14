#!//usr/bin/python
#Davy Ragland | dragland@stanford.edu
#Adrien Truong | aqtruong@stanford.edu 
#CS231N_REDDIT_NET | 2018

#*********************************** SETUP *************************************
import praw
import credentials

#************************************ MAIN *************************************
print("Executing program:")
reddit = praw.Reddit(client_id=credentials.CLIENT_ID, client_secret=credentials.CLIENT_SECRET, user_agent='CS231N_REDDIT_NET')

reddits = ["OldSchoolCool", "streetwear", "cosplay", "womensstreetwear", "malefashionadvice", "gaming"]

for subreddit in reddits:
	print(subreddit + "......................................................")
	for submission in reddit.subreddit(subreddit).top(limit=10):
		print(submission.title)
		print(submission.url)