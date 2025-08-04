#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 19:52:15 2020
@author: Philip Auerbach
"""

import praw
import pandas as pd
import pytz
from datetime import datetime
from utils import *
import warnings
warnings.filterwarnings("ignore")

the_path = ""

subreddit_channel = 'MachineLearning+fishing'

reddit = praw.Reddit(
     client_id="Extension_Bat3486",
     client_secret="nEd-TEvsdfe3N-aZev18bI8LadeJqQ",
     user_agent="testscript by u/fakebot3",
     username="pja2113",
     password="kYkrac-sysha4-tyfbeq",
     check_for_async=False
 )

print(reddit.read_only)

def conv_time(var):
    tmp_df = pd.DataFrame()
    tmp_df = tmp_df.append(
        {'created_at': var},ignore_index=True)
    tmp_df.created_at = pd.to_datetime(
        tmp_df.created_at, unit='s').dt.tz_localize(
            'utc').dt.tz_convert('US/Eastern') 
    return datetime.fromtimestamp(var).astimezone(pytz.utc)

def get_reddit_data(var_in):
    import pandas as pd
    tmp_dict = pd.DataFrame()
    tmp_time = None
    try:
        tmp_dict = {'body': var_in.body}
    except:
        print ("ERROR")
        pass
    return tmp_dict
    
for comment in reddit.subreddit(subreddit_channel).stream.comments():
    tmp_df = get_reddit_data(comment)
    print (tmp_df["body"])