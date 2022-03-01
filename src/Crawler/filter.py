# -*- coding: utf-8 -*-
"""

@author: Gloria del Valle

This script allows filtering a Twitter dataset via identifier.

"""
import re
import tweepy
import time
import ssl
import argparse
import yaml
import numpy as np
import pandas as pd

from functools import partial
from requests.exceptions import Timeout, ConnectionError
from urllib.error import HTTPError
from urllib3.exceptions import ReadTimeoutError
from datetime import datetime

# Open config file for Twitter API authentication
with open(r"conf.yaml") as conf:
    conf = yaml.load(conf, Loader=yaml.FullLoader)

CONSUMER_KEY = conf["CONSUMER_KEY"]
CONSUMER_SECRET = conf["CONSUMER_SECRET"]
ACCESS_TOKEN = conf["ACCESS_TOKEN"]
ACCESS_SECRET = conf["ACCESS_SECRET"]

# Cleaning function for tweet text
repl = partial(re.sub, "( |\n|\t)+", " ")


def connect_to_twitter_OAuth():
    """Twitter API authenticator

    Returns
    -------
    api:
        Tweepy OAuth object
    """
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    api = tweepy.API(auth)

    return api


def extract_status_attributes(tweet):
    """Extracts all possible attributes given a tweet object

    Parameters
    ----------
    tweet:
        Tweet object

    Returns
    -------
    List of tweet attributes

    """
    rt, qt, rp = "", "", ""
    if (
        "retweeted_status" in tweet._json
        and tweet._json["retweeted_status"] is not None
    ):
        rt = tweet._json["retweeted_status"]
    if "quoted_status" in tweet._json and tweet._json["quoted_status"] is not None:
        qt = tweet._json["quoted_status"]
    if (
        "in_reply_to_screen_name" in tweet._json
        and tweet._json["in_reply_to_screen_name"] is not None
    ):
        rp = tweet._json["in_reply_to_screen_name"]

    return [
        str(tweet.user.id),
        str(tweet.user.screen_name),
        str(tweet.id),
        str("" if rt else repl(tweet.text)),
        str(tweet.created_at.timestamp()),
        str(tweet.favorite_count),
        str(tweet.retweet_count),
        str(rp),
        str("" if not rp else tweet.in_reply_to_status_id),
        str("" if not rp else tweet.in_reply_to_user_id),
        str(qt),
        str(
            ""
            if not qt and not hasattr(tweet, "quoted_status")
            else tweet.quoted_status.id
        ),
        str("" if not qt else tweet.quoted_status.user.id),
        str("" if not qt else repl(tweet.quoted_status.text)),
        str(
            ""
            if not qt
            else datetime.strptime(
                str(tweet.quoted_status.created_at), "%Y-%m-%d %H:%M:%S%z"
            ).timestamp()
        ),
        str("" if not qt else tweet.quoted_status.favorite_count),
        str("" if not qt else tweet.quoted_status.retweet_count),
        str(rt),
        str("" if not rt else tweet.retweeted_status.id),
        str("" if not rt else tweet.retweeted_status.user.id),
        str("" if not rt else repl(tweet.retweeted_status.text)),
        str("" if not rt else tweet.retweeted_status.created_at.timestamp()),
        str("" if not rt else tweet.retweeted_status.favorite_count),
        str("" if not rt else tweet.retweeted_status.retweet_count),
    ]


# Init parser
parser = argparse.ArgumentParser()

# Add params
parser.add_argument(
    "--file",
    action="store",
    dest="file",
    help="Path to dataset (csv)",
    required=True,
    type=str,
)
parser.add_argument(
    "--name",
    action="store",
    dest="name",
    help="New dataset name",
    required=True,
    type=str,
)

# Save args
args = parser.parse_args()

# API connection
api = connect_to_twitter_OAuth()

# Save dataset
df = pd.read_csv(args.file)

# Get ids and corresponding label (if needed)
ids = df.ID.to_list()

# labels = df.majority.to_list()

# Filter real ID status
ids = [i for i in ids if len(i) == len(ids[0])]

# New dataset name
name = args.name

# Init
starttime = time.time()
start, end = 0, 869
m, i = 870, 0
tweet_list = []

# 900 requests every 15 minutes
while True:
    print("Retrieving... {0}->{1}:{2}".format(i, start, end))
    for n in ids[start:end]:
        print("Status: ", n)

        # Save filtered dataset
        df = pd.DataFrame(
            tweet_list,
            columns=[
                "user_id",
                "screen_name",
                "tweet_id",
                "tweet_text",
                "tweet_creation_at",
                "tweet_fav_count",
                "tweet_rt_count",
                "is_reply",
                "reply_id_status",
                "reply_id_user",
                "is_quote",
                "quote_id_user",
                "quote_id_status",
                "quote_text",
                "quote_creation_at",
                "quote_fav_count",
                "quote_rt_count",
                "is_rt",
                "rt_id_user",
                "rt_id_status",
                "rt_text",
                "rt_creation_at",
                "rt_fav_count",
                "rt_rt_count",
                # "label",
            ],
        )

        # Export to csv
        df.to_csv(name + ".csv", index=False)

        # Extract attributes and catch exceptions
        try:
            tweet = api.get_status(int(n))
            attr_list = extract_status_attributes(tweet)
            # attr_list.append(labels[i])
            tweet_list.append(attr_list)
        except (
            Timeout,
            ssl.SSLError,
            ReadTimeoutError,
            ConnectionError,
            HTTPError,
            tweepy.errors.TooManyRequests,
        ) as e:
            print("Timeout: {0}, error {1}".format(n, e))
            time.sleep(180)
            continue
        except (tweepy.errors.NotFound, tweepy.errors.Forbidden) as e:
            print("Tweepy error: {0}, error {1}".format(n, e))
            continue

    # Control
    i = i + 1
    start = end + 1
    end = start + m
    if end > len(ids):
        end = len(ids)
    if start >= len(ids):
        break
    time.sleep(900.0 - ((time.time() - starttime) % 900.0))
