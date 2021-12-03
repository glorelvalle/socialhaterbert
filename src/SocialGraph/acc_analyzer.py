from __future__ import unicode_literals

import es_core_news_sm
import tweepy
import numpy
import argparse
import collections
import datetime
import re
import json
import sys
import os
import spacy
import nltk
import glob
import operator
import time
import itertools
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from transformers import pipeline
from ascii_graph import Pyasciigraph
from ascii_graph.colors import Gre, Yel, Red
from ascii_graph.colordata import hcolor
from spacy.lang.es.examples import sentences
from multiprocessing import Process
from tqdm import tqdm
from functools import partial
#from watson_developer_cloud import VisualRecognitionV3
from subject_classification_spanish import subject_classifier
from spellchecker import SpellChecker

cats_classifier = subject_classifier.SubjectClassifier()

#visual_recognition = VisualRecognitionV3(
#    '2018-03-19',
#    iam_apikey='-')

nlp_ = es_core_news_sm.load()
stopwords_ = set(stopwords.words('spanish'))

repl = partial(re.sub, '( |\n|\t)+', ' ')

leet_alph = {'0':'o', '1':'i', '3':'e', '5':'s', '4':'a', '7':'t', '8': 'b'}
regex = "/*[a-zA-ZñÑ$]*[0134578$][a-zA-ZñÑ]*"
regex2 = "https://t.co/\w*"
regex3 = "@[a-zA-Z0-9]*"
regex4 = "#[a-zA-Z0-9]*"
spell = SpellChecker(language='es')

#prog_ = re.compile("(@[A-Za-z0-9]+)|([^0-9A-Za-z' \t])|(\w+:\/\/\S+)")
#prog2_ = re.compile(" +")

#hashtags_ = re.compile("#(\w+)")
#regex_mentions_ = re.compile("@(\w+)")
#urls_ = re.compile("http(s)?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

regex_bad_words_ = re.compile("(" + "|".join(pd.concat([pd.read_csv(f) for f in glob.glob('lexicon/*.txt')], ignore_index=True)["termino"].values) + ")")

path_ = "finiteautomata/beto-sentiment-analysis"
classify_ = pipeline("sentiment-analysis", model=path_, tokenizer=path_)

path2_ = 'models_saved/loocv_bert-base-spanish-wwm-cased_Spanish_translated_baseline_100'
classify2_ = pipeline("sentiment-analysis", model=path2_, tokenizer=path2_)

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

# Twitter API keys
consumer_key = "-"
consumer_secret = "-"
access_token = "-"
access_token_secret = "-"

# global variables used to store data for the execution of program
start_date = 0
end_date = 0
user_data = {}

export = ""


parser = argparse.ArgumentParser(description=
    "Twitter Profile Analyzer",usage='%(prog)s -n <screen_name> [options]')
parser.add_argument('-l', '--limit', metavar='N', type=int, default=1000,
                    help='limit the number of tweets to retreive (default=1000)')
parser.add_argument('-n', '--name', required=True, metavar="screen_name",
                    help='target screen_name')

#parser.add_argument('-f', '--filter', help='filter by source (ex. -f android will get android tweets only)')

parser.add_argument('--no-timezone', action='store_true',
                    help='removes the timezone auto-adjustment (default is UTC)')

parser.add_argument('--utc-offset', type=int,
                    help='manually apply a timezone offset (in seconds)')



parser.add_argument('--no-retweets', action='store_true',
                    help='does not evaluate retweets')


args = parser.parse_args()


activity_hourly = {
    ("%2i:00" % i).replace(" ", "0"): 0 for i in range(24)
}

activity_weekly = {
    "%i" % i: 0 for i in range(7) #7 as there are 7 days in a week
}
# Initializing default values
detected_langs = collections.Counter()
detected_sources = collections.Counter()
detected_places = collections.Counter()
geo_enabled_tweets = 0
detected_hashtags = collections.Counter()
detected_domains = collections.Counter()
detected_timezones = collections.Counter()
retweets = 0
retweeted_users = collections.Counter()
mentioned_users = collections.Counter()
id_screen_names = {}
negativos, positivos, neutros = 0, 0, 0
hate, no_hate = 0, 0
baddies = []
n_baddies = 0
len_status = 0
times_user_quotes= 0
times_user_rt = 0
times_user_replies = 0
num_rts_to_tweets = 0
num_quotes_to_tweets = 0
num_favs_to_tweets = 0
num_replies_to_tweets = 0
neg_score = 0
pos_score = 0
neu_score = 0
no_hate_score = 0
hate_score = 0
user_categorias = {}
leet_counter = 0
misspelling_counter = 0

def leet_converter(word):
    """ Processing a leet token """
    for k, v in leet_alph.items():
        word = word.replace(k,v)
    return word

def process_tweet(tweet):
    """ Processing a single Tweet and updating our datasets """
    global start_date
    global end_date
    global geo_enabled_tweets
    global retweets
    global negativos
    global positivos
    global neutros
    global no_hate
    global hate
    global baddies
    global n_baddies
    global len_status
    global times_user_quotes
    global times_user_rt
    global times_user_replies
    global num_rts_to_tweets
    global num_quotes_to_tweets
    global num_favs_to_tweets
    global num_replies_to_tweets
    global neg_score
    global pos_score
    global neu_score
    global no_hate_score
    global hate_score
    global user_categorias
    global leet_counter
    global misspelling_counter

    if args.no_retweets:
        if hasattr(tweet, 'retweeted_status'):
            return
        if hasattr(tweet, 'is_quote_status') and tweet.is_quote_status:
            return

    tw_date = tweet.created_at

    # Updating most recent tweet
    end_date = end_date or tw_date
    start_date = tw_date

    # Handling retweets
    try:
        rt_id_user = tweet.retweeted_status.user.id_str
        retweeted_users[rt_id_user] += 1

        if tweet.retweeted_status.user.screen_name not in id_screen_names:
            id_screen_names[rt_id_user] = "@%s" % tweet.retweeted_status.user.screen_name

        retweets += 1
    except:
        pass

    # Adding timezone from profile offset to set to local hours
    if tweet.user.utc_offset and not args.no_timezone:
        tw_date = (tweet.created_at + datetime.timedelta(seconds=tweet.user.utc_offset))

    if args.utc_offset:
        tw_date = (tweet.created_at + datetime.timedelta(seconds=args.utc_offset))

    # Updating our activity datasets (distribution maps)
    activity_hourly["%s:00" % str(tw_date.hour).zfill(2)] += 1
    activity_weekly[str(tw_date.weekday())] += 1

    # Updating langs
    detected_langs[tweet.lang] += 1

    # Updating sources
    detected_sources[tweet.source] += 1

    # Sentiment / hate
    if tweet.text:
        entrada = str(tweet.text)
        entrada = entrada.split(" ")
        new_entrada = []
        for w in entrada:
            if len(w) > 3:
                url = re.findall(regex2, w)
                us_mention = re.findall(regex3, w)
                us_hashtag = re.findall(regex4, w) # v 2.0
                if url or us_mention or us_hashtag:
                    w = ""
                leetword = re.findall(regex, w)
                if leetword:
                    w = leet_converter(leetword[0])
                    leet_counter += 1
            new_entrada.append(w)

        salida = []
        misspelled = spell.unknown(new_entrada)
        for word in new_entrada:
            re.sub(r"[^\w\s]", "", str(word))
            for m in misspelled:
                if len(word) > 3 and word == m:
                    word = spell.correction(m)
                    misspelling_counter += 1
            salida.append(word)

        entrada = " ".join(salida)

        result = classify_(str(entrada))
        result2 = classify2_(str(entrada))

        if result[0]['label'] == 'NEG':
            negativos += 1
            neg_score += float(result[0]['score'])
        elif result[0]['label'] == 'POS':
            positivos += 1
            pos_score += float(result[0]['score'])
        elif result[0]['label'] == 'NEU':
            neutros += 1
            neu_score += float(result[0]['score'])
        if result2[0]['label'] == 'LABEL_0':
            no_hate += 1
            no_hate_score += float(result[0]['score'])
        elif result2[0]['label'] == 'LABEL_1':
            hate += 1
            hate_score += float(result[0]['score'])

        # More textual data
        baddies += regex_bad_words_.findall(str(entrada))
        n_baddies = len(baddies)
        len_status += len(str(entrada))
        if "quote_count" in tweet._json:
            num_quotes_to_tweets += tweet.quote_count
        num_rts_to_tweets += tweet.retweet_count
        num_favs_to_tweets += tweet.favorite_count
        if "reply_count" in tweet._json:
            num_replies_to_tweets += tweet.reply_count

        classes_result = cats_classifier.classify(str(entrada))
        user_categorias.update(dict(itertools.islice(classes_result.items(), 3)))


    if tweet.is_quote_status:
        times_user_quotes += 1

    if "retweeted_status" in tweet._json:
        times_user_rt += 1

    if "in_reply_to_user_id" in tweet._json and tweet.in_reply_to_user_id != "":
        times_user_replies += 1

    # Detecting geolocation
    if tweet.place:
        geo_enabled_tweets += 1
        tweet.place.name = tweet.place.name
        detected_places[tweet.place.name] += 1

    # Updating hashtags list
    if tweet.entities['hashtags']:
        for ht in tweet.entities['hashtags']:
            ht['text'] = "#%s" % ht['text']
            detected_hashtags[ht['text']] += 1

    # Updating domains list
    if tweet.entities['urls']:
        for url in tweet.entities['urls']:
            domain = urlparse(url['expanded_url']).netloc
            if domain != "twitter.com":  # removing twitter.com from domains (not very relevant)
                detected_domains[domain] += 1

    # Updating mentioned users list
    if tweet.entities['user_mentions']:
        for ht in tweet.entities['user_mentions']:
            mentioned_users[ht['id_str']] += 1
            if not ht['screen_name'] in id_screen_names:
                id_screen_names[ht['id_str']] = "@%s" % ht['screen_name']


def get_tweets(api, username, limit):
    """ Download Tweets from username account """
    for status in tqdm(tweepy.Cursor(api.user_timeline, screen_name=username).items(limit), unit="tw", total=limit):
        process_tweet(status)


def int_to_weekday(day):
    weekdays = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
    return weekdays[int(day) % len(weekdays)]


def main():
	# Tweepy authentication
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    twitter_api = tweepy.API(auth)
    # Date
    now = datetime.datetime.now()

    # Getting general account's metadata
    print("[+] Getting @%s account data..." % args.name)
    user_data['user_name'] = args.name
    #getting user account data
    user_info = twitter_api.get_user(screen_name=args.name)

    #print("[+] lang           : %s" % user_info.lang)
    #print("[+] geo_enabled    : %s" % user_info.geo_enabled)
    #print("[+] time_zone      : %s" % user_info.time_zone)
    #print("[+] utc_offset     : %s" % user_info.utc_offset)
    print("[+] user_id        : %s" % user_info.id)
    #language of tweets
    #user_data['user_lang'] = user_info.lang
    #Location of Tweets(if tweets are Geo-Enabled)
    user_data['user_id'] = user_info.id
    user_data['user_geo_enabled'] = user_info.geo_enabled
    user_data['profile_image_url'] = user_info.profile_image_url

    #classes_result = visual_recognition.classify(url=user_info.profile_image_url).get_result()
    #user_data['categories_profile_image_url'] = str(classes_result['images'][0]['classifiers'][0]['classes'])

    # Number of Tweets
    #print("[+] statuses_count : %s" % user_info.statuses_count)
    user_data['status_count'] = user_info.statuses_count

    # Retreive all Tweets from account / max limit of API
    num_tweets = numpy.amin([args.limit, user_info.statuses_count])
    print("[+] Retrieving last %d tweets..." % num_tweets)
    user_data['status_retrieving'] = num_tweets

    # Download tweets
    get_tweets(twitter_api, args.name, limit=num_tweets)
    print("[+] Downloaded %d tweets from %s to %s (%d days)" % (num_tweets, start_date, end_date, (end_date - start_date).days))
    user_data['status_start_date'] = "%s" % start_date
    user_data['status_end_date'] = "%s" % end_date
    user_data['status_days'] = "%s" % (end_date - start_date).days

    # Checking if we have enough data (considering it's good to have at least 30 days of data)
    if (end_date - start_date).days < 30 and (num_tweets < user_info.statuses_count):
         print("[!] Looks like we do not have enough tweets from user, you should consider retrying (--limit)")
         user_data['status_note'] = "Looks like we do not have enough tweets from user, you should consider retrying (--limit)"
    else:
        user_data['status_note'] = str("")

    if (end_date - start_date).days != 0:
        print("[+] Average number of tweets per day: %.1f" % (num_tweets / float((end_date - start_date).days)))
        user_data['status_average_tweets_per_day'] = (num_tweets / float((end_date - start_date).days))
    else:
        user_data['status_average_tweets_per_day'] = str("")

    # Activity info
    for k, v in activity_hourly.items():
        user_data["activity_hourly_"+str(k)] = v
    for k, v in activity_weekly.items():
        user_data["activity_weekly_"+str(k)] = v


    #print("[+] Detected languages (top 5)")
    detected_langs1 = dict(sorted(detected_langs.items(), key=operator.itemgetter(1),reverse=True))
    user_data["top_languages"] = str(dict(itertools.islice(detected_langs1.items(), 5)))

    #print("[+] Detected sources (top 10)")
    detected_sources1 = dict(sorted(detected_sources.items(), key=operator.itemgetter(1),reverse=True))
    user_data["top_sources"] = str(dict(itertools.islice(detected_sources1.items(), 10)))

    #print("[+] There are %d geo enabled tweet(s)" % geo_enabled_tweets)
    user_data['geo_enabled_tweet_count'] = geo_enabled_tweets

    if len(detected_places) != 0:
        #print("[+] Detected places (top 10)")
        detected_places1 = dict(sorted(detected_places.items(), key=operator.itemgetter(1),reverse=True))
        user_data["top_places"] = str(dict(itertools.islice(detected_places1.items(), 10)))
    else:
        user_data["top_places"] = str("")

    #print("[+] Num hashtags {0}".format(len(detected_hashtags)))
    user_data["num_hashtags"] = len(detected_hashtags)

    #print("[+] Top 10 hashtags")
    detected_hashtags1 = dict(sorted(detected_hashtags.items(), key=operator.itemgetter(1),reverse=True))
    user_data["top_hashtags"] = str(dict(itertools.islice(detected_hashtags1.items(), 10)))

    if not args.no_retweets:
        #print("[+] @%s did %d RTs out of %d tweets (%.1f%%)" % (args.name, retweets, num_tweets, (float(retweets) * 100 / num_tweets)))
        user_data['rt_count'] = retweets
        # Converting users id to screen_names
        retweeted_users_names = {}
        for k in retweeted_users.keys():
            retweeted_users_names[id_screen_names[k]] = retweeted_users[k]

        #print("[+] Top 5 most retweeted users")
        retweeted_users_names1 = dict(sorted(retweeted_users_names.items(), key=operator.itemgetter(1),reverse=True))
        user_data["top_retweeted_users"] = str(dict(itertools.islice(retweeted_users_names1.items(), 5)))

    mentioned_users_names = {}
    for k in mentioned_users.keys():
        mentioned_users_names[id_screen_names[k]] = mentioned_users[k]
    #print("[+] Top 5 most mentioned users")
    mentioned_users_names1 = dict(sorted(mentioned_users_names.items(), key=operator.itemgetter(1),reverse=True))
    #print("[+] Number of mentions {0}".format(len(mentioned_users)))
    user_data['num_mentions'] = len(mentioned_users)
    user_data["top_mentioned_users"] = str(dict(itertools.islice(mentioned_users_names1.items(), 5)))

    #print("[+] # URLs: {0}".format(len(detected_domains)))
    user_data["num_urls"] = len(detected_domains)
    #print("[+] Most referenced domains (from URLs)")
    #print_stats(detected_domains, top=6)
    detected_domains1 = dict(sorted(detected_domains.items(), key=operator.itemgetter(1),reverse=True))
    user_data["top_referenced_domains"] = str(dict(itertools.islice(detected_domains1.items(), 6)))

    #print("[+] # NEG: {0}, # POS {1}, # NEU {2}".format(negativos, positivos, neutros))
    #print("[+] # HATE: {0}, # NO HATE {1}".format(hate, no_hate))
    user_data["negativos"] = negativos
    user_data["positivos"] = positivos
    user_data["neutros"] = neutros
    user_data["hate"] = hate
    user_data["no_hate"] = no_hate
    try:
        user_data["negativos_score"] = neg_score/negativos
    except ZeroDivisionError:
        user_data["negativos_score"] = 0
    try:
        user_data["positivos_score"] = pos_score/positivos
    except ZeroDivisionError:
        user_data["positivos_score"] = 0
    try:
        user_data["neutros_score"] = neu_score/neutros
    except ZeroDivisionError:
        user_data["neutros_score"] = 0
    try:
        user_data["hate_score"] = hate_score/hate
    except ZeroDivisionError:
        user_data["hate_score"] = 0
    try:
        user_data["no_hate_score"] = no_hate_score/no_hate
    except ZeroDivisionError:
        user_data["no_hate_score"] = 0
    try:
        hater_cond = hate/100*int(user_data['status_retrieving'])
    except ZeroDivisionError:
        hater_cond = 0

    if hater_cond >= 10 and negativos > positivos:
        user_data['is_hater'] = 1
    else:
        user_data['is_hater'] = 0

    #print("[+] # Baddies: {0}, # Baddies/tweet: {1}, Longitud_status/tweet: {2}".format(n_baddies, n_baddies/num_tweets, len_status/num_tweets))
    b = str(baddies)
    user_data["baddies"] = b
    user_data["n_baddies"] = n_baddies
    try:
        user_data["n_baddies_tweet"] = float(n_baddies/num_tweets)
    except ZeroDivisionError:
        user_data["n_baddies_tweet"] = 0

    user_data["len_status"] = len_status/num_tweets

    #print("[+] # Quotes: {0}, # RTs: {1}, # Replies: {2}".format(times_user_rt, times_user_quotes, times_user_replies))
    user_data["times_user_rt"] = times_user_rt
    user_data["times_user_quotes"] = times_user_quotes
    #user_data["times_user_replies"] = times_user_replies ### no funciona bien

    #print(user_data)
    #print("[+] # num_rts_to_tweets: {0}, # num_favs_to_tweets: {1}".format(num_rts_to_tweets, num_favs_to_tweets))
    #user_data["num_quotes_to_tweets"] = num_quotes_to_tweets
    user_data["num_rts_to_tweets"] = num_rts_to_tweets
    user_data["num_favs_to_tweets"] = num_favs_to_tweets
    #user_data["num_replies_to_tweets"] = num_replies_to_tweets
    user_data["top_categories"] = str(dict(itertools.islice(user_categorias.items(), 15)))
    user_data['misspelling_counter'] = int(misspelling_counter)
    user_data['leet_counter'] = int(leet_counter)


    diccionario = pd.DataFrame([user_data])
    diccionario.to_csv("SES_odio/"+str(args.name)+".csv", index=False)



if __name__ == '__main__':
    try:
        main()
    except tweepy.error.TweepError as e:
        print("[!] Twitter error: %s" % e)
    except Exception as e:
        print("[!] Error: %s" % e)
