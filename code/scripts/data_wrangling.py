# -*- coding: utf-8 -*-
"""
Created on Sun May 29 13:09:19 2022

@author: Emmanuelle R Nunes
"""

import pandas as pd
import json
from textblob import TextBlob
from textstat.textstat import textstat

def load_data(app_id):
    # Data folder
    data_path = "data/"

    json_filename = "review_" + app_id + ".json"
    print(json_filename)

    data_filename = data_path + json_filename

    with open(data_filename, 'r', encoding="utf8") as in_json_file:
        review_data = json.load(in_json_file)

    return review_data

def describe_data(review_data):
    try:
        query_summary = review_data['query_summary']

        sentence = 'Number of reviews: {0} ({1} up ; {2} down)'
        sentence = sentence.format(query_summary["total_reviews"], 
                                   query_summary["total_positive"],
                                   query_summary["total_negative"])
    except KeyError:
        query_summary = None

        sentence = 'Query summary cannot be found in the JSON file.'

    print(sentence)

    reviews = list(review_data['reviews'].values())

    sentence = 'Number of downloaded reviews: ' + str(len(reviews))
    print(sentence)

    return query_summary, reviews


def aggregate_reviews(app_id):
    review_data = load_data(app_id)

    (_, reviews) = describe_data(review_data)

    review_stats = dict()

    ##

    # Review ID
    review_stats['recommendationid'] = []

    # Meta-data regarding the reviewers
    review_stats['num_games_owned'] = []
    review_stats['num_reviews'] = []
    review_stats['playtime_forever'] = []

    # Meta-data regarding the reviews themselves
    review_stats['language'] = []
    review_stats['voted_up'] = []
    review_stats['votes_up'] = []
    review_stats['votes_funny'] = []
    review_stats['weighted_vote_score'] = []
    review_stats['comment_count'] = []
    review_stats['steam_purchase'] = []
    review_stats['received_for_free'] = []

    # Stats regarding the reviews themselves
    review_stats['review'] = []
    review_stats['character_count'] = []
    review_stats['syllable_count'] = []
    review_stats['lexicon_count'] = []
    review_stats['sentence_count'] = []
    review_stats['difficult_words_count'] = []
    review_stats['flesch_reading_ease'] = []
    review_stats['dale_chall_readability_score'] = []

    # Sentiment analysis
    review_stats['polarity'] = []
    review_stats['subjectivity'] = []

    ##

    for review in reviews:
        review_content = review['review']

        # Review ID
        review_stats['recommendationid'].append(review["recommendationid"])

        # Meta-data regarding the reviewers
        review_stats['num_games_owned'].append(review['author']['num_games_owned'])
        review_stats['num_reviews'].append(review['author']['num_reviews'])
        review_stats['playtime_forever'].append(review['author']['playtime_forever'])

        # Meta-data regarding the reviews themselves
        review_stats['language'].append(review['language'])
        review_stats['voted_up'].append(review['voted_up'])
        review_stats['votes_up'].append(review['votes_up'])
        review_stats['votes_funny'].append(review['votes_funny'])
        review_stats['weighted_vote_score'].append(review['weighted_vote_score'])
        review_stats['comment_count'].append(review['comment_count'])
        review_stats['steam_purchase'].append(review['steam_purchase'])
        review_stats['received_for_free'].append(review['received_for_free'])

        # Stats regarding the reviews themselves
        review_stats['review'].append(review_content)
        review_stats['character_count'].append(len(review_content))
        review_stats['syllable_count'].append(textstat.syllable_count(review_content))
        review_stats['lexicon_count'].append(textstat.lexicon_count(review_content))
        review_stats['sentence_count'].append(textstat.sentence_count(review_content))
        review_stats['difficult_words_count'].append(textstat.difficult_words(review_content))
        try:
            review_stats['flesch_reading_ease'].append(textstat.flesch_reading_ease(review_content))
        except TypeError:
            review_stats['flesch_reading_ease'].append(None)
        review_stats['dale_chall_readability_score'].append(textstat.dale_chall_readability_score(review_content))

        # Sentiment analysis
        blob = TextBlob(review_content)
        review_stats['polarity'].append(blob.sentiment.polarity)
        review_stats['subjectivity'].append(blob.sentiment.subjectivity)

    return review_stats


def aggregate_reviews_to_pandas(app_ids):
    
    app_ids = list(app_ids)
    review_all = []
    index = 0
    
    for app_id in app_ids:
        print(app_id)
    
        review_stats = aggregate_reviews(app_id)
        review_all.append(review_stats)

        df = pd.DataFrame(data=review_stats)

        # Correction for an inconsistency which I discovered when running df.mean(). These 2 columns did not appear in the
        # output of mean(). I don't think it has any real impact for clustering and other purposes, but just to be sure...
        if "comment_count" in df.columns:
            df["comment_count"] = df["comment_count"].astype('int')
        if "weighted_vote_score" in df.columns:
            df["weighted_vote_score"] = df["weighted_vote_score"].astype('float')   
                
        df['app_id'] = app_id
        
        if index == 0:
            df_final = df
        else:
            df_final = pd.concat([df.reset_index(drop=True), df_final], axis=0)
        
        index = index + 1

    return df_final
