#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import sys
import relation
from zipfile import ZipFile
import os.path
from thesis_functions import *


# TRAINING MODEL
model = gensim.models.Word2Vec.load('Models\\wiki-320-11-15-5-4.gensim')


# train the models, iterating over quarters of years and save them
quarters = ["01_02_03","04_05_06","07_08_09","10_11_12"]
years = ["2010", "2011", "2012", "2013","2014","2015"]
newspapers = ["ADN","DMO","HLN","PAR","VKN"]

# train and save the models for all periods:
MODEL_DICT = {}
for year in years:
    for quarter in quarters:
        for newspaper in newspapers:
            dirname = "Data\\persgroep\\%s\\%s\\%s" %(newspaper, year, quarter)
            if os.path.exists(dirname):
                sentences = ZippedSentenceIterator(dirname)                                            
                model.train(sentences)
        modelfilename = "Models\\%s__%s_exp2.model" %(year, quarter)
        model.save(modelfilename)
        quartername = "%s__%s" %(year, quarter)
        MODEL_DICT[quartername] = model

   


# Make raw dictionary UNKNOWN_WORDS for all periods:
model = gensim.models.Word2Vec.load('Models\\wiki-320-11-15-5-4.gensim')
UNKNOWN_WORDS = {}
for year in years:
    for quarter in quarters:
        logging.info("Building UNKNOWN_WORDS dictionary for year %s, quarter %s" %(year, quarter))
        unk_words = []
        for newspaper in newspapers:
            dirname = "Data\\persgroep\\%s\\%s\\%s" %(newspaper, year, quarter)
            if os.path.exists(dirname):
                sentences = ZippedSentenceIterator(dirname)          
                for sentence in sentences:
                    for word in sentence:
                        if word not in model.vocab:
                            unk_words.append(word)                    
        quartername = "%s__%s" %(year, quarter)
        UNKNOWN_WORDS[quartername] = unk_words

pickle.dump(UNKNOWN_WORDS, open("Models\\Unknown_words_dict_exp2.pickle","wb"))



# Clean the raw UNKNOWN_WORDS dictionary:
UNKNOWN_WORDS = pickle.load(open("Models\\Unknown_words_dict_exp2.pickle","rb"))
unknown_words_cleaner(UNKNOWN_WORDS, "Models\\Unknown_words_dict_exp2_cleaned.pickle")

CLEANED_UNKNOWN_WORDS = UNKNOWN_WORDS = pickle.load(open("Models\\Unknown_words_dict_exp2_cleaned.pickle","rb"))


# Make set of words occurring in ALL periods:
for year in years:
    for quarter in quarters:
        logging.info("Setting up all words per quarter for year %s, quarter %s" %(year, quarter))
        quartername = "%s__%s" %(year, quarter)
        quarterwords = set()
        for newspaper in newspapers:
            dirname = "Data\\persgroep\\%s\\%s\\%s" %(newspaper, year, quarter)
            if os.path.exists(dirname):
                sentences = ZippedSentenceIterator(dirname)                                                                          
                for sentence in sentences:
                    for word in sentence:
                        quarterwords.add(word)
        setfilename = "Models\\Temp\\quarterwords_%s__%s_exp2.pickle" %(year, quarter)
        pickle.dump(quarterwords, open(setfilename, "wb"))

periods = []
for year in years:
    for quarter in quarters:
        quartername = "%s__%s" %(year, quarter)
        periods.append(quartername)

period_sets = []
for period in periods:
    period_filename = "Models\\Temp\\quarterwords_%s_exp2.pickle" %(period)
    period_set = pickle.load(open(period_filename, "rb"))    
    period_sets.append(period_set)

words_in_all_periods = set.intersection(*period_sets)
pickle.dump(words_in_all_periods, open("Models\\words_in_all_periods_exp2.pickle", "wb"))
