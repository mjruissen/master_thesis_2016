#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Buiding and saving models for the detection of linguistic innovations in present day news items using Gensim's Word2Vec.
Experiment 1: data from the Flemish newspaper De Morgen, 1999 - 2004

Marion Ruissen
Masterthesis
Computational Psycholinguistics,
University of Antwerp
Augustus 14, 2016
marionruissen@gmail.com
Supervisors:Walter Daelemans and Mike Kestemont

"""



import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import sys
#sys.path.insert(0, "C:\\Users\\Marion\\Desktop\\Masterproef\\Python_bestanden")	for command line
import relation
from thesis_functions import *

# TRAINING MODEL
model = gensim.models.Word2Vec.load('Models\\wiki-320-11-15-5-4.gensim')




# train the models, iterating over quarters of years and save them
quarters = ["01_02_03","04_05_06","07_08_09","10_11_12"]
years = ["1999","2000", "2001","2002","2003","2004"]

# train and save the models for all periods:
MODEL_DICT = {}
for year in years:
    for quarter in quarters:
        dirname = "Data\\DeMorgen-%s\\%s" %(year, quarter)
        sentences = SentenceIterator(dirname)                                 
        quartername = "%s__%s" %(year, quarter)
        model.train(sentences)
        modelfilename = "Models\\%s__%s_exp1.model" %(year, quarter)
        model.save(modelfilename)
        MODEL_DICT[quartername] = model
   

# Make raw dictionary UNKNOWN_WORDS for all periods:
UNKNOWN_WORDS = {}
for year in years:
    for quarter in quarters:
        logging.info("Building UNKNOWN_WORDS dictionary for year %s, quarter %s" %(year, quarter))
        unk_words = []
        dirname = "Data\\DeMorgen-%s\\%s" %(year, quarter)
        sentences = SentenceIterator(dirname)
        for sentence in sentences:
            for word in sentence:
                if word not in model.vocab:
                    unk_words.append(word)                    
        quartername = "%s__%s" %(year, quarter)
        UNKNOWN_WORDS[quartername] = unk_words

pickle.dump(UNKNOWN_WORDS, open("Models\\Unknown_words_dict_exp1.pickle","wb"))

# Clean the raw UNKNOWN_WORDS dictionary:
UNKNOWN_WORDS = pickle.load(open("Models\\Unknown_words_dict_exp1.pickle","rb"))
unknown_words_cleaner(UNKNOWN_WORDS, "Models\\Unknown_words_dict_exp1_cleaned.pickle")

CLEANED_UNKNOWN_WORDS = pickle.load(open("Models\\Unknown_words_dict_exp1_cleaned.pickle","rb"))


# Make set of words occurring in ALL periods:
for year in years:
    for quarter in quarters:
        quartername = "%s__%s" %(year, quarter)
        logging.info("Setting up all words per quarter for year %s, quarter %s" %(year, quarter))
        quarterwords = set()
        dirname = "Data\\DeMorgen-%s\\%s" %(year, quarter)
        sentences = SentenceIterator(dirname)                                 
        for sentence in sentences:
            for word in sentence:
                quarterwords.add(word)
        setfilename = "Models\\Temp\\quarterwords_%s__%s.pickle" %(year, quarter)
        pickle.dump(quarterwords, open(setfilename, "wb"))

periods = []
for year in years:
    for quarter in quarters:
        quartername = "%s__%s" %(year, quarter)
        periods.append(quartername)

period_sets = []
for period in periods:
    period_filename = "Models\\Temp\\quarterwords_%s.pickle" %(period)
    period_set = pickle.load(open(period_filename, "rb"))    
    period_sets.append(period_set)

words_in_all_periods = set.intersection(*period_sets)
pickle.dump(words_in_all_periods, open("Models\\words_in_all_periods_exp1.pickle", "wb"))
