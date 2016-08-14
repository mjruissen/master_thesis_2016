#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performing experiments for the detection of linguistic innovations in present day news items using Gensim's Word2Vec.

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
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import TreebankWordTokenizer 
import gensim
from gensim.models import Word2Vec
from gensim import corpora, models, similarities
import os
import numpy
import pickle
import json
import re
from collections import OrderedDict
from collections import Counter
import sys
import relation
import thesis_functions
from thesis_functions import *


# Load the training model:
#model = gensim.models.Word2Vec.load('Models\\wiki-320-11-15-5-4.gensim')


# Evaluating of the training model:
# Loads the category file for the Dutch relation test words.
#cats = json.load(open("Data\\semtest.json"))
# Load the predicates.
#rel = Relation("Data\\question-words.txt")
# in commandline: rel = relation.Relation("Data\\question-words.txt")
# Test the model
#rel.test_model(model)


# Specify timeslot 1 (earlier) en 2 (later) to compare, in the form of year__quarter (for example: 1999__01_02_03)
# Experiment 1 is in the period of 1999__01_02_03 untill 2004_10_11_12
# Experiment 2 is in the period of 2010__01_02_03 untill 2015__10_11_12
timeslot1 = ""
timeslot2 = ""

# Load the models:
#For EXP1:
model1 = gensim.models.Word2Vec.load("Models\\%s_Exp1.model" %(timeslot1))
model2 = gensim.models.Word2Vec.load("Models\\%s_Exp1.model" %(timeslot2))
#For Exp2:
#model1 = gensim.models.Word2Vec.load("Models\\%s_Exp2.model" %(timeslot1))
#model2 = gensim.models.Word2Vec.load("Models\\%s_Exp2.model" %(timeslot2))


# Create MODEL_DICT with needed models to use with functions from masterproef_functies
MODEL_DICT = {}
MODEL_DICT[timeslot1] = model1
MODEL_DICT[timeslot2] = model2

# Import cleaned dictionary with words per period that do not occur in the trainingvocabulary or in earlier timeslots:
# For Exp1:
CLEANED_UNKNOWN_WORDS = pickle.load(open("Models\\Unknown_words_dict_exp1_cleaned.pickle","rb"))
# For Exp2:
#CLEANED_UNKNOWN_WORDS = pickle.load(open("Models\\Unknown_words_dict_exp1_cleaned.pickle","rb"))


# put the logging to a lower level:
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)


## Several functions from masterproef_functies:

# measure semantic change of a word in a dictionary between the first and the last timeslot
print(semanic_change_simple("euro", timeslot1, timeslot2))

# a function to output the top x most/least changed words between two timeslotes in a dictionary
print(top_x_most_changed(timeslot2, timeslot1="1999__01_02_03", x=10))
print(top_x_least_changed(timeslot2, timeslot1="1999__01_02_03", x=10))

# a function to output the top x most changed words of the words occurring in ALL periods
print(top_x_most_changed_extended(timeslot2, timeslot1="2000__01_02_03", x=10, exp=1)

# outputs the de top K nearest neighbors of a word in a model, default K is 10
print(nearest_neighbors(word, timeslot2, K=10))

# outputs the N (default is 10) most used new words (not occurring in training data or in previous time slots) 
print(most_common_new_words(timeslot2, N=10))


# Summarizing function, outputs a list of most interesting (new and changed) words and their nearest neighbors
interesting_words(timeslot2, N_newest_words=10, x_most_changed_words=10, K_nearest_neighbors=5, timeslot1="1999__01_02_03")

