"""
Functions used for the detection of linguistic innovations in present day news items using Gensim's Word2Vec.

Marion Ruissen
Masterthesis
Computational Psycholinguistics,
University of Antwerp
Augustus 14, 2016
marionruissen@gmail.com
Supervisors:Walter Daelemans and Mike Kestemont

"""





#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import TreebankWordTokenizer 
import os
import gensim
from gensim import corpora, models, similarities
from gensim.models import Word2Vec
import numpy
import re
from collections import OrderedDict
from collections import Counter
import pickle
import json
import sys
from zipfile import ZipFile

# From Stackoverflow: function to get around repeating UnicodeEncodeErrors:
def uprint(*objects, sep=' ', end='\n', file=sys.stdout):
    enc = file.encoding
    if enc == 'UTF-8':
        print(*objects, sep=sep, end=end, file=file)
    else:
        f = lambda obj: str(obj).encode(enc, errors='backslashreplace').decode(enc)
        print(*map(f, objects), sep=sep, end=end, file=file)


import _compat_pickle
_compat_pickle.IMPORT_MAPPING.update({
    'UserDict': 'collections',
    'UserList': 'collections',
    'UserString': 'collections',
    'whichdb': 'dbm',
    'StringIO':  'io',
    'cStringIO': 'io',
})


"""
FOR TRAINING AND PREPROCESSING
"""


#open a txt-file en tokenize the content as lists of sentences with lists of words:
def txt_tokenizer(txt_file):
    with open(txt_file, encoding="utf8") as f:
        text = f.read().splitlines()
        text_tokenized = []
        alnum = re.compile(r"\w")
        for i in text:
                        
            # split paragraph per sentence
            sent_splitter = nltk.data.load('tokenizers/punkt/english.pickle') 
            i_sent_splitted = sent_splitter.tokenize(i)

            # split sentence per word                                 
            for sent in i_sent_splitted:
                cleaned_sent = []             
                tokenized_sent = TreebankWordTokenizer().tokenize(sent)
                for wordd in tokenized_sent:                 # remove all words containing only non-alphanumerics:
                    word = wordd.lower()
                    if alnum.match(word):
                        cleaned_sent.append(word)

                if len(cleaned_sent) > 4:                   # only sentences containing 5 or more words
                    text_tokenized.append(cleaned_sent)


        # return the paragraphs as a list (of sentences) of lists (of words)
        return(text_tokenized)

#open a xml-file and tokenize the content as lists of sentences with lists of words: 
def xml_tokenizer(xml_file): 
    with open(xml_file, 'rb') as text:
        soup = BeautifulSoup(text, "lxml")

    alnum = re.compile(r"\w")
    tokenized_txt = []
    for p in soup.find_all('body.content'):
        par = p.get_text(" ",strip=True)


        # split paragraph per sentence
        sent_splitter = nltk.data.load('tokenizers/punkt/english.pickle') 
        sent_splitted_par = sent_splitter.tokenize(par)

        # split sentence per word
        tokenized_par = []                                  
        for sentence in sent_splitted_par:                  
            tokenized_sent = TreebankWordTokenizer().tokenize(sentence)
            cleaned_sent = [] 
            for wordd in tokenized_sent: 
                word = wordd.lower()               
                if alnum.match(word):                    # remove all words containing only non-alphanumerics:
                    cleaned_sent.append(word)
            if len(cleaned_sent) > 4:                   # only sentences containing 5 or more words
                tokenized_par.append(cleaned_sent)
        
        tokenized_txt.append(tokenized_par)

    # return the paragraphs as a list (of sentences) of lists (of words)
    return(tokenized_txt)

#open a zip archive with xml-files and tokenize the contents as lists of sentences with lists of words:
def zipped_xml_tokenizer(zipfile):
    tokenized_txt = []
    alnum = re.compile(r"\w")
    with ZipFile(zipfile) as zf:
        for xml_filename in zf.namelist():
            t = zf.open(xml_filename)
            text = t.read()
            soup = BeautifulSoup(text, "xml")
            #print(soup)
            for p in soup.find_all('text'):
                par = p.get_text(" ",strip=True)
            
                # split paragraph per sentence
                sent_splitter = nltk.data.load('tokenizers/punkt/english.pickle') 
                sent_splitted_par = sent_splitter.tokenize(par)

                # split sentence per word
                tokenized_par = []                                  
                for sentence in sent_splitted_par:                  
                    tokenized_sent = TreebankWordTokenizer().tokenize(sentence)
                    cleaned_sent = [] 
                    for wordd in tokenized_sent: 
                        word = wordd.lower()               
                        if alnum.match(word):                    # remove all words containing only non-alphanumerics:
                            cleaned_sent.append(word)
                    if len(cleaned_sent) > 4:                   # only sentences containing 5 or more words
                        tokenized_par.append(cleaned_sent)
                    
                tokenized_txt.append(tokenized_par)
    return(tokenized_txt)


# iterator that opens all files from a dir and outputs all sentences one by one:
class SentenceIterator(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        for _file in os.listdir(self.dirname):
            fsentences = xml_tokenizer(os.path.join(self.dirname, _file))
            for par in fsentences:
                for s in par:
                    yield(s)

# iterator that opens all files from a zipped archive and outputs all sentences one by one:
class ZippedSentenceIterator(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        for _file in os.listdir(self.dirname):
            fsentences = zipped_xml_tokenizer(os.path.join(self.dirname, _file))
            for par in fsentences:
                for s in par:
                    yield(s)

"""
SEVERAL FUNCTIONS
"""

# input: raw UNKNOWN_WORDS dictionary, with all words per period not occourring in the training vocabulary
# output: cleaned UNKNOWN_WORDS dictionary: removed all 'words' containing other character than letters 
# and removed all words that occurred in previous periods of the dictionary
def unknown_words_cleaner(unknown_words_dict, outputname):
    UNKNOWN_WORDS = unknown_words_dict
    CLEANED_UNKNOWN_WORDS = {}

    true_word = re.compile(r"\b[^\W\d_]+\b")
    Known_words_total = {}
    known_words_total = set(Known_words_total)

    quarters = ["01_02_03","04_05_06","07_08_09","10_11_12"]
    #years = ["1999","2000","2001","2002","2003","2004"]
    years = ["2010","2011","2012","2013","2014","2015"]


    for year in years:
        for quarter in quarters:
            logging.info("Cleaning UNKNOWN_WORDS dictionary for year %s, quarter %s" %(year, quarter))                  
            quartername = "%s__%s" %(year, quarter)
            unk_words_quarter = []
            Known_words_quarter = {}
            known_words_quarter = set(Known_words_quarter)
            for word in UNKNOWN_WORDS[quartername]:
                if true_word.match(word):
                    if word not in known_words_total:
                        unk_words_quarter.append(word)
                known_words_quarter.add(word)
            CLEANED_UNKNOWN_WORDS[quartername] = unk_words_quarter
            known_words_total.update(known_words_quarter)
            temp_file_name = "%s_%s_temp" %(outputname, quartername)
            pickle.dump(CLEANED_UNKNOWN_WORDS, open(temp_file_name,"wb"))

    pickle.dump(CLEANED_UNKNOWN_WORDS, open(outputname,"wb"))
    return

# simple function to output the previous quarter of a year
def prev_quarter(quarter, year):
    if quarter is "01_02_03":
        previous_year = str(int(year)-1)
        previous_quarter = "%s__10_11_12" %(previous_year)
    if quarter is "04_05_06":
        previous_quarter = "%s__01_02_03" %(year)
    if quarter is "07_08_09":
        previous_quarter = "%s__04_05_06" %(year)
    if quarter is "10_11_12":
        previous_quarter = "%s__07_08_09" %(year)  
    return(previous_quarter)



# measures semantic change of a word in a dictionary between the first and the last timeslot
# source: http://stackoverflow.com/questions/21979970/how-to-use-word2vec-to-calculate-the-similarity-distance-by-giving-2-words
def semantic_change_simple(word, t1, t2):
    # t1 en t2 als jaar__kwartaal, bijv 1999__01_02_03 
    global MODEL_DICT
    model_t1 = MODEL_DICT[t1]
    model_t2 = MODEL_DICT[t2]
    semantic_change_simple = numpy.dot(model_t1[word], model_t2[word])/(numpy.linalg.norm(model_t1[word])* numpy.linalg.norm(model_t2[word]))
    return(semantic_change_simple)


# measures semantic change of a word between all timeslots (with mean en std as a value for stability) 
# only usable with computers with enough RAM to load all models simultaneously
def semantic_change(word, t1, t2):
    global quarters
    global years
    global MODEL_DICT    
    timeslots = []
    for year in years:
        for quarter in quarters:
            timeslot = "%s__%s" %(year, quarter)
            timeslots.append(timeslot)
    x = timeslots.index(t1)
    y = (timeslots.index(t2))+1
    needed_timeslots = timeslots[x:y]
    cosine_similarities = []
    for idx, t in enumerate(needed_timeslots):
        if t is needed_timeslots[0]:
            pass
        else:
            prev_t = timeslots[idx-1]
            model_1 = MODEL_DICT[prev_t]
            model_2 = MODEL_DICT[t]
            cosine_similarity = numpy.dot(model_1[word], model_2[word])/(numpy.linalg.norm(model_1[word])* numpy.linalg.norm(model_2[word]))
            cosine_similarities.append(cosine_similarity)
    mean = numpy.mean(cosine_similarities)
    std = numpy.std(cosine_similarities)
    return(cosine_similarities, ("mean:", mean), ("std:", std))

def some_statistics(t1, t2):
    cosine_similarities = []
    for word in model2.vocab:
        cosine_similarity = semantic_change_simple(word, t1, t2)
        cosine_similarities.append(cosine_similarity)
    number = len(cosine_similarities)
    mean = numpy.mean(cosine_similarities)
    std = numpy.std(cosine_similarities)
    rapport = "Some statistics about the semantic change between timeslot %s and timeslot %s:\nThe number of different words = %s\nThe mean of all cosine similarities = %s\nThe standard deviation of all cosine similarities = %s" %(t1, t2, number, mean, std)
    return(rapport)


# a function to output the top x most changed words in a dictionary
def top_x_most_changed(t2, t1="1999__01_02_03", x=10, threshold=100):
    semantic_changes = {}
    global MODEL_DICT
    function_words = pickle.load(open("Models\\function_words.pickle","rb"))
    stop_words = stopwords.words('dutch')
    t2_model = MODEL_DICT[t2]                               
    for word in t2_model.vocab:
        if word not in function_words and word not in stop_words:
            if t2_model.vocab[word].count > threshold:
                try:
                    semantic_change = semantic_change_simple(word, t1, t2)
                    semantic_changes[word] = semantic_change
                except KeyError:
                    pass
    sorted_dict_by_value = sorted((value,key) for (key,value) in semantic_changes.items())         
    most_changed = []
    for i in sorted_dict_by_value[:x]:                                                             
        try:
            most_changed.append(i)
        except UnicodeEncodeError:
            pass                                                  
    return(most_changed)


# a function to output the top x most changed words of all words that occur in ALL periods
def top_x_most_changed_extended(t2, t1, x=10, exp=1, threshold=100):
    semantic_changes = {}
    function_words = pickle.load(open("Models\\function_words.pickle","rb"))
    stop_words = stopwords.words('dutch')
    timeslot2_model = MODEL_DICT[timeslot2] 
    if exp == 1:
        words_in_all_periods = pickle.load(open("Models\\words_in_all_periods_exp1.pickle", "rb"))
    elif exp == 2:
        words_in_all_periods = pickle.load(open("Models\\words_in_all_periods_exp2.pickle", "rb"))
    for word in words_in_all_periods:
        if word not in function_words and word not in stop_words:
            if word in timeslot2_model.vocab and timeslot2_model.vocab[word].count > threshold:
                try:
                    semantic_change = semantic_change_simple(word, timeslot1, timeslot2)
                    semantic_changes[word] = semantic_change
                except KeyError:
                    pass
    sorted_dict_by_value = sorted((value,key) for (key,value) in semantic_changes.items())         
    most_changed = []
    for i in sorted_dict_by_value[:x]:
        try:
            most_changed.append(i)
        except UnicodeEncodeError:
        pass
    return(most_changed)



# a function to output the top x least changed words in a dictionary
def top_x_least_changed(t2, t1="1999__01_02_03", x=10):
    semantic_changes = {}
    global MODEL_DICT
    function_words = pickle.load(open("Models\\function_words.pickle","rb"))
    t2_model = MODEL_DICT[t2]
    for word in t2_model.vocab:
        if t2_model.vocab[word].count > 100:
            if len(word) >= 2:
                try:
                    if semantic_change_simple(word, t1, t2) < 1:
                        semantic_change = semantic_change_simple(word, t1, t2)
                        semantic_changes[word] = semantic_change
                except KeyError:
                    pass
    sorted_dict_by_value = sorted((value,key) for (key,value) in semantic_changes.items())
    least_changed = []
    for i in sorted_dict_by_value[-x:]:
        try:
            least_changed.append(i)
        except UnicodeEncodeError:
            pass
    uprint(least_changed)

# outputs the top K nearest neighbours of a word in a model, default K is 10
def nearest_neighbors(word, quartername, K=10):
    global MODEL_DICT
    model = MODEL_DICT[quartername]
    kNN = model.most_similar(positive=word, topn=K)
    return(kNN)



# outputs a list of top N most common (default 10) new words (not occurring in training vocabulary or previous periods)
def most_common_new_words(quartername, N=10):
    global CLEANED_UNKNOWN_WORDS
    unk_words = CLEANED_UNKNOWN_WORDS[quartername]
    counts = Counter(unk_words)
    return(counts.most_common(N))

# summarizing function, outputs a list of most interesting (new and most changed) words and their nearest neighbours:
def interesting_words(quartername2, N_newest_words=10, x_most_changed_words=10, K_nearest_neighbours=5, quartername1 ="1999__01_02_03"):
    year2 = quartername2[:4]
    quarter2 = quartername2[-8:]
    year1 = quartername1[:4]
    quarter1 = quartername1[-8:]
    global CLEANED_UNKNOWN_WORDS
    global MODEL_DICT
    global function_words
        #quartername = "%s__%s" %(year, quarter)
    #t1 = "%s__%s" %(year_comp, quarter_comp)
    #t2 = "%s__%s" %(year, quarter)
    most_changed = top_x_most_changed(quartername2, quartername1, x_most_changed_words)
    kNN_of_x_most_changed = {}
    for i in most_changed:
        most_changed_word = i[1]
        kNN = nearest_neighbours(most_changed_word, quartername2, K_nearest_neighbours)
        kNN_words = []
        for i in kNN:
            word = i[0]
            kNN_words.append(word)
        kNN_of_x_most_changed[most_changed_word] = kNN_words
    print("\n")
    print("\n")
    print("-------------------------------------------------------------------------")
    print("THE SEMANTICALLY MOST INTERESTING WORDS OF QUARTER %s OF YEAR %s:" %(quarter2, year2))
    print("\n")
    print("The %s most common new words and their frequencies are: " %(N_newest_words))
    for i in most_common_new_words(quartername2, N_newest_words):
        print("%s: %s" %(i[0], i[1]))
    print("\n")
    print("The %s semantically most changed words since quarter %s of year %s and their cosine similarities are:" %(x_most_changed_words, quarter1, year1))
    for i in most_changed:
        try:
            print("%s: %s" %(i[1], i[0]))
        except UnicodeEncodeError:
            uprint("%s: %s" %(i[1], i[0]))
    print("\n")
    print("The %s nearest neighbours of the %s semantically most changed words are:" %(K_nearest_neighbours, str(x_most_changed_words)))
    for i in kNN_of_x_most_changed:
        try:
            print("%s:" %i)
            print(kNN_of_x_most_changed[i])
        except UnicodeEncodeError:
            uprint("%s:" %i)
            uprint(kNN_of_x_most_changed[i])
    print("-------------------------------------------------------------------------")





