from __future__ import division
from string import punctuation
import numpy as np
from textblob import TextBlob

class StyleFeatures(dict):
    ''' 
    Retrieves features of writing style
    '''

    def find_freq(self,lst, search_item, normalizer):
        '''
        Finds normalized frequencies of tokens.
        Input: lst is a list of the item you are counting through, 
        search_item (list or array-like object) is the item you are counting,
        normalizer (also list, or array-like) is the object you are normalizing it by 
        (words,sentences, or punctuation usually).
        '''
        return lst.count(search_item) / len(normalizer)

    def find_freq_per_thousand(self,lst, search_item, normalizer):
        '''
        Same as function 'find_freq' except multiples normalizer by 1000
        Ex: finds frequency of commas (search_item) per 1000 punctuation_marks (normalizer)
        '''
        return (lst.count(search_item) / len(normalizer)) * 1000

    def __init__(self,article):
        article = TextBlob(article)
        #words = [word.singularize() for word in article.words]
        #sentences = article.sentences

        words = article.split()
        sentences = article.split('.')

        self['polarity'] = article.sentiment.polarity
        self['subjectivity'] = article.sentiment.subjectivity

        word_lens = [len(word) for word in words]
        sentence_lens = [len(sentence.split()) for sentence in sentences]
        punct = [char for char in article if char in punctuation]

        freq_items = {
                        'freq_question_marks':[punct,'?',sentences],\
                        'freq_exclamation_marks':[punct,'!',sentences],\
                        'freq_quotation_marks':[punct,'?',sentences]
                     }

        freq_items_per_thousand = {
                                'freq_commas':[punct,',',words], \
                                'freq_semi_colons':[punct,';',words],\
                                'freq_ands': [words, 'and', words],\
                                'freq_buts': [words, 'but', words],\
                                'freq_howevers': [words, 'however', words],\
                                'freq_ifs': [words, 'if', words],\
                                'freq_thats': [words, 'that', words],\
                                'freq_mores': [words, 'more', words],\
                                'freq_verys': [words, 'very', words]
                                }

        for item,params in freq_items.iteritems():
            self[item] = self.find_freq(params[0],params[1],params[2])

        for item,params in freq_items_per_thousand.iteritems():
            self[item] = self.find_freq_per_thousand(params[0],params[1],params[2])

        self['article_len'] = len(words)
        self['type_token_ratio'] = len(set(words)) / self['article_len']
        self['mean_word_len'] = np.mean(word_lens)
        self['mean_sentence_len'] = np.mean(sentence_lens)
        self['std_sentence_len'] = np.std(sentence_lens)
