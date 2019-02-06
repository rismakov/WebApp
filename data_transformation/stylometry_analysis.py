from __future__ import division

import numpy as np

from string import punctuation
from textblob import TextBlob

from data_transformation.stylometry_constants import (
    freq_punctuation_per_sentence, freq_punctuation_per_word, freq_token_types_per_word, INVALID_URLS, LINK_TYPES
)
from utils.constants import IMAGE_TERMS, MBTI_TYPES_UPPER
from utils.utils import (
    get_count_of_characters_in_text, get_freq_of_characters_in_text_in_list, get_freq_of_items_in_list
)


class StyleFeatures(object):
    '''Retrieves features of writing style. This includes frequency of certain tokens, frequency of various punctuation
    marks, subjectivity/polarity of the text, etc.
    '''

    def __init__(self, posts):
        self.stylometry_markers = {}
        self.posts = posts

        self.textblob_posts = TextBlob(self.get_posts_with_periods_at_ends())

        self.sentences = self.get_sentences()
        self.words = self.get_words()
        self.separate_links_from_words()

        self.track_number_of_words_and_sentences()

        self.posts_removing_links = self.remove_links_from_text(self.posts)
        self.all_punctuation_in_text = [char for char in self.posts_removing_links if char in punctuation]

    def get_all_featurized_features(self):
        self.add_parts_of_speech_frequencies_to_stylometry_markers()
        self.add_frequencies_of_punctuation_tokens_per_sentence()
        self.add_frequencies_of_punctuation_tokens_per_word()
        self.add_frequencies_of_tokens_to_stylometry_markers()
        self.add_frequencies_of_link_types_to_stylometry_markers()
        self.add_avg_lens_to_stylometry_markers()
        self.add_sentiment_analysis_to_stylometry_markers()
        self.add_frequency_of_emojis_and_emoticons_to_stylometry_markers()
        self.add_capitilization_info_to_stylometry_markers()

        return self.stylometry_markers

    def track_number_of_words_and_sentences(self):
        self.num_of_posts = len(self.posts.split('|||'))
        self.num_of_words = len(self.words)
        self.num_of_non_link_words = len(self.non_link_words)
        self.num_of_sentences = len(self.sentences)

    def get_posts_with_periods_at_ends(self):
        '''This adds periods to the end of posts which are missing punctuation at the end. This is so TextBlob will be
        able to extract sentences from the text properly.
        '''
        split_posts = self.posts.split('|||')

        for i, post in enumerate(split_posts):
            if post:
                if post[-1] not in ['?', '!', '.']:
                    split_posts[i] = post + '.'
        return ' '.join(split_posts)

    def get_sentences(self):
        '''Extracts all sentences from a string of text.

        :param posts: string from which sentences will be extracted.
        :returns: TextBlob sentences object
        '''

        return self.textblob_posts.sentences

    def get_words(self):
        '''Extracts all sentences from a string of text.

        :param posts: string from which sentences will be extracted.
        :returns: TextBlob sentences object
        '''

        return [
            word.singularize() for word in self.textblob_posts.words
            if ('v=' not in word) and (word not in INVALID_URLS)
        ]

    def separate_links_from_words(self):
        '''Extracts all unique url links from list of words.

        :param words: list of strings representing all words in text
        :returns: list of strings representing url links
        '''
        non_link_words = []
        links = []
        for word in self.words:
            if ('.com' in word) or ('www.' in word):
                links.append(word)
            else:
                non_link_words.append(word)
        self.links = links
        self.non_link_words = non_link_words

    def remove_links_from_text(self, text):
        for link in self.links:
            text.replace(link, '')
        return text

    def add_parts_of_speech_frequencies_to_stylometry_markers(self):
        pos_tags = [tag for word, tag in self.textblob_posts.pos_tags]

        keys = [
            'adjs_per_word', 'interjections_per_word', 'proper_nouns_per_word', 'nouns_per_word',
            'personal_pronouns_per_word', 'prepositions_per_word',
            'determiners_per_word', 'adverbs_per_word', 'verbs_per_word'
        ]
        pos_abbrs = [
            'JJ', 'UH', 'NNP', 'NN',
            'PRP', 'IN',
            'DT', 'RB', 'VB'
        ]

        for key, pos_abbr in zip(keys, pos_abbrs):
            self.stylometry_markers[key] = (pos_tags.count(pos_abbr) / self.num_of_non_link_words)

    def add_frequencies_of_punctuation_tokens_per_sentence(self):
        for punct in freq_punctuation_per_sentence:
            self.stylometry_markers[punct + '_per_sentence'] = get_freq_of_items_in_list(
                self.all_punctuation_in_text, [punct], self.num_of_sentences
            )

    def add_frequencies_of_punctuation_tokens_per_word(self):
        for punct in freq_punctuation_per_word:
            self.stylometry_markers[punct + '_per_word'] = get_freq_of_items_in_list(
                self.all_punctuation_in_text, [punct], self.num_of_words) * 1000

    def add_frequencies_of_tokens_to_stylometry_markers(self):
        for key, items in freq_token_types_per_word.items():
            self.stylometry_markers[key] = get_freq_of_items_in_list(
                self.words, items, self.num_of_words) * 1000

    def add_frequencies_of_link_types_to_stylometry_markers(self):
        num_of_links = len(self.links)
        self.stylometry_markers['links_per_post'] = (num_of_links / self.num_of_posts) * 100

        for search_term in LINK_TYPES:
            self.stylometry_markers[search_term + '_per_post'] = (
                get_freq_of_characters_in_text_in_list(self.links, [search_term], self.num_of_posts)
            ) * 100

        self.stylometry_markers['images_per_post'] = (
            get_freq_of_characters_in_text_in_list(self.links, IMAGE_TERMS, self.num_of_posts)
         ) * 100

    def add_avg_lens_to_stylometry_markers(self):
        self.stylometry_markers['avg_words_per_post'] = self.num_of_words / self.num_of_posts
        self.stylometry_markers['word_diversity'] = (len(set([word.lower() for word in self.non_link_words])) /
                                                     self.num_of_non_link_words)
        self.stylometry_markers['avg_word_length'] = np.mean([len(word) for word in self.non_link_words])

        words_per_sentences = [len(sentence.split()) for sentence in self.sentences]
        self.stylometry_markers['mean_sentence_len'] = np.mean(words_per_sentences)
        self.stylometry_markers['std_sentence_len'] = np.std(words_per_sentences)

    def add_sentiment_analysis_to_stylometry_markers(self):
        self.stylometry_markers['polarity'] = self.textblob_posts.sentiment.polarity
        self.stylometry_markers['subjectivity'] = self.textblob_posts.sentiment.subjectivity

    @staticmethod
    def is_emoji(sentence):
        '''Checks if input is an emoji.

        :returns type: Boolean.
        '''
        return (sentence[0] == ':') and (sentence[-2] == ':')

    def add_frequency_of_emojis_and_emoticons_to_stylometry_markers(self):
        # NOTE: .sentences removes ellipses that come at the end of sentences. fix this.
        self.stylometry_markers['ellipses_per_sentence'] = (
            get_count_of_characters_in_text(self.posts_removing_links, '...') / self.num_of_sentences
        )

        self.stylometry_markers['emojis_per_sentence'] = (
            sum([self.is_emoji(sentence) for sentence in self.sentences]) / self.num_of_sentences
        )

        self.stylometry_markers['smilies_per_sentence'] = (
            get_count_of_characters_in_text(self.posts_removing_links, ':)') +
            get_count_of_characters_in_text(self.posts_removing_links, ':D') +
            get_count_of_characters_in_text(self.posts_removing_links, ': )') +
            get_count_of_characters_in_text(self.posts_removing_links, ':-)')
        ) / self.num_of_sentences

        self.stylometry_markers['hashtags_per_sentence'] = (
            get_count_of_characters_in_text(self.posts_removing_links, '#')
        ) / self.num_of_sentences

        self.stylometry_markers['mentions_per_sentence'] = (
            get_count_of_characters_in_text(self.posts_removing_links, '@')
        ) / self.num_of_sentences


    def add_capitilization_info_to_stylometry_markers(self):
        self.stylometry_markers['all_caps_per_word'] = (sum([(word.upper() == word) and (word not in MBTI_TYPES_UPPER)
                                                             for word in self.non_link_words])
                                                        / self.num_of_non_link_words)

        self.stylometry_markers['sentence_capitalization_freq'] = sum([
            sentence[0].upper() == sentence[0] for sentence in self.sentences
        ]) / self.num_of_sentences
