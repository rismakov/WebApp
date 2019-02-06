from __future__ import division

import pandas as pd
import pickle
import scipy
import zipfile


def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def convert_sparse_mat_to_df(matrix, columns):
    return pd.DataFrame(matrix.toarray(), columns=columns)


def convert_df_to_sparse_mat(df):
    return scipy.sparse.csr_matrix(df.values)


def get_distribution_of_ylabel_classes(data, label_col):
    '''Returns the distribution of all unique ylabel classes in the data.

    :param data type: Pandas DataFrame.
    :returns type: Pandas DataFrame.
    '''
    grouped_data = data.groupby(label_col).count()

    col = grouped_data.columns[0]
    counts = grouped_data[col]

    total = sum(counts)
    return counts.apply(lambda x: round(x / total * 100, 2))


def get_freq_of_characters_in_text_in_list(lst, search_items, normalizer):
    '''Calculates the combined frequency given strings of characters within a list appear in a list of text.

    Example:
        lst = ['apple.gif', 'apple', '.gif', 'png']
        search_items = ['gif', 'apple']
        normalizer = 4

        returns: Float(3/4)

    :param lst: List of items to search through.
    :param search_items: List of items you want the combined count of.
    :param normalizer: Integer by which to normalize count.

    :returns type: Float.
    '''

    cnts = (any(search_item in lst_item for search_item in search_items) for lst_item in lst)

    return sum(cnts) / normalizer


def get_freq_of_items_in_list(lst, search_items, normalizer):
    '''Calculates the combined frequency of each item within a specified list divided by the normalizer specified.

    Example:
        lst = ['apple.gif', 'apple', '.gif']
        search_items = ['.gif', 'apple']
        normalizer = 3

        returns: Float(2/3)

    :param lst: List of items to search through.
    :param search_items: List of items you want the combined count of.
    :param normalizer: Integer by which to normalize count.

    :returns type: Float.
    '''
    cnts = [lst.count(item) for item in search_items]

    return sum(cnts) / normalizer


def get_count_of_characters_in_text(text, characters):
    '''Gets the count of a specific word or string of characters within a block of text.

    :param text: String of text which to search through.
    :param characters: String of characters which to count.

    :returns type: Int.
    '''
    return len(text.split(characters)) - 1
