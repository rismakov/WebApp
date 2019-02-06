import pandas as pd
import pickle

from constants import TFIDF_MAX_FEATURES
from data_transformation.stylometry_analysis import StyleFeatures

VECTORIZER_FILENAME =
TFIDF_VEC = open_pickle_file(VECTORIZER_FILENAME)


def open_pickle_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def predict_text(text):
	'''
	Filters text and return gender prefiction and features of writing style
	Input: string
	Output: tuple of string and dictionary
	'''
	tfidf = TFIDF_VEC.transform(text)

	features = StyleFeatures(text)
	# features = {feature: [v] for feature, v in features.items()}

	df = pd.DataFrame.from_dict(features)
	df['polarity'] = df['polarity'] + 1.0

	clf = open_pickle_file('Fitted_Model_AdaBoostClassifier_Style')

	concat_df = pd.concat([df, tfidf], axis=1)
	prediction = clf.predict(features)

	return prediction


if __name__ == '__main__':
	with open("text_files/my_text.txt") as f:
		text = ' '.join([line.decode('utf-8').strip() for line in f.readlines()])

	print predict_text(text)

