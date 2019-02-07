import pandas as pd
import pickle

from data_transformation.stylometry_analysis import StyleFeatures
from utils.utils import convert_sparse_mat_to_df, open_cpickle_file, open_model

TFIDF_MAX_FEATURES = 7000


def get_tfidf_vec():
	filename = 'mbti_tfidf_{}'.format(TFIDF_MAX_FEATURES)
	return open_model(filename)


def get_model():
	return open_model('Fitted_LR_MBTI_Model_{}'.format(TFIDF_MAX_FEATURES))


def predict_text(text):
	'''
	Filters text and return gender prefiction and features of writing style
	Input: string
	Output: tuple of string and dictionary
	'''
	tfidf_vec = get_tfidf_vec()
	tfidf = tfidf_vec.transform([text])
	feature_names = tfidf_vec.get_feature_names()
	tfidf_df = convert_sparse_mat_to_df(tfidf, feature_names)

	features = StyleFeatures(text).get_all_featurized_features()
	features = {feature: [v] for feature, v in features.items()}
	features_df = pd.DataFrame.from_dict(features)
	features_df['polarity'] = features_df['polarity'] + 1.0

	concat_df = pd.concat([features_df, tfidf_df], axis=1)
	prediction = get_model().predict(concat_df)

	return prediction[0]


if __name__ == '__main__':
	with open("text_files/my_text.txt") as f:
		text = ' '.join([line.decode('utf-8').strip() for line in f.readlines()])

	print(predict_text(text))

