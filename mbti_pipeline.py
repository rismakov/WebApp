import pandas as pd
import pickle

from data_transformation.stylometry_analysis import StyleFeatures
from utils.utils import convert_sparse_mat_to_df, open_cpickle_file, open_model

TFIDF_MAX_FEATURES = 7000


VECTORIZER_FILENAME = 'mbti_tfidf_{}'.format(TFIDF_MAX_FEATURES)
TFIDF_VEC = open_cpickle_file(VECTORIZER_FILENAME)

CLF = open_model('Fitted_LR_MBTI_Model_{}'.format(TFIDF_MAX_FEATURES))

def predict_text(text):
	'''
	Filters text and return gender prefiction and features of writing style
	Input: string
	Output: tuple of string and dictionary
	'''
	tfidf = TFIDF_VEC.transform([text])
	feature_names = TFIDF_VEC.get_feature_names()
	tfidf_df = convert_sparse_mat_to_df(tfidf, feature_names)

	features = StyleFeatures(text).get_all_featurized_features()
	features = {feature: [v] for feature, v in features.items()}
	features_df = pd.DataFrame.from_dict(features)
	features_df['polarity'] = features_df['polarity'] + 1.0

	concat_df = pd.concat([features_df, tfidf_df], axis=1)
	prediction = CLF.predict(concat_df)

	return prediction[0]


if __name__ == '__main__':
	with open("text_files/my_text.txt") as f:
		text = ' '.join([line.decode('utf-8').strip() for line in f.readlines()])

	print(predict_text(text))

