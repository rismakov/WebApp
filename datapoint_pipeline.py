import pandas as pd
import pickle

from stylometry_analysis_v1 import StyleFeatures
from utils.utils import open_cpickle_file


def get_model():
	return open_cpickle_file('Fitted_Model_AdaBoostClassifier_Style')


def featurize_text(text):
	features = StyleFeatures(text)
	features = {feature:[v] for feature, v in features.items()}
	features = pd.DataFrame.from_dict(features)

	features['polarity'] = features['polarity'] + 1.0

	return features

def predict_gender_from_text(text):
	'''
	Filters text and return gender prefiction and features of writing style
	Input: string
	Output: tuple of string and dictionary
	'''
	features = featurize_text(text)

	prediction = get_model().predict(features)

	if prediction == 1:
		return 'male', features
	elif prediction == 0:
		return 'female', features


if __name__ == '__main__':
	with open("text_files/my_text.txt") as f:
		text = ' '.join([line.decode('utf-8').strip() for line in f.readlines()])

	print(extract_and_filter_datapoint_text(text))

