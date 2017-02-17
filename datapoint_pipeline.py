import pandas as pd
from stylometry_analysis import StyleFeatures
import cPickle

def open_cPickle_file(filename):
    with open(filename, 'rb') as f:
        return cPickle.load(f)

def extract_and_filter_datapoint_text(text):
	'''
	Filters text and return gender prefiction and features of writing style
	Input: string
	Output: tuple of string and dictionary  
	'''
	features = StyleFeatures(text)
	features = {feature:[v] for feature, v in features.iteritems()}
	features = pd.DataFrame.from_dict(features)

	features['polarity'] = features['polarity'] + 0.5

	clf = open_cPickle_file('Fitted_Model_AdaBoostClassifier_Style')

	prediction = clf.predict(features)

	if prediction == 1:
		return 'male', features
	elif prediction == 0:
		return 'female', features


if __name__ == '__main__':
	with open("text_files/my_text.txt") as f:
		text = ' '.join([line.decode('utf-8').strip() for line in f.readlines()])

	print extract_and_filter_datapoint_text(text)

