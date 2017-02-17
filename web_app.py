import json
import pandas as pd
from flask import Flask, jsonify, render_template, request
import cPickle as pickle
import numpy as np
import pandas as pd
from datapoint_pipeline import open_cPickle_file, extract_and_filter_datapoint_text
from collections import OrderedDict

app = Flask(__name__)
model = open_cPickle_file('Fitted_Model_AdaBoostClassifier_Style') # opens trained Add Boost model

MEANS = OrderedDict()
MEANS['article_len'] = (1084,'longer','shorter')
MEANS['mean_sentence_len'] = (26.44, 'longer','shorter')
MEANS['mean_word_len'] = (4.834, 'longer', 'shorter')
MEANS['type_token_ratio'] = (0.493, 'greater','weaker')
MEANS['freq_commas'] = (57.7, 'higher','lower')
MEANS['freq_quotation_marks'] = (0.15,'more','less')
MEANS['freq_semi_colons'] = (1.02, 'greater','lower')
MEANS['polarity'] = (0.084, 'more', 'less')
MEANS['subjectivity'] = (0.42, 'more', 'less')
MEANS['std_sentence_len'] = (16.48, 'more', 'less')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/app')
def webapp():
    return render_template('web_app.html')

def get_predictive_info(user_text):
    '''
    Gets prediction and features of writing style from inputted text.
    Input: string
    Output: tuple of string (gender prediction) and list 
    (property of the text, whether is more or less than mean) 
    '''
    prediction,features = extract_and_filter_datapoint_text(user_text)

    properties = []
    for feature,params in MEANS.iteritems():
        if features[feature][0] > params[0]:
            properties.append(params[1])
        elif features[feature][0] <= params[0]:
            properties.append(params[2])

    return prediction,properties

@app.route('/prediction', methods =['POST'])
def predict_gender():

    user_text = request.form['user_input']
    prediction,properties = get_predictive_info(user_text)

    detailed_analysis = "Your style of writing suggests you are a {}. \
        Your text tends to be {} than the average political article, with {} sentence \
        lengths and {} word lengths. You tend to use {} diversity of vocabulary.\
        Your writing tends to have a {} frequency of commas. \
        Additionally, your text contains {} use of quotations and dialouge, and {} level of \
        formality. Your writing tend to be {} polar, {} subjective, and posseses {} varied sentence\
        lengths.".format(prediction, properties[0], properties[1], properties[2], properties[3],\
                        properties[4], properties[5], properties[6], properties[7], \
                        properties[8], properties[9])

    return render_template('prediction.html', detailed_analysis=detailed_analysis, prediction=prediction)

if __name__ == '__main__':
    # Start Flask app
    app.run(host='0.0.0.0', port=8080, debug=True)

