import json
import numpy as np
import pandas as pd

from ast import literal_eval
from flask import Flask, jsonify, render_template, request

from datapoint_pipeline import predict_gender_from_text
from mbti_pipeline import predict_text
from info import ABOUT_MBTI

app = Flask(__name__)

MEANS = {
    'article_len': (1084,'longer','shorter'),
    'mean_sentence_len': (26.44, 'longer','shorter'),
    'mean_word_len': (4.834, 'longer', 'shorter'),
    'type_token_ratio': (0.493, 'greater','weaker'),
    'freq_commas': (57.7, 'higher','lower'),
    'freq_quotation_marks': (0.15,'more','less'),
    'freq_semi_colons': (1.02, 'greater','lower'),
    'polarity': (0.084, 'more', 'less'),
    'subjectivity': (0.42, 'more', 'less'),
    'std_sentence_len': (16.48, 'more', 'less')
}

ORDERED_FEATURES = [
    'article_len', 'mean_sentence_len', 'mean_word_len', 'type_token_ratio', 'freq_commas',
    'freq_quotation_marks', 'freq_semi_colons', 'polarity', 'subjectivity', 'std_sentence_len'
]

LOWER_IND = 2
HIGHER_IND = 1

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/lexophilia')
def lexophilia():
    return render_template('lexophilia.html')


@app.route('/mbti')
def mbti():
    return render_template('mbti.html')


@app.route('/app')
def webapp():
    return render_template('web_app.html')


@app.route('/mbti_app')
def mbti_webapp():
    return render_template('mbti_web_app.html')


def get_predictive_info(user_text):
    '''
    Gets prediction and features of writing style from inputted text.
    Input: string
    Output: tuple of string (gender prediction) and list
    (property of the text, whether is more or less than mean)
    '''
    prediction, feature_df = predict_gender_from_text(user_text)

    properties = []
    for feature in ORDERED_FEATURES:
        feature_mean = MEANS[feature][0]
        text_value = feature_df[feature][0]

        if text_value > feature_mean:
            properties.append(MEANS[feature][HIGHER_IND])
        elif text_value <= feature_mean:
            properties.append(MEANS[feature][LOWER_IND])

    return prediction, properties

def is_json(text):
    '''This is hack-y way. Fix this so that different buttons in html code automatically check this.
    '''
    return (text[0] == '{') and (text[-1] == '}')


@app.route('/prediction', methods =['POST'])
def predict_gender():
    user_text = request.form['user_input']
    prediction, properties = get_predictive_info(user_text)

    detailed_analysis = ("Your style of writing suggests you are a {}. Your text tends to be {} than the average politi"
                         "cal article, with {} sentence lengths and {} word lengths. You tend to use {} diversity of vo"
                         "cabulary. Your writing tends to have a {} frequency of commas. Additionally, your text contai"
                         "ns {} use of quotations and dialouge, and {} level of formality. Your writing tend to be {} p"
                         "olar, {} subjective, and posseses {} varied sentence lengths.").format(
                            prediction, properties[0], properties[1], properties[2], properties[3], properties[4],
                            properties[5], properties[6], properties[7],
                            properties[8], properties[9]
                        )

    return render_template('prediction.html', detailed_analysis=detailed_analysis, prediction=prediction)


@app.route('/mbti_prediction', methods =['POST'])
def predict_mbti():
    user_text = request.form['user_input']

    if is_json(user_text):
        user_text = literal_eval(user_text)

        '''
        formatted_user_text = []
        for comment_info in user_text['comments']:
            for single_comments in comment_info['data']:
                post = single_comments['comment'].get('comment')
                if post:
                    formatted_user_text.append(post)
        '''

        user_text = [single_comments['comment']['comment'] for comment_info in user_text['comments']
                     for single_comments in comment_info['data'] if single_comments['comment'].get('comment')]

        user_text = '|||'.join(user_text)

    prediction = predict_text(user_text)
    about_type = ABOUT_MBTI[prediction]

    return render_template('mbti_prediction.html', about_type=about_type, prediction=prediction)


if __name__ == '__main__':
    # Start Flask app
    app.run(host='0.0.0.0', port=8080, debug=True)

