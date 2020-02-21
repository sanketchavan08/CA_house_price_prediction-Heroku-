# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:27:36 2020

@author: SHANKY
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

flask_app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@flask_app.route('/')
def home():
    return render_template('index.html')

@flask_app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2) * 100000

    return render_template('index.html', 
                           prediction_text='Predicted House price is $ {}'.format(output))


if __name__ == "__main__":
    flask_app.run(debug=True)
    