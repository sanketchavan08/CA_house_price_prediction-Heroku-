# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 09:16:14 2020

@author: SHANKY
"""
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle

RANDOM_STATE = 42

dataset = fetch_california_housing()
dataset_df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
X_data, y_data = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=RANDOM_STATE)
print('Train data shape:', X_data.shape)
print('Test data shape:', X_test.shape)


model=GradientBoostingRegressor(max_depth= 8, max_features=6, min_samples_split=200, n_estimators=100,random_state=42)
model.fit(X_data, y_data)
pickle.dump(model, open('model.pkl','wb'))

#model.predict([[100000, 5,  4, 1, 35000, 10, 37.54, -121.72]])


