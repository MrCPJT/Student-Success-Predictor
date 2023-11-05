import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier

#################################

df = pd.read_csv('dataset.csv')

df = df.rename(columns={'Nacionality': 'Nationality'})

df.columns = df.columns.str.lower().str.replace(' ', '_')

df['target'], uniques = pd.factorize(df['target'])

#################################

train_full, test = train_test_split(df, test_size=0.2, random_state=42)

y_train_full = train_full['target']
y_test = test['target']

del train_full['target']
del test['target']

def train(df_train, y_train):
    dicts = df_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = RandomForestClassifier(max_depth = None, 
                                   max_leaf_nodes = None,
                                   min_samples_leaf = 1,
                                   n_estimators = 300,
                                   random_state = 42)
    model.fit(X_train, y_train)
    
    return dv, model

def predict(df, dv, model):
    dicts = df.to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict(X)

    return y_pred

#################################

dv, model = train(train_full, y_train_full)
y_pred = predict(test, dv, model)

lb = LabelBinarizer()

print(f'AUC: {roc_auc_score(lb.fit_transform(y_test), lb.transform(y_pred)):.2f}')

#################################

with open('model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'\nThe model is saved to: "model.bin"')