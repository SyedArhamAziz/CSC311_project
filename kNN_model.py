import sys
import csv
import random
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

def predict(x):
    return None

def predict_all(x):
    return None

def dist_single(v, x):
    diff = x - v
    sqdiff = diff**2
    sumval = sum(sqdiff)
    return sumval

def dist_all(v, X):
    diff = X - v
    sqdiff = diff**2
    sumval = np.sum(sqdiff, axis=1)
    return sumval

def predict_knn(v, X_train, t_train, k=1):
    dists = dist_all(v, X_train)
    indices = np.argsort(dists)[:k]
    ts = t_train[np.array(indices)]
    values, frequencies =  np.unique(ts, return_counts=True)
    prediction = values[frequencies.argmax()]
    return prediction

def compute_accuracy(X_new, t_new, X_train, t_train, k=1):
    num_predictions = 0
    num_correct = 0
    for i in range(X_new.shape[0]): 
        v = X_new[i] 
        t = t_new[i] 
        y = predict_knn(v, X_train, t_train, k) 
        num_correct += y==t 
        num_predictions += 1
    return num_correct / num_predictions

df = pd.read_csv("ml_challenge_dataset_fixed.csv")
df = df.dropna(subset=['This art piece makes me feel sombre.'])
df = df.drop(columns=['unique_id', 'Describe how this painting makes you feel.', 'If you could view this art in person, who would you want to view it with?', 'What season does this art piece remind you of?', 'If this painting was a food, what would be?', 'Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.'])

df['This art piece makes me feel sombre.'] = df['This art piece makes me feel sombre.'].apply(lambda x: float(x[0]) if type(x)==type('a') else 0.0)
df['This art piece makes me feel content.'] = df['This art piece makes me feel content.'].apply(lambda x: float(x[0]) if type(x)==type('a') else 0.0)
df['This art piece makes me feel calm.'] = df['This art piece makes me feel calm.'].apply(lambda x: float(x[0]) if type(x)==type('a') else 0.0)
df['This art piece makes me feel uneasy.'] = df['This art piece makes me feel uneasy.'].apply(lambda x: float(x[0]) if type(x)==type('a') else 0.0)

#normalize values
columns = df.columns.tolist()
columns.remove('If you could purchase this painting, which room would you put that painting in?')
for column in columns[1:]:
    mean = np.mean(df[column])
    std = np.std(df[column])
    df[column] = df[column].apply(lambda x: (x - mean) / std)

#print(df['How much (in Canadian dollars) would you be willing to pay for this painting?'][:30])

#add columns for rooms in the form of one-hot encoding
bathroom = []
dining = []
bed = []
living = []
office = []

for room_list in df['If you could purchase this painting, which room would you put that painting in?']:
    if room_list is np.nan:
        room_list = ''
    room_list = room_list.split(',')
    if 'Bathroom' in room_list:
        bathroom.append(1)
    else:
        bathroom.append(0)
    if 'Dining room' in room_list:
        dining.append(1)
    else:
        dining.append(0)
    if 'Office' in room_list:
        office.append(1)
    else:
        office.append(0)
    if 'Living room' in room_list:
        living.append(1)
    else:
        living.append(0)
    if 'Bedroom' in room_list:
        bed.append(1)
    else:
        bed.append(0)

df['Dining'] = dining
df['Bedroom'] = bed
df['Office'] = office
df['Living Room'] = living
df['Bathroom'] = bathroom

df = df.drop(columns=['If you could purchase this painting, which room would you put that painting in?'])

df_train = df.sample(frac=0.8, random_state=5)

df_valid = df.drop(df_train.index)
df_valid = df_valid.sample(frac=1, random_state=5)

df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)

t_train = df_train['Painting'].tolist()
t_train = np.stack(t_train)

df_temp = df_train.drop(columns=['Painting'])
X_train = df_temp.values.tolist()
X_train = np.stack(X_train)

t_valid = df_valid['Painting'].tolist()
t_valid = np.stack(t_train)

df_temp = df_valid.drop(columns=['Painting'])
X_valid = df_temp.values.tolist()
X_valid = np.stack(X_valid)

accuracies = []
for i in range(1,30):
    accuracies.append(compute_accuracy(X_valid, t_valid, X_train, t_train, k=i))

plt.plot(range(1,30), accuracies)
plt.show()
