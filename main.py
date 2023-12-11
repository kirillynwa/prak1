from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from math import sqrt

def myhash(x):
    return 1/len(x)
df = pd.read_csv("source.csv")
cols = df.columns.tolist()
cols = cols[-1:] + cols[1:-1] + cols[:1]
df = df[cols]

r = {'5-6 утра': 0.55, '6-7 утра': 0.65, '7-8 утра': 0.75, 'Позднее 8 утра': 0.85}
s = {'Менее 3 часов': 0.3, 'От 3 до 6 часов': 0.5, '7-8 часов': 0.75, 'Более 8 часов': 0.85}
df['Пол'] = df['Пол'].apply(lambda x: 1 if x == 'М' else 0)
df['Занимаетесь спортом?'] = df['Занимаетесь спортом?'].apply(lambda x: 1 if x == 'Да' else 0)
df['Есть ли работа?'] = df['Есть ли работа?'].apply(lambda x: 1 if x == 'Да' else 0)
df['Есть ли сердечные заболевания?'] = df['Есть ли сердечные заболевания?'].apply(lambda x: 1 if x == 'Да' else 0)
df['Сова или Жаворонок'] = df['Сова или Жаворонок'].apply(lambda x: 1 if x == 'Сова' else 0)
df['Есть ли молоко в холодильнике?'] = df['Есть ли молоко в холодильнике?'].apply(lambda x: 1 if x == 'Да' else 0)
df['Время подъема'] = df['Время подъема'].apply(lambda x: r[x])
df['Административный округ'] = df['Административный округ'].apply(lambda x: myhash(x))
df['Время сна'] = df['Время сна'].apply(lambda x: s[x])

train_size = int(df .shape[0]*0.8)
train_df = df .iloc[:train_size, :]
test_df = df .iloc[train_size:, :]
train = train_df.values
test = test_df.values
y_true = test[:, -1]
print('Train_Shape: ', train_df.shape)
print('Test_Shape: ', test_df.shape)

def euclidean_distance(x_test, x_train):
    distance = 0
    for i in range(len(x_test)-1):
        distance += (x_test[i]-x_train[i])**2
    return sqrt(distance)

def get_neighbors(x_test, x_train, num_neighbors):
    distances = []
    data = []
    for i in x_train:
        distances.append(euclidean_distance(x_test,i))
        data.append(i)
    distances = np.array(distances)
    data = np.array(data)
    sort_indexes = distances.argsort()
    data = data[sort_indexes]
    return data[:num_neighbors]

def prediction(x_test, x_train, num_neighbors):
    classes = []
    neighbors = get_neighbors(x_test, x_train, num_neighbors)
    for i in neighbors:
        classes.append(i[-1])
    predicted = max(classes, key=classes.count)
    return predicted


def accuracy(y_true, y_pred):
    num_correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            num_correct += 1
    accuracy = num_correct/len(y_true)
    return accuracy

y_pred = []
for i in test:
    y_pred.append(prediction(i, train, 5))
print(y_pred)
accuracy = accuracy(y_true, y_pred)
print(accuracy)
