'''
Author: Jiasheng Li
data is available both in project directory /Data and online: http://azuremlsamples.azureml.net/templatedata/PM_train.txt
'''

import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM

# parameter settings

# Setting seed for reproducibility
np.random.seed(1234)
PYTHONHASHSEED = 0

# trained model will be saved in Output file following reproducible principle
model_path = '../Output/binary_classification_model.h5'

# windows for practical use - problem formulation: knowing whether this engine going to fail within w1 cycles
# w0: a backup parameter for further works
w1 = 30
w0 = 15

# read training, test data
train_dataframe = pd.read_csv('../Data/PM_train.txt', sep=" ", header=None)
train_dataframe.drop(train_dataframe.columns[[26, 27]], axis=1, inplace=True)
train_dataframe.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                    's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                    's15', 's16', 's17', 's18', 's19', 's20', 's21']

train_dataframe = train_dataframe.sort_values(['id', 'cycle'])

# read test data - aircraft engine operating data without remaining cycles
test_dataframe = pd.read_csv('../Data/PM_test.txt', sep=" ", header=None)
test_dataframe.drop(test_dataframe.columns[[26, 27]], axis=1, inplace=True)
test_dataframe.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                   's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                   's15', 's16', 's17', 's18', 's19', 's20', 's21']

# read ground truth data - remaining cycles for each engine id in test data set
truth_dataframe = pd.read_csv('../Data/PM_truth.txt', sep=" ", header=None)
truth_dataframe.drop(truth_dataframe.columns[[1]], axis=1, inplace=True)

# Data Preprocessing

# Data Labeling - generate column RUL(Remaining Usefull Life)
rul = pd.DataFrame(train_dataframe.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
train_dataframe = train_dataframe.merge(rul, on=['id'], how='left')
train_dataframe['RUL'] = train_dataframe['max'] - train_dataframe['cycle']
train_dataframe.drop('max', axis=1, inplace=True)

# generate label for training data
# we will only make use of "label1" for binary classification
train_dataframe['label1'] = np.where(train_dataframe['RUL'] <= w1, 1, 0)
train_dataframe['label2'] = train_dataframe['label1']
train_dataframe.loc[train_dataframe['RUL'] <= w0, 'label2'] = 2

# MinMax normalization interval : (0,1)
train_dataframe['cycle_norm'] = train_dataframe['cycle']
cols_normalize = train_dataframe.columns.difference(['id', 'cycle', 'RUL', 'label1', 'label2'])
min_max_scaler = preprocessing.MinMaxScaler()
norm_train_dataframe = pd.DataFrame(min_max_scaler.fit_transform(train_dataframe[cols_normalize]),
                             columns=cols_normalize,
                             index=train_dataframe.index)
join_df = train_dataframe[train_dataframe.columns.difference(cols_normalize)].join(norm_train_dataframe)
train_dataframe = join_df.reindex(columns=train_dataframe.columns)

# preprocessing for test data
test_dataframe['cycle_norm'] = test_dataframe['cycle']
norm_test_dataframe = pd.DataFrame(min_max_scaler.transform(test_dataframe[cols_normalize]),
                            columns=cols_normalize,
                            index=test_dataframe.index)
test_join_df = test_dataframe[test_dataframe.columns.difference(cols_normalize)].join(norm_test_dataframe)
test_dataframe = test_join_df.reindex(columns=test_dataframe.columns)
test_dataframe = test_dataframe.reset_index(drop=True)

rul = pd.DataFrame(test_dataframe.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
truth_dataframe.columns = ['more']
truth_dataframe['id'] = truth_dataframe.index + 1
truth_dataframe['max'] = rul['max'] + truth_dataframe['more']
truth_dataframe.drop('more', axis=1, inplace=True)

test_dataframe = test_dataframe.merge(truth_dataframe, on=['id'], how='left')
test_dataframe['RUL'] = test_dataframe['max'] - test_dataframe['cycle']
test_dataframe.drop('max', axis=1, inplace=True)

test_dataframe['label1'] = np.where(test_dataframe['RUL'] <= w1, 1, 0)
test_dataframe['label2'] = test_dataframe['label1']
test_dataframe.loc[test_dataframe['RUL'] <= w0, 'label2'] = 2

# LSTM

# window size for LSTM
sequence_length = 50

# function to reshape features into keras LSTM layer (samples, time steps, features)
def gen_sequence(id_df, seq_length, seq_cols):
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

sensor_cols = ['s' + str(i) for i in range(1, 22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(sensor_cols)


seq_gen = (list(gen_sequence(train_dataframe[train_dataframe['id'] == id], sequence_length, sequence_cols))
           for id in train_dataframe['id'].unique())

seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
print('The shape of generated sequence{}'.format(seq_array.shape))

# function to generate labels
def gen_labels(id_df, seq_length, label):
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length:num_elements, :]

# generate labels
label_gen = [gen_labels(train_dataframe[train_dataframe['id'] == id], sequence_length, ['label1'])
             for id in train_dataframe['id'].unique()]
label_array = np.concatenate(label_gen).astype(np.float32)
label_array.shape

# Network configurations
nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]

model = Sequential()

model.add(LSTM(
    input_shape=(sequence_length, nb_features),
    units=100,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    units=50,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=nb_out, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# fitting
history = model.fit(seq_array, label_array, epochs=100, batch_size=200, validation_split=0.05, verbose=2,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0,
                                                             mode='min'),
                               keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True,
                                                               mode='min', verbose=0)]
                    )

#print(history.history.keys())

# Accuracy plot
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig("../Output/model_accuracy.png")

# loss plot
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig("../Output/model_loss.png")

# Training Evaluation
scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
print('Accurracy: {}'.format(scores[1]))

# make predictions and compute confusion matrix
y_pred = model.predict_classes(seq_array, verbose=1, batch_size=200)
y_true = label_array

test_set = pd.DataFrame(y_pred)
test_set.to_csv('../Output/output_train.csv', index=None)

print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
confusionMatr = confusion_matrix(y_true, y_pred)
print(confusionMatr)

# compute precision and recall
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
print('precision = ', precision, '\n', 'recall = ', recall)

# Test Evaluation

seq_array_test_last = [test_dataframe[test_dataframe['id'] == id][sequence_cols].values[-sequence_length:]
                       for id in test_dataframe['id'].unique() if len(test_dataframe[test_dataframe['id'] == id]) >= sequence_length]

seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
print("seq_array_test_last")
#print(seq_array_test_last)
print(seq_array_test_last.shape)

y_mask = [len(test_dataframe[test_dataframe['id'] == id]) >= sequence_length for id in test_dataframe['id'].unique()]
#print("y_mask")
#print(y_mask)
label_array_test_last = test_dataframe.groupby('id')['label1'].nth(-1)[y_mask].values
label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(np.float32)
#print("label_array_test_last")
print(label_array_test_last.shape)
#print(label_array_test_last)

# reload model
if os.path.isfile(model_path):
    estimator = load_model(model_path)

# test metrics
scores_test = estimator.evaluate(seq_array_test_last, label_array_test_last, verbose=2)
print('Test accurracy: {}'.format(scores_test[1]))

# make predictions and compute confusion matrix
y_pred_test = estimator.predict_classes(seq_array_test_last)
y_true_test = label_array_test_last

test_set = pd.DataFrame(y_pred_test)
test_set.to_csv('../Output/output_test.csv', index=None)

print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
confusionMatr = confusion_matrix(y_true_test, y_pred_test)
print(confusionMatr)

# compute precision and recall
precision_test = precision_score(y_true_test, y_pred_test)
recall_test = recall_score(y_true_test, y_pred_test)
f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
print('Precision: ', precision_test, '\n', 'Recall: ', recall_test, '\n', 'F1-score:', f1_test)

# Prediction visualization
fig_verify = plt.figure(figsize=(100, 50))
plt.plot(y_pred_test, color="blue")
plt.plot(y_true_test, color="green")
plt.title('prediction')
plt.ylabel('value')
plt.xlabel('row')
plt.legend(['predicted', 'actual data'], loc='upper left')
plt.show()
fig_verify.savefig("../Output/test_prediction_visualization.png")