import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import svm

# read data and apply one-hot encoding
# data = pd.read_csv("data/marbling_dataset_v2/marbling_features_shape_dataset.csv", header=None)
data = pd.read_csv("data/marbling_dataset_v2/marbling_features_lbp_dataset.csv", header=None)
len_features = len(data.columns)-1

X = data.iloc[:, 1:len_features]
y = data.iloc[:, len_features]

# 012; 34; 567
# 0123; 4567
y.replace(to_replace=1, value=0, inplace=True)
y.replace(to_replace=2, value=0, inplace=True)
y.replace(to_replace=3, value=0, inplace=True)

y.replace(to_replace=4, value=1, inplace=True)
y.replace(to_replace=5, value=1, inplace=True)
y.replace(to_replace=6, value=1, inplace=True)
y.replace(to_replace=7, value=1, inplace=True)


# n_X = (X-X.mean())/X.std()  # Rescale to mean = 0; sd ~ 1
# X = n_X
# # n_X.to_csv("data/marbling_dataset/marbling_dataset_n_X.csv")

# mmn_X = (X-X.min())/(X.max()-X.min())  # Normalized (rescale  0 to 1)
# X = mmn_X
# # mmn_X.to_csv("data/marbling_dataset/marbling_dataset_mmn_X.csv")

# scaler = MinMaxScaler(feature_range=(-1, 1))  # Rescale -1 to 1
# mmn_sk_X = pd.DataFrame(scaler.fit_transform(X))
# X = mmn_sk_X
# mmn_sk_X.to_csv("data/marbling_dataset/marbling_dataset_mmn_sk_X.csv")

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=1)

# From https://www.baeldung.com/cs/svm-multiclass-classification
svm_class = svm.SVC(kernel='rbf', gamma=5.0, C=10).fit(X_train, y_train)
# svm_class = svm.SVC(kernel='poly', degree=10, C=100).fit(X_train, y_train)
y_pred = svm_class.predict(X_test)

# rbf_accuracy = accuracy_score(y_test, rbf_pred)
# rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
# print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
# print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(cm)
print(acc)
print(prec)
print(rec)
print(f1)


# # Restore best model
# model.load_state_dict(best_weights)
# model.eval()
# y_pred = model(X_test)

# y_pred_np = torch.argmax(y_pred, 1).detach().numpy()
# y_test_np = torch.argmax(y_test, 1).detach().numpy()

# print(y_pred_np)
# print(y_test_np)

# cm = confusion_matrix(y_test_np, y_pred_np)
# acc = accuracy_score(y_test_np, y_pred_np)
# prec = precision_score(y_test_np, y_pred_np, average='macro')
# rec = recall_score(y_test_np, y_pred_np, average='macro')

# print(cm)
# print(acc)
# print(prec)
# print(rec)

# # Plot the loss and accuracy
# plt.plot(train_loss_hist, label="train")
# plt.plot(test_loss_hist, label="test")
# plt.xlabel("epochs")
# plt.ylabel("cross entropy")
# plt.legend()
# plt.show()

# plt.plot(train_acc_hist, label="train")
# plt.plot(test_acc_hist, label="test")
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
# plt.legend()
# plt.show()