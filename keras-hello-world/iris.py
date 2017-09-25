import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

iris = sns.load_dataset("iris")
iris.head()

X = iris.values[:, :4]
y = iris.values[:, 4]

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.5, random_state=0)

lr = LogisticRegressionCV()
lr.fit(train_X, train_y)

print("Accuracy = {:.2f}".format(lr.score(test_X, test_y)))