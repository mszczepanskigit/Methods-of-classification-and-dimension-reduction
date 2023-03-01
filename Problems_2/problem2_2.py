from sklearn import datasets, decomposition
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

mnist = datasets.load_digits()  # original data

X = mnist.data
Y = mnist.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

classifier = KNeighborsClassifier(n_neighbors=13)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

# print(Y_pred)
# print(Y_test)

result_of_classification = sum([1 for (x, y) in zip(Y_pred, Y_test) if x == y])/len(Y_test)

print(result_of_classification)

n = 2