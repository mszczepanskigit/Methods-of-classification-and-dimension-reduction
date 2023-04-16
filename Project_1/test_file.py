from testing import fill_missing
import numpy as np

with open("ratings.csv", "r") as ratings:
    ratings = np.genfromtxt(ratings, delimiter=',')[1:]
A = np.array([[4.2, 0.9, 2.7272277], [0.111, 0., 0.], [5.5, 0.5, 0.]])
#B = fill_missing(A, method=1, column=1)
nonzero_indices = np.nonzero(A)
nonempty_columns = np.unique(nonzero_indices[1])
empty_columns = list(set(range(A.shape[1])) - set(nonempty_columns))
A1 = np.delete(A, empty_columns, axis=1)
print(np.round(2*A)/2)
print(A.transpose((1, 0)))
print(A1)