import json

import numpy as np


tmp = np.array([[3 / 8, 1 / 8, 2 / 8, 2 / 8],
                [1 / 10, 2 / 10, 3 / 10, 4 / 10],
                [1 / 7, 2 / 7, 1 / 7, 3 / 7]])
Theta = tmp.T

ThetaB = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])

params = {
    "w": 3,
    "alpha": 0.5,
    "k": 100,
    "Theta": Theta.tolist(),
    "ThetaB": ThetaB.tolist()
}
with open('params_set1.json', 'w') as outfile:
    json.dump(params, outfile)