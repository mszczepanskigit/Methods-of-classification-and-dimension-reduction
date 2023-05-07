import json

import numpy as np

import argparse

# Przyklad: tak zapisane zostaly ponizsze parametry:


# position weight matrix:
lst = []
for _ in range(30):
    a, b, c, d = np.random.random(4)
    s = a + b + c + d
    a1, b1, c1, d1 = a / s, b / s, c / s, d / s
    lst.append([a1, b1, c1, d1])

tmp = np.array(lst)
Theta = tmp.T

# background distribution
ThetaB = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])

params = {
    "w": 30,
    "alpha": 0.8,
    "k": 1000,
    "Theta": Theta.tolist(),
    "ThetaB": ThetaB.tolist()
}

# Uwaga: powzyej nie mozna podac macierzy -- tutaj sa zamieniane na listy
# (potem po wczytaniu, wystarczy zamienic na macierz np.asarray(.))


with open('params_set_30x1000.json', 'w') as outfile:
    json.dump(params, outfile)
