import json
import numpy as np
import argparse


def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--params',
                        '-p',
                        default="params_set_3x10.json",
                        required=False,
                        help='Parameters file (default: %(default)s)')
    parser.add_argument('--output',
                        '-o',
                        default="data_3x10.json",
                        required=False,
                        help='Generated data with specified parameters (default: %(default)s)')
    args = parser.parse_args()
    return args.params, args.output


param_file, output_file = ParseArguments()

with open(param_file, 'r') as inputfile:
    params = json.load(inputfile)

w = params['w']
k = params['k']
alpha = params['alpha']
Theta = np.asarray(params['Theta'])
ThetaB = np.asarray(params['ThetaB'])

X = []
theta_prim = Theta.transpose()

for i in range(k):
    z_tmp = np.random.choice([0, 1], 1, p=[1 - alpha, alpha])
    if z_tmp == 1:  # motif
        x_tmp = []
        for j in range(w):
            x_tmp.append(np.random.choice([1, 2, 3, 4], 1, p=theta_prim[j,])[0])
        X.append(x_tmp)
    elif z_tmp == 0:  # not_motif
        X.append(list(np.random.choice([1, 2, 3, 4], w, p=ThetaB)))
    else:
        pass

gen_data = {
    "alpha": alpha,
    "X": [[int(x) for x in sublist] for sublist in X]
}

with open(output_file, 'w') as outfile:
    json.dump(gen_data, outfile)
