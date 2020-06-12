
import numpy as np
from SMD_slurm import SMD, shuffle_data
from scipy.sparse import load_npz
import time
import os
import sys

def main():
	ki = int(sys.argv[1])
	k_guess = 33
	k_prior_min = 20
	n_sub = 4000
	trials = 150
	E = load_npz('TF_expression_log_norm.npz')
	X = np.array(E.todense())
	if ki<10:
		counts = SMD(X,k_guess,trials = trials, n_sub = n_sub, cluster_prior_min = k_prior_min)
		np.save(f'counts_{ki}.npy',counts)
	else:
		Xs = shuffle_data(X)
		counts = SMD(Xs,k_guess,trials = trials, n_sub = n_sub, cluster_prior_min = k_prior_min)
		np.save(f'shuffcounts_{ki}.npy',counts)


if __name__ == '__main__':
	main()