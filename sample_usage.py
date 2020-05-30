from sklearn.datasets import make_blobs
import numpy as np
from SMD import SMD
import ray

def main():
	N = 1000
	#make 5 clusters in 10 dimensions
	X_s,y = make_blobs(N,n_features = 10,centers = 5)
	D_s = 10
	#Add 100 noisy dimensions, so D_s = 10, and D_n = 100. D/D_s = (D_N + D_s)/D_s = 11
	X = np.hstack([X_s,np.random.randn(N,100)])
	D_n = 100
	#Try SMD, pretend we don't know that there are 5 clusters, and guess 6
	ray.init()
	z = SMD(X,k_guess = 6,trials = 100)
	print(f'Good dimensions have an average Z-score: {z[:D_s].mean():.2f} +/- {z[:D_s].std():.2f}\n')
	print(f'Noisy dimensions have an average Z-score: {z[D_s:].mean():.2f} +/- {z[D_s:].std():.2f}')


if __name__ == '__main__':
	main()