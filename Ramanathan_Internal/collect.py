
import numpy as np
import os
import sys

def main():
	counts = []
	shuffcounts = []
	for ki in range(20):
		if ki<10:
			temp = np.load(f'counts_{ki}.npy')
			os.remove(f'counts_{ki}.npy')
			counts.append(temp)
		else:
			temp = np.load(f'shuffcounts_{ki}.npy')
			os.remove(f'shuffcounts_{ki}.npy')
			shuffcounts.append(temp)
	counts = [i for k in counts for i in k]
	shuffcounts=[i for k in shuffcounts for i in k]
	j = max(max(counts),max(shuffcounts))
	gd,_ = np.histogram(counts, np.arange(j+1), density = True)
	g_shuffled,_ = np.histogram(shuffcounts, np.arange(j+1), density = True)
	Z_d = (gd-g_shuffled.mean())/g_shuffled.std()
	np.save('Z_scores.npy',Z_d)

if __name__ == '__main__':
	main()