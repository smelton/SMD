

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
### Read Me:

#requires Ray, numpy, and sklearn

#The function SMD has the following inputs:

#X := Data matrix with dimensions (number of data points) x (number of features).

#k_guess := best guess of how many clusters are present. can be approximate. Better to guess too high than too low.

#trials := the number of cluster proposals to average over in the ensemble. Larger values of this parameter result in greater computation time, but
# more accurate results. Start with something around the number of features you have. If not specific, it will set trials = 2x number of features

#n_sub := this is the number of data points considered in each proposal clustering configuration. Should be less than 90% of the total number of data
# to introduce suitable noise into the proposals. Values <50% the number of data points result in noisier results, but less memory required for computation.
# Default parameter is set to 80% of the data points, which should be suitable in most cases

#prop_algo := the algorithm to construct cluster proposals. Currently, agglomerative clustering and k-means are supported, 
# so it must take the values: 'agglo', or 'kmeans'. Default is agglo

#class_algo := type of classifier used. Currently supports 'entropy' (default) or 'maxmargin'

#cluster_prior_min := the cluster proposals are generated such that the number of clusters in each realization of the ensemble is drawn from a uniform distribution
# between cluster_prior_min and k_guess*2

#The function will output an array of length(number of features), which is a z-score of the weight of each feature. Generally consider features with a value > 1

def SMD(X,k_guess, trials = None,n_sub = None,prop_algo = 'agglo', class_algo = 'entropy', cluster_prior_min=None):
	N = X.shape[0]
	D = X.shape[1]

	#normalize X
	X = X/X.std(0)

	if n_sub is None:
		n_sub = int(N*0.8)

	if trials is None:
		trials = int(2*D)

	if cluster_prior_min is None:
		cluster_prior_min = 3

	if prop_algo == 'agglo':
		cluster_algo = AgglomerativeClustering
	elif prop_algo == 'kmeans':
		cluster_algo = KMeans
	else:
		print(prop_algo, ' is not a supported clustering algorithm.')
		return 0

	if class_algo == 'entropy':
		get_classifier_dims = find_classifier_dims_entropy
	elif class_algo =='maxmargin':
		get_classifier_dims = find_classifier_dims_maxmargin
	else:
		print(class_algo, ' is not a supported classifier.')
		return 0

	cluster_prior = [cluster_prior_min,int(2*k_guess)]

	counts = [get_classifier_dims(X,n_sub, cluster_prior, cluster_algo) for _ in range(trials)]
	counts = [i for k in counts for i in k]
	gd,_ = np.histogram(counts, np.arange(X.shape[1]+1), density = True)
	X_shuffled = shuffle_data(X)
	counts = [get_classifier_dims(X_shuffled,n_sub, cluster_prior,cluster_algo) for _ in range(trials)]
	counts = [i for k in counts for i in k]
	g_shuffled,_ = np.histogram(counts, np.arange(X.shape[1]+1), density = True)
	Z_d = (gd-g_shuffled.mean())/g_shuffled.std()
	return Z_d


def find_classifier_dims_entropy(X,cell_sample,k_sub, cluster_algo):
	n_cells,n_genes = X.shape
	cell_use = np.random.choice(np.arange(n_cells),cell_sample,replace = False)
	k_sub_i = np.random.randint(k_sub[0],k_sub[1])
	Xit = X[cell_use,:]
	k_guess = cluster_algo(k_sub_i).fit_predict(Xit)
	gnout = [one_tree(Xit,k_guess,ikk1,ikk2) for ikk1 in np.unique(k_guess) for ikk2 in np.unique(k_guess) if ikk1<ikk2 ]
	return gnout

def find_classifier_dims_maxmargin(X,cell_sample,k_sub, cluster_algo):
	n_cells,n_genes = X.shape
	cell_use = np.random.choice(np.arange(n_cells),cell_sample,replace = False)
	k_sub_i = np.random.randint(k_sub[0],k_sub[1])
	Xit = X[cell_use,:]
	k_guess = cluster_algo(k_sub_i).fit_predict(Xit)
	gnout = [one_plane(Xit,k_guess,ikk1,ikk2) for ikk1 in np.unique(k_guess) for ikk2 in np.unique(k_guess) if ikk1<ikk2 ]
	return gnout

def one_tree(Xit,k_guess,ik1,ik2):            
	clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 1)
	XTT = Xit[np.logical_or(k_guess==ik1,k_guess==ik2),:]
	KTT = k_guess[np.logical_or(k_guess==ik1,k_guess==ik2)]
	clf.fit(XTT,KTT)
	feature_pick = np.flatnonzero(clf.feature_importances_)[0]
	return feature_pick

def one_plane(Xit,k_guess,ik1,ik2):            
	clf = LinearSVC(penalty = 'l1', dual = False, C=0.05)
	XTT = Xit[np.logical_or(k_guess==ik1,k_guess==ik2),:]
	KTT = k_guess[np.logical_or(k_guess==ik1,k_guess==ik2)]
	clf.fit(XTT,KTT)
	feature_pick = np.abs(clf.coef_).argmax()
	return feature_pick       

def shuffle_data(X):
	Xc = X.copy()
	for feature in range(X.shape[1]):
		xx = X[:,feature].copy()
		np.random.shuffle(xx)
		Xc[:,feature] = xx
	return Xc