# SMD

See sample_usage.py for an example.

The function SMD has the following inputs:

X := Data matrix with dimensions (number of data points) x (number of features).

k_guess := best guess of how many clusters are present. can be approximate. Better to guess too high than too low.

trials := the number of cluster proposals to average over in the ensemble. Larger values of this parameter result in greater computation time, but
more accurate results. Start with something around the number of features you have. If not specific, it will set trials = 2x number of features

n_sub := this is the number of data points considered in each proposal clustering configuration. Should be less than 90% of the total number of data
to introduce suitable noise into the proposals. Values <50% the number of data points result in noisier results, but less memory required for computation.
Default parameter is set to 80% of the data points, which should be suitable in most cases

prop_algo := the algorithm to construct cluster proposals. Currently, agglomerative clustering and k-means are supported, 
so it must take the values: 'agglo', or 'kmeans'. Default is agglo

class_algo := type of classifier used. Currently supports 'entropy' (default) or 'maxmargin'

cluster_prior_min := the cluster proposals are generated such that the number of clusters in each realization of the ensemble is drawn from a uniform distribution
between cluster_prior_min and k_guess*2

The function will output an array of length(number of features), which is a z-score of the weight of each feature. Generally consider features with a value > 1

Requires Ray, numpy, and sklearn

Typical usage, for some X

z = SMD(X,7,trials = 1000)

SMD_serial.py and sample_usage_serial.py are contingencies that implement the algorithm without parallelization. It will be slower, but works if Ray is giving errors.

See here for details: https://arxiv.org/abs/1910.05814

Ramanathan Lab members, or those using the Harvard Odyssey cluster, see Ramanathan_Internal for instructions on running via slurm job arrays instead of using Ray to parallelize.
