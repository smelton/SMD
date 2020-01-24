import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier


def plane_cutting(X,trials, cell_sample,cluster_param):
    feature_weights = np.hstack([PCMBK(X,cell_sample,cluster_param) for k in range(trials)])
    gx,_ = np.histogram(feature_weights, np.arange(X.shape[1]+1), density = True)
    return gx

def PCMBK(X,cell_sample,k_sub):
    n_cells,n_genes = X.shape
    cell_use = np.random.choice(np.arange(n_cells),cell_sample,replace = False)
    k_sub_i = np.random.randint(k_sub[0],k_sub[1])
    Xit = X[cell_use,:]
    k_guess = AgglomerativeClustering(k_sub_i).fit_predict(Xit)
    gnout = np.array([one_tree(Xit,k_guess,ikk1,ikk2) for ikk1 in np.unique(k_guess) for ikk2 in np.unique(k_guess) if ikk1<ikk2 ])
    return np.hstack(gnout)

def one_tree(Xit,k_guess,ik1,ik2,md = 1):            
    clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = md)
    XTT = Xit[np.logical_or(k_guess==ik1,k_guess==ik2),:]
    KTT = k_guess[np.logical_or(k_guess==ik1,k_guess==ik2)]
    clf.fit(XTT,KTT)
    feature_pick = np.flatnonzero(clf.feature_importances_)
    return feature_pick

def shuffle_data(X):
    Xc = X.copy()
    for feature in range(X.shape[1]):
        xx = X[:,feature].copy()
        np.random.shuffle(xx)
        Xc[:,feature] = xx
    return Xc

from scipy.sparse import csc_matrix,lil_matrix
def sparse_shuffle_data(X):
    Xc = lil_matrix(X.copy())
    for feature in range(X.shape[1]):
        xx = X[:,feature].copy().todense()
        np.random.shuffle(xx)
        Xc[:,feature] = xx
    return csc_matrix(Xc)

def main():
#    X = np.load(f'') X is array X.shape = (cells,genes) 
    X = mmread('mouse6p5to8p5-TFonly-matrixTlog.mtx')

    #number of iterations
    iterations = 500

    # subsampled cells. Smaller = noisier but faster
    n_sub = 3000

    #cluster number prior. if you think there might be K clusters, make this [K/2,3*K]
    CP = [3,30]

    w=plane_cutting(csc_matrix(X),iterations,n_sub,CP)

    Xs = sparse_shuffle_data(X)
    sw=plane_cutting_agglo(Xs,iterations,n_sub,CP)

    Z_d = (gx-gs.mean())/gs.std()
    np.save('z.npy',Z_d)
    return True

if __name__ == '__main__':
    main()

