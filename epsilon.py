import scipy.sparse as spsparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def sparse_similarity(X, epsilon=0.99, Y=None, similarity_metric=cosine_similarity):
    '''
    X : ndarray
        An m by n array of m original observations in an n-dimensional space.
    '''
    Nx, Dx = X.shape
    if Y is None:
        Y=X
    Ny, Dy = Y.shape

    assert Dx==Dy


    data = []
    indices = []
    indptr = [0]
    for ix in range(Nx):
        xsim = similarity_metric([X[ix]], Y)
        _ , kept_points = np.nonzero(xsim>=epsilon)
        data.extend(xsim[0,kept_points])
        indices.extend(kept_points)
        indptr.append(indptr[-1] + len(kept_points))

    return spsparse.csr_matrix((data, indices, indptr), shape=(Nx,Ny))


X = np.random.random(size=(1000,10))
sparse_similarity(X, epsilon=0.95)