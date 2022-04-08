from scipy.spatial import KDTree, Delaunay
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
import numpy as np

class idw:
    def __init__(self, X, Z, leafsize=10, stat=0, K=5, p=2):
        self.tree = KDTree(data=X, leafsize=leafsize)
        self.X = X
        self.Z = Z
        self.stat = stat
        self.K = K
        self.p = p
        self.nn_interp = NearestNDInterpolator(X, Z)


    def k_nearest_neightbours(self, unknown_data):
        _, neighbour_idxs = self.tree.query(x=unknown_data, k=self.K, p=self.p)
        
        if self.K == 1:
            neighbour_idxs = [neighbour_idxs]
        
        neighbours = []
        for idx in (neighbour_idxs):
            neighbour = np.array([self.X[idx][-2], self.X[idx][-1]])
            neighbours.append(neighbour)
        neighbours = np.array(neighbours)
        return neighbours
    

    def __call__(self, unknown_data):
        neighbour_dists, neighbour_idxs = self.tree.query(x=unknown_data, k=self.K, p=self.p)
        
        if self.K == 1:
            neighbour_dists = [neighbour_dists]
            neighbour_idxs = [neighbour_idxs]

        pred = 0
        denom = [(1/(dist**2)) for dist in neighbour_dists if dist!=0]
        denom = np.sum(denom)
        
        for dist, idx in zip(neighbour_dists, neighbour_idxs):
            # print(dist, idx)
            if dist == 0:
                weighting = 1
            else:
                weighting = (1/(dist**2))
            pred += weighting*self.nn_interp(self.X[idx])

        ret = pred if denom == 0 else (pred/denom)
        return ret

def get_interpolator(style, inp, real, K=None, leafsize=10):
    if style == "NN":
        return NearestNDInterpolator(inp, real)
    if style == "LIN":
        return LinearNDInterpolator(inp, real, rescale=True)
    if style == "DEL":
        return LinearNDInterpolator(Delaunay(inp), real)
    if style == "IDW":
        assert(K is not None)
        return idw(inp, real, leafsize=leafsize, K=K, stat=0)
    raise Exception("Invalid Interpolator Style")