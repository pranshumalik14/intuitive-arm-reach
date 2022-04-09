import numpy as np
from scipy.spatial import KDTree, Delaunay
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

class Scalar:
    def __init__(self, type="STD"):
        self.cust = False
        if type == "STD":
            self.scalar = StandardScaler()
        if type == "MABS":
            self.scalar = MaxAbsScaler()
        if type == "CUST":
            self.cust = True
            pass
    
    def scale_train(self, X):
        if self.cust:
            return X
        return self.scalar.fit_transform(X)
    
    def scale_test(self, unknown):
        if self.cust:
            return unknown
        return self.scalar.transform([unknown])[0]


class LinearInterpolator:
    def __init__(self, X, Z, rescale=False, d=False, scalar=None):
        self.X = X
        self.Z = Z

        self.scalar = None
        if scalar is not None:
            self.scalar = Scalar(scalar)
            self.X = self.scalar.scale_train(self.X)

        if d:
            self.X = Delaunay(self.X)

        self.interp = LinearNDInterpolator(self.X, self.Z, rescale=rescale)

    def __call__(self, unknown_data):
        if self.scalar is not None:
            unknown_data = self.scalar.scale_test(unknown_data)
        
        return self.interp(unknown_data)[0]


class IDW:
    def __init__(self, X, Z, leafsize=10, stat=0, K=5, p=2, scalar=None):
        self.X = X

        self.scalar = None
        if scalar is not None:
            self.scalar = Scalar(scalar)
            self.X = self.scalar.scale_train(self.X)
        self.Z = Z
        self.K = K
        self.tree = KDTree(data=self.X, leafsize=leafsize)
        self.p = p
        self.stat = stat
        
        if self.K == 1:
            
            self.interp = NearestNDInterpolator(self.X, self.Z)
        else:
            self.interp = None
            


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
        if self.scalar is not None:
            unknown_data = self.scalar.scale_test(unknown_data)

        if self.interp is not None:
            return self.interp(unknown_data)[0]
        
        neighbour_dists, neighbour_idxs = self.tree.query(x=unknown_data, k=self.K, p=self.p)
        
        if self.K == 1:
            neighbour_dists = [neighbour_dists]
            neighbour_idxs = [neighbour_idxs]

        pred = 0
        denom = [(1/(dist**2)) for dist in neighbour_dists if dist!=0]
        denom = np.sum(denom)
        
        for dist, idx in zip(neighbour_dists, neighbour_idxs):
            if dist == 0:
                weighting = 1
            else:
                weighting = (1/(dist**2))
            pred += weighting*self.Z[idx]

        ret = pred if denom == 0 else (pred/denom)
        return ret