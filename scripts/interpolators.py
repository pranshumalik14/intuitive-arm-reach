import numpy as np
from scipy.spatial import KDTree, Delaunay
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler

class Scalar:
    def __init__(self, type="STD"):
        self.cust = False
        if type == "STD":
            self.scalar = StandardScaler()
        if type == "MABS":
            self.scalar = MaxAbsScaler()
        if type == "CUST":
            self.cust = True
            self.angle_scalar = MinMaxScaler()
            self.point_scalar = MinMaxScaler()
    
    def scale_train(self, X):
        if self.cust:
            scaled_X = np.zeros(X.shape)
            scaled_X[:, 0:3] = 0.6*self.angle_scalar.fit_transform(X[:, 0:3])
            scaled_X[:, 3:] = 0.4*self.point_scalar.fit_transform(X[:, 3:])
            return scaled_X

        return self.scalar.fit_transform(X)
    
    def scale_test(self, unknown):
        if self.cust:
            scaled_unknown = np.zeros(unknown.shape)
            scaled_unknown[0:3] = 0.6*self.angle_scalar.transform([unknown[0:3]])[0]
            scaled_unknown[3:] = 0.4*self.point_scalar.transform([unknown[3:]])[0]
            return scaled_unknown
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
    def __init__(self, X, Z, leafsize=10, stat=0, K=5, p=1, threshold=np.inf, scalar=None):
        self.X = X
        self.unscaled_x = X
        self.scalar = None
        if scalar is not None:
            self.scalar = Scalar(scalar)
            self.X = self.scalar.scale_train(self.X)
        self.Z = Z
        self.K = K
        self.tree = KDTree(data=self.X, leafsize=leafsize)
        self.p = p
        self.stat = stat
        self.threshold = threshold
        
        if self.K == 1:
            
            self.interp = NearestNDInterpolator(self.X, self.Z)
        else:
            self.interp = None
            


    def k_nearest_neightbours(self, unknown_data):
        if self.scalar is not None:
            unknown_data = self.scalar.scale_test(unknown_data)
        _, neighbour_idxs = self.tree.query(x=unknown_data, k=self.K, p=self.p, distance_upper_bound=self.threshold)
        non_thresh_idxs = []
        if self.threshold != np.inf:
            _, non_thresh_idxs = self.tree.query(x=unknown_data, k=self.K, p=self.p)
        rejected_idxs = [idx for idx in non_thresh_idxs if idx not in neighbour_idxs]
        
        if self.K == 1:
            neighbour_idxs = [neighbour_idxs]
        
        neighbours = []
        neighbours_all = []
        for idx in (neighbour_idxs):
            if idx != len(self.X):
                neighbours_all.append(self.unscaled_x[idx])
                neighbour = np.array([self.unscaled_x[idx][-2], self.unscaled_x[idx][-1]])
                neighbours.append(neighbour)
        neighbours = np.array(neighbours)

        rejected = []
        for idx in (rejected_idxs):
            rej = np.array([self.unscaled_x[idx][-2], self.unscaled_x[idx][-1]])
            rejected.append(rej)
        rejected = np.array(rejected)
        # print("K: " + str(self.K))
        # print("Neighbours")
        # print(neighbours_all)
        return neighbours, rejected

    def nearest_neighbour_dist(self, unknown_data):
        if self.scalar is not None:
            unknown_data = self.scalar.scale_test(unknown_data)
        neighbour_dist, _ = self.tree.query(x=unknown_data, k=1, p=self.p)
        return neighbour_dist

    def __call__(self, unknown_data):
        if self.scalar is not None:
            unknown_data = self.scalar.scale_test(unknown_data)

        if self.interp is not None:
            return self.interp(unknown_data)[0]
        
        neighbour_dists, neighbour_idxs = self.tree.query(x=unknown_data, k=self.K, p=self.p, distance_upper_bound=self.threshold)
        print("neighbour distances")
        print(neighbour_dists)
        # print(neighbour_idxs)

        if len(set(neighbour_idxs)) == 1:
            neighbour_dists, neighbour_idxs = self.tree.query(x=unknown_data, k=1, p=self.p) 

        if self.K == 1:
            neighbour_dists = [neighbour_dists]
            neighbour_idxs = [neighbour_idxs]

        pred = 0
        denom = [(1/(dist**2)) for dist in neighbour_dists if (dist!=0 and dist!=np.inf)]
        denom = np.sum(denom)
        
        for dist, idx in zip(neighbour_dists, neighbour_idxs):
            if dist == np.inf:
                continue
            if dist == 0:
                weighting = 1
            else:
                weighting = (1/(dist**2))
            pred += weighting*self.Z[idx]

        ret = pred if denom == 0 else (pred/denom)
        print("input")
        print(unknown_data)
        print("output")
        print(ret)
        return ret