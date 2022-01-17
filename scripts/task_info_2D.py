import numpy as np


def numpy_linspace(start, stop, step):
    return np.linspace(start, stop, int((stop-start)/step) + 1)


class TaskInfo2D:
    def __init__(self, robotarm, lambda_min, lambda_max, B, K, N, T, h, target_xy, dt):
        self.robotarm = robotarm
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.B = B
        self.K = K
        self.N = N
        self.T = T
        self.h = h  # eliteness param
        self.dt = dt
        self.target_xy = target_xy

        self.w = T / (2 * B)

        def gen_cs(w, T):
            return w + numpy_linspace(0, T, 2*w)  # step is 2*w
        self.cs = gen_cs(self.w, T)

    def get_robot_arm_2D(self):
        return self.robotarm

    def get_lambda_min(self):
        return self.lambda_min

    def get_lambda_max(self):
        return self.lambda_max

    def get_target_xy(self):
        return self.target_xy

    def get_B(self):
        return self.B

    def get_K(self):
        return self.K

    def get_N(self):
        return self.N

    def get_T(self):
        return self.T

    def get_h(self):
        return self.h

    def get_dt(self):
        return self.dt

    def get_w(self):
        return self.w

    def get_cs(self):
        return self.cs
