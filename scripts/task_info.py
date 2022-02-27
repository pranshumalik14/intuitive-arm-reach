import numpy as np
import pandas as pd
import robot_arm as rb


def numpy_linspace(start, stop, step):
    return np.linspace(start, stop, int((stop-start)/step) + 1)


class TaskInfo:
    def __init__(self, robotarm, lambda_min, lambda_max, B, K, N, T, h, target_pos, dt):
        self.robotarm = robotarm
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.B = B
        self.K = K
        self.N = N
        self.T = T
        self.h = h  # eliteness param
        self.dt = dt
        self.target_pos = target_pos

        self.w = T / (2 * B)

        def gen_cs(w, T):
            return w + numpy_linspace(0, T, 2*w)  # step is 2*w
        self.cs = gen_cs(self.w, T)

    def get_robot_arm(self):
        return self.robotarm

    def get_lambda_min(self):
        return self.lambda_min

    def get_lambda_max(self):
        return self.lambda_max

    def get_target_pos(self):
        return self.target_pos

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

    def to_pd(self):
        df_dict = {}
        # df_dict["robot_arm"]  = self.robotarm.to_df()
        df_dict["lambda_min"] = self.lambda_min
        df_dict["lambda_max"] = self.lambda_max
        df_dict["target_pos"] = self.target_pos
        df_dict["B"]          = self.B
        df_dict["K"]          = self.K
        df_dict["dt"]         = self.dt
        df_dict["N"]          = self.N
        df_dict["h"]          = self.h
        df_dict["w"]          = self.w
        df_dict["cs"]         = self.cs   

        return pd.DataFrame.from_dict(
            data = df_dict, 
            dtype=object, 
            orient = 'index'
            ).transpose()

def from_pd(df):
    return TaskInfo(
        robotarm = rb.robot2D_from_df(df["robot_arm"]),
        lambda_min = df["lambda_min"],
        lambda_max = df["lambda_max"],
        B = df["B"],
        K = df["K"],
        N = df["N"],
        T = df["T"],
        h = df["h"],  # eliteness param
        dt = df["dt"],
        target_pos = df["target_pos"]
    )


