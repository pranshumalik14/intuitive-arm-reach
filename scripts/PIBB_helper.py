import numpy as np

# Helper functions


def psi(c, t, w):
    return np.exp(-((t-c)/w)**2)


def psi_cumulative(t, cs, w, B):
    return [psi(cs[b], t, w) for b in range(B)]


def g(b, t, task_info):
    cs = task_info.get_cs()
    B = task_info.get_B()
    w = task_info.get_w()
    return psi(cs[b], t, w)/sum(psi_cumulative(t, cs, w, B))


def g_over_b(task_info, t):
    B = task_info.get_B()
    return np.array([g(b, t, task_info) for b in range(B)])


def qdotdot_gen(task_info, Theta, t):
    return np.matmul(Theta.transpose(), g_over_b(task_info, t))
