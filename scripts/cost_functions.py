import imp


import numpy as np
from robot_arm_2D import angles_to_link_positions_2D

def acceleration_cost(qdotdot, t):
  N = qdotdot.shape[1]
  numerator = 0
  for n in range(qdotdot.shape[1]):
    numerator += (N+1-n) * qdotdot[t, n]**2
  return numerator


def cost_function(x_target, q, qdotdot, robot_arm, kt = 1e-5, kT = 1e2):
  fk = angles_to_link_positions_2D(q, robot_arm)
  fk_dim = fk.shape[1] # x y for each dim 
  x_TN = np.array([fk[-1, fk_dim-2], fk[-1, fk_dim-1]])
  # take the last row and last column/second last column

  # reach cost
  norm_reach_cost = np.linalg.norm(x_TN - x_target)**2 # The squared-distance between the end-effector and the goal positions at the end of the movement
  comfort_cost = np.max(q[-1, :]) #: A cost that corresponds to the largest angle over all the joints at the end of the movement
  C_T = kT*(norm_reach_cost) + comfort_cost

  # acce cost
  N = qdotdot.shape[1]
  denom = 0
  for n in range(qdotdot.shape[1]):
    denom += (N+1 - n)

  acce_cost = 0
  for t in range(qdotdot.shape[0]):
    acce_cost += acceleration_cost(qdotdot, t)
  acce_cost = (kt*acce_cost)/denom

  # objective function
  J_Qdd = C_T + acce_cost
  return J_Qdd