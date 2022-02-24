from roboticstoolbox.backends.swift import Swift
from Braccio import Braccio
from spatialmath import SE3
import roboticstoolbox as rtb
import numpy as np
from numpy import pi

robot = Braccio()

print(robot.fkine([np.pi/2, np.pi/2, 0, 0, 0]).t)

Tend = SE3([1,  0,  0])
q = robot.ikine_LM(Tend, mask=[1, 1, 1, 0, 0, 0])
robot.q = q.q
print(np.rad2deg(q.q))
robot.plot(robot.q)

env = Swift()
# env.launch(realTime=True)
# env.add(robot)

# qt = rtb.tools.trajectory.jtraj(np.array(
#     [0, 0.3, pi/2, pi/2, pi/2]), np.array([pi/2, pi/2, pi, pi/2, pi/2]), 1000)

# for q in qt.q:
#     robot.q = q
#     env.step(3)
env.hold()
