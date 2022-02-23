# -*- coding: utf-8 -*-
"""
This is the example to execute the planar3DOF robot

@author olmerg

"""
#import sys  
#sys.path.insert(0, 'planar_3d')

from Braccio import Braccio

import roboticstoolbox as rtb
import numpy as np
from math import pi
import time

robot=Braccio()
print(robot)
print(robot.links)

T =robot.fkine(robot.qz)
print(T)

# robot.plot(q=robot.qz)


# robot.plot(robot.qz,backend='swift')
from roboticstoolbox.backends.swift import Swift
env = Swift()
env.launch(realTime = True)
env.add(robot)

# env.step(1.0)


qt = rtb.tools.trajectory.jtraj(np.array([0, 0.3, pi/2, pi/2, pi/2, 0.18, 1.28]), np.array([pi/2, pi/2, pi, pi/2, pi/2, 1.2, 2.3]), 1000)
    
for q in qt.q:
         print(q)
         robot.q=q
         env.step(3)
    # return to home


# robot.teach(backend='pyplot')
env.hold()

