import numpy as np
from robot_driver import BraccioRobotDriver
import time


q1_stream = np.arange(0, 92, 2)
q2_stream = np.array([90]*len(q1_stream))
q3_stream = np.array([90]*len(q1_stream))
q4_stream = np.array([90]*len(q1_stream))
q5_stream = np.array([90]*len(q1_stream))
q6_stream = np.array([10]*len(q1_stream))

q_stream = np.array([q1_stream, q2_stream, q3_stream,
                    q4_stream, q5_stream, q6_stream]).transpose()

braccio_driver = BraccioRobotDriver(
    loop_rate=0.2,
    port="5"
)

braccio_driver.connect("5")
# braccio_driver.calibrate()

for angles in q_stream:
    braccio_driver.set_joint_angles(angles)
time.sleep(5)

print(braccio_driver.read())
time.sleep(5)

# braccio_driver.homecoming()
# time.sleep(2)

# print(braccio_driver.read())
