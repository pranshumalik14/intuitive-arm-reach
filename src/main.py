import os
import time
import numpy as np
import keyboard
from spatialmath import SE3
from robot.Braccio import Braccio
from driver.robot_driver import BraccioRobotDriver
from vision.vision import vision_run
from multiprocessing import Process, Pipe

MODE = "DEMO"  # or "SIMULATION"

if __name__ == "__main__":
    if MODE == "DEMO":
        # setup robot driver and robot objects
        main_path = os.getcwd()
        braccio_driver = BraccioRobotDriver(
            loop_rate=0.2,
            port="5"
        )
        braccio = Braccio()
        os.chdir(main_path)

        # make robot go to calibration position
        braccio_driver.calibrate()

        # set reach solver
        SOLVER = "IK"  # "RTA", or "RTM" (reach task algo/model)

        # create a pipe to communicate with the vision process
        main_driverproc_pipe, vision_subproc_pipe = Pipe(duplex=True)

        vision_worker = Process(target=vision_run, args=(vision_subproc_pipe,))
        vision_worker.start()

        vis_msg = None
        orig2vis_frame = None
        state = "calibrate_origin"

        while vis_msg != "vision_exiting":
            if state == "calibrate_origin":
                main_driverproc_pipe.send("vision_calib_orig2vis_frame")
                orig2vis_frame = main_driverproc_pipe.recv()
                print("[MAIN] Camera Origin: {}".format(orig2vis_frame.t))
                state = "get_curr_goal"
            elif state == "get_curr_goal":
                main_driverproc_pipe.send("vision_curr_orig2goal_pos")
                goal_pos = main_driverproc_pipe.recv()
                print("[MAIN] Current Goal: {}".format(goal_pos))

                # make sure not invalid (i.e. [-1, -1, -1])
                if (goal_pos == np.array([-1, -1, -1])).all():
                    print("[MAIN] Goal Invalid (Skipped)")
                    continue

                # get current joint config
                curr_q = np.array(braccio_driver.read()['joint_angles'])
                print("[MAIN] Current Joint Configuration: {}".format(curr_q))

                # send problem to reach solver and make robot go to goal pos
                if SOLVER == "IK":
                    q_sol = np.rad2deg(
                        braccio.ikine_min(
                            SE3(goal_pos),
                            qlim=True
                        ).q
                    ).astype(int)
                    q_sol = np.append(q_sol, curr_q[-2:])
                    braccio_driver.set_joint_angles(q_sol)
                elif SOLVER == "RTA":
                    raise NotImplementedError  # todo
                elif SOLVER == "RTM":
                    raise NotImplementedError  # todo
                else:
                    raise ValueError("Invalid Solver")

            if main_driverproc_pipe.poll():
                vis_msg = main_driverproc_pipe.recv()
            elif keyboard.is_pressed("q") or keyboard.is_pressed("esc"):
                print("[MAIN] Received Vision Kill Request")
                main_driverproc_pipe.send("vision_stop")
                vis_msg = main_driverproc_pipe.recv()
                state = "vision_terminated"

            time.sleep(0.5)
            # todo: make robot 4dof.

        vision_worker.join()
        vision_worker.close()
    elif MODE == "SIMULATION":
        raise NotImplementedError
        # generate linear trajectory (each joint)/PDFF traj
        # q stream
        # simulate that
        # get new q stream
        # at each loop, we get new goal pos and plot it
    else:
        raise ValueError("Invalid mode: {}".format(MODE))
