import os
import time
import numpy as np
import keyboard
from spatialmath import SE3
from src.robot.Braccio import Braccio
from src.driver.robot_driver import BraccioRobotDriver
from src.vision.vision import vision_run
from multiprocessing import Process, Pipe
from roboticstoolbox.backends.swift import Swift
import spatialgeometry as sg
from scripts.robot_arm import Braccio3D
from scripts.pdff_kinematic_sim_funcs import pdff_traj


# set reach solver and operation mode
global SOLVER
SOLVER = "IK"  # "IK", "RTA", or "RTM" (reach task algo/model)
global MODE
MODE = "DEMO"  # or "SIMULATION"
# MODE = "SIMULATION"

# define system state
global STATE
STATE = "calibrate_origin"

# setup robot object
global braccio
main_path = os.getcwd()
# braccio = Braccio()
braccio = Braccio3D()
os.chdir(main_path)

# setup visualization environment
global env, goal_obj
braccio.q = np.deg2rad([0, 15, 90, 90])
env = Swift()
env.launch()
env.add(braccio)
goal_obj = sg.Cuboid(
    scale=[0.05, 0.05, 0.05],
    base=SE3(0, 0, 0),
    color="blue"
)
env.add(goal_obj)

# define goal region
global goal_min, goal_max
goal_min, goal_max = np.array(
    [[-0.37, 0.37], [0, 0.37], [0.05, 0.5]]).T

# create a pipe to communicate with the vision process
global main_driverproc_pipe, vision_subproc_pipe, vision_worker
main_driverproc_pipe, vision_subproc_pipe = Pipe(duplex=True)
vision_worker = Process(target=vision_run, args=(vision_subproc_pipe,))

# vision data containers
global vis_msg, orig2vis_frame
vis_msg = None
orig2vis_frame = None

global rad_q, deg_q
deg_q = np.array([0, 30, 90, 90, 90, 10])
rad_q = np.deg2rad(deg_q)

if __name__ == "__main__":
    def check_procexit_wait():
        global main_driverproc_pipe, STATE, vis_msg, SOLVER

        if main_driverproc_pipe.poll():
            vis_msg = main_driverproc_pipe.recv()
        elif keyboard.is_pressed("q") or keyboard.is_pressed("esc"):
            print("[MAIN] Received Vision Kill Request")
            main_driverproc_pipe.send("vision_stop")
            # vis_msg = main_driverproc_pipe.recv()
            STATE = "vision_terminated"
        elif keyboard.is_pressed("i"):
            SOLVER = "IK"
            print("[MAIN] Solver set to IK")
        elif keyboard.is_pressed("a"):
            SOLVER = "RTA"
            print("[MAIN] Solver set to RTA")
        elif keyboard.is_pressed("m"):
            SOLVER = "RTM"
            print("[MAIN] Solver set to RTM")
        time.sleep(0.2)

    def update_viz(q, goal_pos, dt=0.2):
        global env, goal_obj, braccio

        braccio.q = q
        env.remove(goal_obj)
        goal_obj.base = SE3(goal_pos)
        env.add(goal_obj)
        env.step(dt=dt)

    def solver_loop():
        global main_driverproc_pipe, STATE, vis_msg, SOLVER, orig2vis_frame
        global goal_min, goal_max, braccio_driver, braccio, env, goal_obj
        global deg_q, rad_q

        if STATE == "calibrate_origin":
            main_driverproc_pipe.send("vision_calib_orig2vis_frame")
            orig2vis_frame = main_driverproc_pipe.recv()
            print("[MAIN] Camera Origin: {}".format(orig2vis_frame.t))
            STATE = "get_curr_goal"
        elif STATE == "get_curr_goal":
            main_driverproc_pipe.send("vision_curr_orig2goal_pos")
            goal_pos = main_driverproc_pipe.recv()
            print("[MAIN] Current Goal: {}".format(goal_pos))

            # make sure not invalid (i.e. [-1, -1, -1]) or out of bounds
            if np.array_equal(goal_pos, np.array([-1, -1, -1])) or \
                    ((goal_pos < goal_min) | (goal_pos > goal_max)).any():
                print("[MAIN] Goal Invalid (Skipped)")
                check_procexit_wait()
                return

            # get current joint config
            if MODE == "DEMO":
                deg_q = np.array(braccio_driver.read()['joint_angles'])
                print("[MAIN] Current Joint Configuration: {}".format(deg_q))
                rad_q = np.deg2rad(deg_q)
            update_viz(rad_q[:-2], goal_pos)

            # send problem to reach solver and make robot go to goal pos
            if SOLVER == "IK":
                q_sol = np.rad2deg(
                    braccio.inverse_kinematics(goal_pos, curr_q=rad_q)
                ).astype(int)
                q_sol = np.append(q_sol, deg_q[-2:])
                if ((q_sol >= 0) & (q_sol <= 180)).all():
                    if MODE == "DEMO":
                        braccio_driver.set_joint_angles(q_sol)
                    deg_q = q_sol
                    rad_q = np.deg2rad(deg_q)
                    update_viz(rad_q[:-2], goal_pos)
                else:
                    print("[MAIN] IK qsol Invalid (Skipped)")
            elif SOLVER == "RTA":
                q_sol = np.rad2deg(
                    pdff_traj([rad_q[:-2], np.zeros(4)], goal_pos, braccio)
                ).astype(int)

                for i, q in enumerate(q_sol):
                    if ((i != (len(q_sol)-1)) and (i % 2 == 0)):
                        continue
                    if ((q >= 0) & (q <= 180)).all():
                        if MODE == "DEMO":
                            braccio_driver.set_joint_angles(np.append(q, deg_q[-2:]))
                        deg_q = np.append(q, deg_q[-2:])
                        rad_q = np.deg2rad(deg_q)
                        update_viz(rad_q[:-2], goal_pos)
            elif SOLVER == "RTM":
                raise NotImplementedError  # todo
            else:
                check_procexit_wait()
                # raise ValueError("Invalid Solver")
            check_procexit_wait()

    if MODE == "DEMO":
        # setup robot driver
        global braccio_driver
        braccio_driver = BraccioRobotDriver(
            loop_rate=0.2,
            port="4"
        )

        # make robot go to calibration position
        braccio_driver.calibrate()

        # start vision process
        vision_worker.start()

        while STATE != "vision_terminated":
            solver_loop()

        vision_worker.join()
        vision_worker.close()

    elif MODE == "SIMULATION":
        # start vision process
        vision_worker.start()

        while STATE != "vision_terminated":
            solver_loop()

        vision_worker.join()
        vision_worker.close()

    else:
        raise ValueError("Invalid mode: {}".format(MODE))
