import time
import keyboard
from vision.vision import vision_run
from multiprocessing import Process, Pipe

MODE = "DEMO"  # or "SIMULATION"

if __name__ == "__main__":
    if MODE == "DEMO":
        # create a pipe to communicate with the vision process
        main_driverproc_pipe, vision_subproc_pipe = Pipe(duplex=True)

        vision_worker = Process(target=vision_run, args=(vision_subproc_pipe,))
        vision_worker.start()

        vis_msg = None
        orig2vis_frame = None
        state = "calibrate_origin"

        # todo: if requesting goal pos: make sure not invalid (i.e. [-1, -1, -1])
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

            if main_driverproc_pipe.poll():
                vis_msg = main_driverproc_pipe.recv()
            elif keyboard.is_pressed("q") or keyboard.is_pressed("esc"):
                print("[MAIN] Received Vision Kill Request")
                main_driverproc_pipe.send("vision_stop")
                vis_msg = main_driverproc_pipe.recv()
                state = "vision_terminated"

        vision_worker.join()
        vision_worker.close()
    elif MODE == "SIMULATION":
        raise NotImplementedError
    else:
        raise ValueError("Invalid mode: {}".format(MODE))
