import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time

from sklearn import neighbors

from robot_arm import RobotArm, RobotArm2D, RobotArm3D
from task_info import TaskInfo, numpy_linspace
import data_prep
from interpolators import IDW
from PIBB_helper import qdotdot_gen
from pdff_kinematic_sim_funcs import get_traj_and_simulate2d, get_traj, draw_sim_start, get_multi_traj_and_simulate2d
import matplotlib; matplotlib.use("TkAgg")
# plt.rcParams["figure.figsize"] = [15.00, 10.00]

X_TARGET = 0
Y_TARGET = 0
XS = []
YS = []

# Function to print mouse click event coordinates 
def onclick(event, robot_arm, filter_condition): 
    global X_TARGET, Y_TARGET
    X_TARGET = event.xdata
    Y_TARGET = event.ydata
    print([event.xdata, event.ydata])

    plt.close()
    fig, ax = draw_sim_start(robot_arm, filter_condition)
    plt.scatter(XS, YS, c="m", label="Training Points"); #inform matplotlib of the new data
    plt.scatter(X_TARGET, Y_TARGET, c="c", label="Target Point")
    plt.legend(fontsize=14)
    plt.show(block=False) #redraw
    ax.set_title("Generating Trajectories to Target Point", fontsize=20)
    plt.pause(5)
    plt.close()


def animate_multi_traj_to_point(interpolators, input, task_info, robot_arm, train):
    theta_preds = [interpolator(input) for interpolator in interpolators]
    thetas_reshaped = [np.reshape(pred, ( task_info.B, task_info.N)) for pred in theta_preds]

    neighbors = [None] * len(interpolators)
    for i in range(len(interpolators)):
        nearest_neighbours = interpolators[i].k_nearest_neightbours(input)
        nearest_x = nearest_neighbours[:, 0]
        nearest_y = nearest_neighbours[:, 1]
        neighbors[i] = ((nearest_x, nearest_y))

    target_pt = [input[-2], input[-1]]
    init_condit = [list(input[:3]), [0,0,0]]

    predicted_qdotdots = []
    for thet in thetas_reshaped:
        qdd = np.array(  
        [
            qdotdot_gen(task_info, thet, t) for t in numpy_linspace(0, task_info.T, task_info.dt)
        ]  
        )
        predicted_qdotdots.append(qdd)

    ani = get_multi_traj_and_simulate2d(
        qdotdots     = predicted_qdotdots, 
        robot_arm   = robot_arm, 
        x_goal      = target_pt, 
        init_condit = init_condit, 
        dt          = task_info.dt,
        train=neighbors 
        )
    plt.show()

def animate_traj_to_point(interpolator, input, task_info, robot_arm):
    theta_pred = interpolator(input)

    theta_reshaped = np.reshape(theta_pred, ( task_info.B, task_info.N))
    target_pt = [input[-2], input[-1]]
    init_condit = [list(input[:3]), [0,0,0]]
    print(target_pt)
    print(init_condit)

    predicted_qdotdot = np.array(  
        [
            qdotdot_gen(task_info, theta_reshaped, t) for t in numpy_linspace(0, task_info.T, task_info.dt)
        ]  
        )

    time_steps, q, qdot, gen_qdotdot, ani = get_traj_and_simulate2d(
        qdotdot     = predicted_qdotdot, 
        robot_arm   = robot_arm, 
        x_goal      = target_pt, 
        init_condit = init_condit, 
        dt          = task_info.dt
        )
    # to put it into the upper left corner for example:
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()

data_path = '../training_data/20220327_2215_pibb_2D.csv'
task_info_path = '../training_data/20220327_2215_task_info.csv'

concat_input_file = "concat_input.npy"
flatten_theta_file = "flatten_theta.npy"


pibb_data_df, task_info_df = data_prep.load_data(data_path, task_info_path)
task_info = data_prep.task_info_from_df(task_info_df)
robot_arm = RobotArm2D(
    n_dims = 3,
    link_lengths = np.array([0.6, 0.3, 0.1])
)
task_info.robotarm = robot_arm
task_info = data_prep.clean_task_info(task_info, task_info_df)

if concat_input_file in os.listdir():
    print("Skip data prep, files already exist")
else:
    print("Data Prep ...")
    concat_input, flatten_theta, _, _ = data_prep.clean_data(pibb_data_df, task_info)

    np.save(concat_input_file, concat_input)
    np.save(flatten_theta_file, flatten_theta)
    print("Data prep complete")

concat_input = np.load(concat_input_file)
flatten_theta = np.load(flatten_theta_file)

print("Training Interpolators ...")
SCALAR = "MABS"
nn_interp = IDW(concat_input, flatten_theta, K=1, scalar=SCALAR)
idw_interp_3 = IDW(concat_input, flatten_theta, K=3, scalar=SCALAR)
idw_interp_5 = IDW(concat_input, flatten_theta, K=5, scalar=SCALAR)
print("Interpolator Training Complete")

# Plot training data
init_conditions = np.unique(concat_input[:, 0:3], axis=0)

while(True):
    for i in range(len(init_conditions)):
        print(str(i) + ": " + str(init_conditions[i]))
    init_choice = int(input("Choose an initial condition: "))

    filter_condition = init_conditions[init_choice]
    filter = (concat_input[:,0] == filter_condition[0]) & (concat_input[:,1] == filter_condition[1]) & (concat_input[:,2] == filter_condition[2])
    filtered = concat_input[filter]
    c_xs = filtered[:, -2]
    c_ys = filtered[:, -1]
    XS = c_xs
    YS = c_ys
    filter_condition = np.reshape(filter_condition, (1, 3))


    fig, ax = draw_sim_start(robot_arm, filter_condition)
    plt.scatter(c_xs, c_ys, label="Training Points", c="m")
    fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, robot_arm, filter_condition))

    plt.get_current_fig_manager().full_screen_toggle()
    plt.legend(fontsize=14)
    ax.set_title("Choose a target point", fontsize=20)
    plt.show()
    # animate_traj_to_point(nn_interp, [*filter_condition[0], X_TARGET, Y_TARGET], task_info, robot_arm)
    # animate_traj_to_point(idw_interp_5, [*filter_condition[0], X_TARGET, Y_TARGET], task_info, robot_arm)
    interps = [nn_interp, idw_interp_5]
    animate_multi_traj_to_point(interps, [*filter_condition[0], X_TARGET, Y_TARGET], task_info, robot_arm, train=[XS, YS])