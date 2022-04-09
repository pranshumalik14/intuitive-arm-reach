import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA
import matplotlib.animation as animation
from collections import deque
import time
import os
import pandas as pd
from roboticstoolbox.backends.swift import *

from robot_arm import RobotArm2D, Braccio3D, RobotArm
from task_info import numpy_linspace, TaskInfo
from cost_functions import cost_function
from PIBB_helper import qdotdot_gen
from data_prep import clean_data

from spatialmath import SE3
import spatialgeometry as sg


def get_traj(qdotdot, robot_arm, dt, init_condit=[None, None]):
    """
    Takes in a joint accelerations qdotdot, and return

    Keyword arguments:
    argument -- qdotdot: an t x n_dim matrix (t timesteps, n_dim dimensions) (numpy array)
    Return: return_description
    """

    n_time_steps = qdotdot.shape[0]
    n_dims_qdotdot = qdotdot.shape[1] if qdotdot.ndim > 1 else 1

    n_dims, arm_length, link_lengths = robot_arm.get_arm_params()
    # assert(n_dims == n_dims_qdotdot)

    time_steps = dt * np.arange(n_time_steps)

    # initialize q, qdot to 0
    q = np.zeros((n_time_steps, n_dims))
    qdot = np.zeros((n_time_steps, n_dims))

    if init_condit is not None:
        q0, qdot0 = init_condit[0], init_condit[1]
        # assert(qdot.shape[1] == len(qdot0))
        qdot[0, :] = qdot0
        q[0, :] = q0
    else:
        # if None, then assume 0 init conditions
        pass

    if qdotdot.ndim > 1:
        for i in range(1, n_time_steps):
            qdot[i, :] = qdot[i-1, :] + dt*qdotdot[i, :]
            q[i, :] = q[i-1, :] + dt*qdot[i, :]
    else:
        for i in range(1, n_time_steps):
            qdot[i] = qdot[i-1] + dt*qdotdot[i]
            q[i] = q[i-1] + dt*qdot[i]

    # Could use np.trapez but it gave me some weird error
    # qdot[:,0] = np.trapez(qdotdot, x = time_steps)
    # https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html
    return time_steps, q, qdot, qdotdot

def draw_sim_start(robot_arm, init_condit):
    # Forward kinematics
    [link_positions_x, link_positions_y] = robot_arm.angles_to_link_positions(init_condit)

    # Robot params
    n_dims, arm_length, link_lengths = robot_arm.get_arm_params()

    # Figure set up
    fig = plt.figure()
    padding_factor = 1.5
    axes_lim = arm_length*padding_factor

    ax = fig.add_subplot(autoscale_on=False, xlim=(-axes_lim,
                                                   axes_lim), ylim=(-axes_lim, axes_lim))
    ax.set_aspect('equal')
    ax.grid()

    # Adding the base of the robot arm
    points = [[0, 0], [-0.05, -0.1], [0.05, -0.1]]
    line = plt.Polygon(points, closed=True, fill=True, color='red')
    plt.gca().add_patch(line)

    line, = ax.plot([], [], 'o-', lw=2)
    line.set_data(link_positions_x[0,:], link_positions_y[0,:])

    return fig, ax


def get_multi_traj_and_simulate2d(qdotdots, robot_arm, x_goal, init_condit, dt, train):
    # Get q and qdot
    time_steps_1, q_1, qdot_1, qdotdot_1 = get_traj(
        qdotdots[0], 
        robot_arm, 
        dt,
        init_condit = init_condit
    )
    time_steps_2, q_2, qdot_2, qdotdot_2 = get_traj(
        qdotdots[1], 
        robot_arm, 
        dt,
        init_condit = init_condit
    )
    n_time_steps = len(time_steps_1)

    # Forward kinematics
    [link_positions_x1, link_positions_y1] = robot_arm.angles_to_link_positions(q_1)
    [link_positions_x2, link_positions_y2] = robot_arm.angles_to_link_positions(q_2)
    # Robot params
    n_dims, arm_length, link_lengths = robot_arm.get_arm_params()

    # Figure set up
    fig, axs = plt.subplots(1, 2)
    padding_factor = 1.5
    axes_lim = arm_length*padding_factor
    
    for ax in axs:
        ax.set(autoscale_on=False, xlim=(-axes_lim, axes_lim), ylim=(-axes_lim, axes_lim), aspect='equal')
        ax.grid()

    # tracking the history of movements
    tracking_history_points = 1000
    history_x1, history_y1 = deque(maxlen=tracking_history_points), deque(
        maxlen=tracking_history_points)
    history_x2, history_y2 = deque(maxlen=tracking_history_points), deque(
        maxlen=tracking_history_points)

    # Adding the base of the robot arm
    points = [[0, 0], [-0.05, -0.1], [0.05, -0.1]]
    line1 = plt.Polygon(points, closed=True, fill=True, color='red')
    line2 = plt.Polygon(points, closed=True, fill=True, color='red')
    axs[0].add_patch(line1)
    axs[0].scatter(train[0][0], train[0][1], c="y", label="neighbours")
    axs[1].add_patch(line2)
    axs[1].scatter(train[1][0], train[1][1], c="y", label="neighbours")

    # Dynamic lines (theese are the lines/vals that will update during the simulation)
    line1, = axs[0].plot([], [], 'o-', lw=2)
    line2, = axs[1].plot([], [], 'o-', lw=2)
    line = [line1, line2]

    # animation for each frame
    def animate(i):
        # all x axis values are in the even-numbered columns
        thisx1 = link_positions_x1[i, :] # get the current row of joint angles (x vals)
        thisx2 = link_positions_x2[i, :] # get the current row of joint angles (x vals)
        # all y axis value are in the odd-numbered columns
        thisy1 = link_positions_y1[i, :] # get the current row of joint angles (y vals)
        thisy2 = link_positions_y2[i, :] # get the current row of joint angles (y vals)

        if i == 0:
            history_x1.clear()
            history_y1.clear()
            history_x2.clear()
            history_y2.clear()

        # History only tracks the end effector
        history_x1.appendleft(thisx1[-1])
        history_y1.appendleft(thisy1[-1])
        history_x2.appendleft(thisx2[-1])
        history_y2.appendleft(thisy2[-1])

        # Set current state of (x,y) for each joint
        line[0].set_data(thisx1, thisy1)
        # trace[0].set_data(history_x1, history_y1)
        line[1].set_data(thisx2, thisy2)
        # trace[1].set_data(history_x2, history_y2)

        return line #, trace, time_text

    ani = animation.FuncAnimation(
        fig, animate, n_time_steps, interval=30*(dt* n_time_steps), blit=True
    )

    # Goal position
    axs[0].plot(x_goal[0], x_goal[1], '-o')  # Goal position
    axs[1].plot(x_goal[0], x_goal[1], '-o')  # Goal position
    axs[0].annotate('x_g', xy=(x_goal[0], x_goal[1]))
    axs[1].annotate('x_g', xy=(x_goal[0], x_goal[1]))

    annotation_str = "Initial Configuration: {}\nTarget Point: {}".format([round(i,3) for i in init_condit[0]], x_goal)
    axs[0].set_title('Kinematic Simulation: {}NN'.format(len(train[0][0])), fontsize=14)
    axs[1].set_title('Kinematic Simulation: {}NN'.format(len(train[1][0])), fontsize=14)
    axs[0].legend(fontsize=14)
    axs[1].legend(fontsize=14)
    plt.get_current_fig_manager().full_screen_toggle()

    return ani

def get_traj_and_simulate2d(qdotdot, robot_arm, x_goal, init_condit, dt):
    """

    """
    # initial checks
    # assert(len(x_goal) == 2 and isinstance(robot_arm, RobotArm2D))

    # Get q and qdot

    time_steps, q, qdot, qdotdot = get_traj(
        qdotdot, 
        robot_arm, 
        dt,
        init_condit = init_condit
    )
    n_time_steps = len(time_steps)

    # Forward kinematics
    [link_positions_x, link_positions_y] = robot_arm.angles_to_link_positions(q)
    # Robot params
    n_dims, arm_length, link_lengths = robot_arm.get_arm_params()

    # Figure set up
    fig = plt.figure()
    padding_factor = 1.5
    axes_lim = arm_length*padding_factor

    ax = fig.add_subplot(autoscale_on=False, xlim=(-axes_lim,
                                                   axes_lim), ylim=(-axes_lim, axes_lim))
    ax.set_aspect('equal')
    ax.grid()

    # tracking the history of movements
    tracking_history_points = 1000
    history_x, history_y = deque(maxlen=tracking_history_points), deque(
        maxlen=tracking_history_points)

    # Adding the base of the robot arm
    points = [[0, 0], [-0.05, -0.1], [0.05, -0.1]]
    line = plt.Polygon(points, closed=True, fill=True, color='red')
    plt.gca().add_patch(line)

    # Dynamic lines (theese are the lines/vals that will update during the simulation)
    line, = ax.plot([], [], 'o-', lw=2)
    trace, = ax.plot([], [], '.-', lw=1, ms=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    # Annotations for each link/end effector
    # annotate_links = []
    # for n in range(n_dims):
    #     link_to_annotate = "q{}".format(str(1+n))
    #     annotate_links.append(ax.annotate(
    #         link_to_annotate, xy=(link_lengths[n], 0)))

    annotate_end_effector = ax.annotate('E', xy=(link_lengths[-1], 0))

    # animation for each frame
    def animate(i):
        # all x axis values are in the even-numbered columns
        # thisx = [link_positions[i, j] for j in range(0, len(link_positions[0, :]), 2)]
        thisx = link_positions_x[i, :] # get the current row of joint angles (x vals)
        # all y axis value are in the odd-numbered columns
        # thisy = [link_positions[i, j] for j in range(1, len(link_positions[0, :]), 2)]
        thisy = link_positions_y[i, :] # get the current row of joint angles (y vals)

        if i == 0:
            history_x.clear()
            history_y.clear()

        # History only tracks the end effector
        history_x.appendleft(thisx[-1])
        history_y.appendleft(thisy[-1])

        # Set current state of (x,y) for each joint
        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        time_text.set_text(time_template % (i*dt))

        offset_factor = 0.3
        # This annotates the links but it wasn't keepin up in the animation
        # for n in range(n_dims):
        #     annotate_links[n].set_position(
        #         (
        #             link_positions_x[i, n] + offset_factor *
        #             link_lengths[n]*np.cos(q[i, n]),
        #             link_positions_x[i, n] + offset_factor *
        #             link_lengths[n]*np.sin(q[i, n])
        #         )
        #     )

        annotate_end_effector.set_position((
            link_positions_x[i, -1],
            link_positions_y[i, -1]
        ))

        return line, trace, time_text

    ani = animation.FuncAnimation(
        fig, animate, n_time_steps, interval=10*(dt* n_time_steps), blit=True
    )

    # Goal position
    plt.plot(x_goal[0], x_goal[1], '-o')  # Goal position
    ax.annotate('x_g', xy=(x_goal[0], x_goal[1]))

    annotation_str = "Initial Configuration: {}\nTarget Point: {}".format([round(i,3) for i in init_condit[0]], x_goal)
    fig.suptitle('Kinematic Simulation: \n' + annotation_str, fontsize=8)

    return time_steps, q, qdot, qdotdot, ani

def eval_rollout(task_info, Theta_matrix, init_condit):
    T = task_info.get_T()
    dt = task_info.get_dt()
    robot_arm = task_info.get_robot_arm()
    target_pos = task_info.get_target_pos()

    # J(Θ) = hcat([q̈(Θ, t) for t ∈ 0:Δt:T]...) |> J̃; # J = J(q̈(Θ), 0:Δt:T)
    gen_qdotdot = np.array(
        [qdotdot_gen(task_info, Theta_matrix, t)
         for t in numpy_linspace(0, T, dt)]
    )

    _, gen_q, _, _ = get_traj(gen_qdotdot, robot_arm, dt, init_condit)
    return cost_function(target_pos, gen_q, gen_qdotdot, robot_arm)


def boundcovar(Sigma, lambda_min, lambda_max):
    Q, _ = LA.qr(Sigma)
    eigvals, _ = LA.eig(Sigma)
    eigvals = np.clip(eigvals, lambda_min, lambda_max)
    Sigma = np.matmul(
        Q, 
        np.matmul(
            eigvals*np.eye(len(eigvals)), 
            Q.transpose() 
        )
    )

    return Sigma + 1e-12*np.eye(len(eigvals))
    # https://stackoverflow.com/questions/41515522/numpy-positive-semi-definite-warning


def PIBB(task_info, Theta, Sigma, init_condit, tol=1e-3, max_iter=1000):
    print("######################### PIBB Algorithm Started #########################")
    print(init_condit)
    start_time = time.time()

    lambda_min = task_info.get_lambda_min()
    lambda_max = task_info.get_lambda_max()

    B = task_info.get_B()
    K = task_info.get_K()
    N = task_info.get_N()
    h = task_info.get_h()

    iter_count = 0
    delta_J = eval_rollout(task_info, Theta, init_condit)
    J_hist = [delta_J]

    Thetas = np.zeros(B * K * N).reshape((B, K, N))
    Ps = Js = np.zeros(K)

    while (iter_count < max_iter) and (abs(delta_J) > tol):
        iter_count += 1

#         print("######################### ITER COUNT: {} #########################".format(iter_count))

        # Sigma shape is N, B, B
        # Theta shape is B, N

        for k in range(K):

            Thetas[:, k, :] = np.array(
                [np.random.multivariate_normal(
                    Theta[:, n], Sigma[n, :, :]) for n in range(N)]
            ).transpose()

            Js[k] = eval_rollout(task_info, Thetas[:, k, :], init_condit)

        J_min = np.min(Js)
        J_max = np.max(Js)

        den = sum([np.exp(-h * (Js[l] - J_min) / (J_max - J_min))
                   for l in range(K)])
        for k in range(K):
            Ps[k] = np.exp(-h * (Js[k] - J_min) / (J_max - J_min)) / den

        Sigma = np.zeros(B * B * N).reshape((N, B, B))

        for n in range(N):
            for k in range(K):

                x = Ps[k] * \
                    np.matmul(
                        np.array([(Thetas[:, k, n] - Theta[:, n])]
                                 ).transpose(),
                        np.array([(Thetas[:, k, n] - Theta[:, n])])
                )
                Sigma[n, :, :] += x

            Sigma[n, :, :] = boundcovar(Sigma[n, :, :], lambda_min, lambda_max)

        Theta = np.zeros(B * N).reshape((B, N))

        for k in range(K):
            Theta += (Ps[k] * Thetas[:, k, :])

        J_hist.append(eval_rollout(task_info, Theta, init_condit))

        last_5_J = J_hist[-5:]  # is safe
        # function(works even when no of elements is less than 5)
        delta_J = np.mean(np.diff(last_5_J))

    print("######################### PIBB Algorithm Finished. Time Elapsed : {} #########################".format(
        time.time() - start_time))
    return Theta, iter_count, J_hist


def gen_theta(
    x_target, 
    init_condit, 
    robot_arm, 
    **args
    ):
    """

    """
    lambda_init = 0.05  if args.get('lambda_init')  is None else args.get('lambda_init')
    lambda_min  = 0.05  if args.get('lambda_min')   is None else args.get('lambda_min')
    lambda_max  = 5     if args.get('lambda_max')   is None else args.get('lambda_max')
    dt          = 1e-2  if args.get('dt')           is None else args.get('dt')
    T           = 1     if args.get('T')            is None else args.get('T')
    h           = 10    if args.get('h')            is None else args.get('h')
    B           = 5     if args.get('B')            is None else args.get('B')
    K           = 20    if args.get('K')            is None else args.get('K')

    N, _, _ = robot_arm.get_arm_params()
    # Shape should be B * B * N but we have N * B * B -> indexing has to change accordingly
    # no default action to start with
    Theta_matrix = np.zeros(B*N).reshape((B, N)) if args.get('Theta_matrix') is None else args.get('Theta_matrix')
    assert(isinstance(Theta_matrix, np.ndarray))
    assert(Theta_matrix.shape == (B, N))

    Sigma_matrix = np.array([lambda_init*np.eye(B) for i in range(N)]) if args.get('Sigma_matrix') is None else args.get('Sigma_matrix')
    assert(isinstance(Sigma_matrix, np.ndarray))
    assert(Sigma_matrix.shape == (N, B, B)) 
    

    task_info = TaskInfo(
        robotarm    =   robot_arm,
        lambda_min  =   lambda_min,
        lambda_max  =   lambda_max,
        B           =   B,
        K           =   K,
        N           =   N,
        T           =   T,
        h           =   h,
        target_pos  =   x_target,
        dt          =   dt
    )
    # print(Theta_matrix)
    Theta, iter_count, J_hist = PIBB(
        task_info, 
        Theta_matrix, 
        Sigma_matrix, 
        init_condit
        )
    # print(Theta)
    # plt.plot(range(iter_count+1), J_hist)
    # plt.show()
    return Theta, iter_count, J_hist, task_info

    """
    gen_qdotdot = np.array(  [qdotdot_gen(task_info, Theta, t)
                           for t in numpy_linspace(0, 1, 1e-2)]  )
    time_steps, q, qdot, gen_qdotdot, ani = get_traj_and_simulate2d(
        gen_qdotdot, robot_arm, x_target, init_condit = init_condit, dt = 0.01)
    plt.show()
    from IPython import display
    video = ani.to_jshtml(fps = 60)
    # video = ani.to_html5_video() # to save as mp4, use this
    html = display.HTML(video)
    display.display(html)
    """


def training_data_gen(robot_arm):
    # N, robot_arm_length, link_lengths = robot_arm.get_arm_params()

    # a function that generates a point within the robot_arm_length circle radius
    assert(isinstance(robot_arm, RobotArm))
    x_target = np.array([0.14380348, 0.3495103, 0.11569066])
    init_condit = [np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2]), np.array([0, 0, 0, 0])]

    Theta, _, _, task_info = gen_theta(x_target, init_condit, robot_arm)

    return Theta, task_info
    

if __name__ == "__main__":
    braccio_robot = Braccio3D()
    Theta, task_info = training_data_gen(braccio_robot)

    # TODO: @pranshu you can put the path to the data you need here
    data_csv_path = "/Users/varunsampat/ECE496/intuitive-arm-reach/training_data/init_data_3D.csv"
    pibb_data_df = pd.read_csv(data_csv_path)
    q_ends = pibb_data_df.pop("gen_q")

    # TODO: might need some cleaning
    q_ends = q_ends.to_list()
    q_ends_clean = []
    for q_end in q_ends:
        q_end = q_end[1:-1].strip().split(" ")
        q_end = [q for q in q_end if q != ""]
        q_end = np.array([float(q) for q in q_end])
        q_ends_clean.append(q_end)
    q_ends = q_ends_clean
    print(q_ends_clean)
    
    concat_input, flattened_theta, _, _ = clean_data(pibb_data_df, task_info, planar=False)
    print(pibb_data_df.head())

    # Theta = np.array(
    #     [
    #         [ 8.26676483, -5.30155064,  6.84200035,  3.97223424],
    #         [ 3.21325747,  0.94689616, -0.40225434,  1.46663381], 
    #         [ 5.24084877,  0.46927471, -1.60005178,  1.37439546],
    #         [ 1.26010728,  0.32400262,  1.23270931,  0.68556777],
    #         [ 2.26210504, -2.35412082,  0.91483854, -0.34907162]
    #     ]
    # )
    # gen_qdotdot = np.array(  [qdotdot_gen(task_info, Theta, t)
    #                        for t in numpy_linspace(0, task_info.T, task_info.dt)]  )
    # init_condit = [np.deg2rad([0, 30, 90, 90]), np.array([0, 0, 0, 0])]
    # _, gen_q, _, _ = get_traj(gen_qdotdot, braccio_robot, task_info.dt, init_condit)

    # print(gen_q[-1, :])

    # backend = Swift()   
    # backend.launch()                    # activate it

    # c = sg.Cuboid(
    #     scale=[0.05, 0.05, 0.05],
    #     base=SE3(-0.1315935483132926,	0.0633721128665039,	0.2746948066672217),
    #     color="blue"
    # )

    # cube_id = backend.add(c)

    
    # backend.add(braccio_robot)          # add robot to the 3D scene

    # for qk in gen_q:                    # for each joint configuration on trajectory
    #     braccio_robot.q = qk            # update the robot state
    #     backend.step(dt = 1)            # update visualization
    #     time.sleep(0.2)
    # backend.hold()