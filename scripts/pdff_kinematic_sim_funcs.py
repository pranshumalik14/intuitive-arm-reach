import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from collections import deque

class RobotArm2D:
    def __init__(self, n_dims = 2, arm_length = 0.8, rel_link_lengths = np.array([2, 0.6, 0.4, 0.15, 0.15, 0.1])):
        self.n_dims = n_dims
        self.arm_length = arm_length
        self.link_lengths = arm_length/sum(rel_link_lengths) * rel_link_lengths[0:n_dims+1]
        # np.array([1.5, 1.25, 0.4, 0.15, 0.15, 0.1])

    def get_arm_params(self):
        return self.n_dims, self.arm_length, self.link_lengths



def get_traj(qdotdot, robot_arm, dt = 0.05, init_condit = [None, None]):
    """
    Takes in a joint accelerations qdotdot, and return 
    
    Keyword arguments:
    argument -- qdotdot: an t x n_dim matrix (t timesteps, n_dim dimensions) (numpy array)
    Return: return_description
    """
    
    n_time_steps = qdotdot.shape[0]
    n_dims_qdotdot = qdotdot.shape[1] if qdotdot.ndim > 1 else 1

    n_dims, arm_length, link_lengths = robot_arm.get_arm_params()
    assert(n_dims == n_dims_qdotdot)

    time_steps = dt * np.arange(n_time_steps)
    
    # initialize q, qdot to 0
    q = np.zeros((n_time_steps, n_dims))
    qdot = np.zeros((n_time_steps, n_dims))
    
    if init_condit is not None:
        q0, qdot0 = init_condit[0], init_condit[1]
        assert(qdot.shape[1] == len(qdot0))
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


def angles_to_link_positions_2D(q, robot_arm):
    # Forward kinematics
    n_time_steps = q.shape[0]
    n_dims = q.shape[1]
    
    n_dims_robot, arm_length, link_lengths = robot_arm.get_arm_params()
    assert(n_dims == n_dims_robot)
    
    links_x = np.zeros((n_time_steps, n_dims+1))
    links_y = np.zeros((n_time_steps, n_dims+1))
    
    for t in range(n_time_steps):
        sum_angles = 0
        
        for i_dim in range(n_dims):
            sum_angles += q[t, i_dim]
            links_x[t, i_dim + 1] = links_x[t, i_dim] + np.cos(sum_angles) * link_lengths[i_dim]
            links_y[t, i_dim + 1] = links_y[t, i_dim] + np.sin(sum_angles) * link_lengths[i_dim]
            
    link_positions = np.zeros((n_time_steps, 2*(n_dims+1)))
    # desired structure of link positions is x y x y x y 
    # (first x y are for the base joint)
    for n in range(n_dims + 1):
        link_positions[:, 2*n] = links_x[:,n]
        link_positions[:, 2*n+1] = links_y[:,n]
    
    return link_positions


def get_traj_and_simulate(qdotdot, robot_arm, x_goal, init_condit = [None, None], dt = 0.005):
    """
    
    """
    # initial checks
    assert(len(x_goal) == 2)
    
    # Get q and qdot
    time_steps, q, qdot, qdotdot = get_traj( qdotdot, robot_arm, init_condit = init_condit, dt = dt)
    
    # Forward kinematics
    link_positions = angles_to_link_positions_2D(q, robot_arm)
    
    # Robot params
    n_dims, arm_length, link_lengths = robot_arm.get_arm_params()
    
    # Figure set up
    fig = plt.figure()
    padding_factor = 1.5
    axes_lim = arm_length*padding_factor
    
    ax = fig.add_subplot(autoscale_on=False, xlim=(-axes_lim, axes_lim), ylim=(-axes_lim, axes_lim))
    ax.set_aspect('equal')
    ax.grid()
    
    # tracking the history of movements
    tracking_history_points = 1000
    history_x, history_y = deque(maxlen=tracking_history_points), deque(maxlen=tracking_history_points)

    # Adding the base of the robot arm
    points = [[0, 0], [-0.05, -0.1], [0.05, -0.1]]
    line = plt.Polygon(points, closed = True, fill = True, color = 'red')
    plt.gca().add_patch(line)

    # Dynamic lines (theese are the lines/vals that will update during the simulation)     
    line, = ax.plot([], [], 'o-', lw=2)
    trace, = ax.plot([], [], '.-', lw=1, ms=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    
    # Annotations for each link/end effector
    annotate_links = []
    for n in range(n_dims):
        link_to_annotate = "q{}".format(str(1+n))
        annotate_links.append(ax.annotate(link_to_annotate, xy = (link_lengths[n], 0)))
    
    annotate_end_effector = ax.annotate('E', xy = (link_lengths[-1] , 0))

    # animation for each frame
    def animate(i):
        # all x axis values are in the even-numbered columns
        thisx = [link_positions[i, j] for j in range(0, len(link_positions[0,:]), 2)]
        
        # all y axis value are in the odd-numbered columns
        thisy = [link_positions[i, j] for j in range(1, len(link_positions[0,:]), 2)]

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
        for n in range(n_dims):
            annotate_links[n].set_position(
                (
                link_positions[i, 2*n] + offset_factor*link_lengths[n]*np.cos(q[i,n]),
                link_positions[i, 2*n+1] + offset_factor*link_lengths[n]*np.sin(q[i,n])
                )
            )

        annotate_end_effector.set_position((
            link_positions[i, len(link_positions[0,:])-2],
            link_positions[i, -1]
        ))

        return line, trace, time_text

    ani = animation.FuncAnimation(
        fig, animate, n_time_steps, interval = dt* n_time_steps, blit= True)
    
    # Goal position
    plt.plot(x_goal[0], x_goal[1], '-o') # Goal position
    ax.annotate('x_g',xy = (x_goal[0], x_goal[1]))

    fig.suptitle('Kinematic Simulation', fontsize=14)
    
    return time_steps, q, qdot, qdotdot, ani
    