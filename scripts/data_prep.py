import pandas as pd
import numpy as np
import platform
from scripts.robot_arm import RobotArm, RobotArm2D, RobotArm3D
from scripts.task_info import TaskInfo, numpy_linspace

HOLDOUT = None

def load_data(data_csv, task_info_csv):
    pibb_data_df = pd.read_csv(data_csv)
    task_info_df = pd.read_csv(task_info_csv)
    return pibb_data_df, task_info_df


def task_info_from_df(df):
    return TaskInfo(
        robotarm = None, #TODO need to fix this
        lambda_min = df["lambda_min"].values[0],
        lambda_max = df["lambda_max"].values[0],
        B = df["B"].values[0],
        K = df["K"].values[0],
        N = df["N"].values[0],
        T = df["T"].values[0],
        h = df["h"].values[0],  # eliteness param
        dt = df["dt"].values[0],
        target_pos = df["target_pos"],
        w = df["w"].values[0],
        cs = df["cs"]
    )


def robot2D_from_df(df):
    return RobotArm2D(df["n_dims"], df["link_lenghts"])


def str_to_floats(angle_str):
    try:
        angle_floats = angle_str[1:len(angle_str)-1].strip().split(" ")
        angle_floats = [angle for angle in angle_floats if angle != ""]
        angle_floats = [float(angle) for angle in angle_floats]
    except Exception as ex:
        print(angle_str, angle_str[0], type(angle_str[0]))
    return angle_floats


def clean_task_info(task_info, task_info_df):
    # clean task info
    target_pos = task_info_df.pop('target_pos')
    target_pos_floats = []
    for angles in target_pos:
        target_pos_floats.append(str_to_floats(angles))
    target_pos_np = np.array(target_pos_floats)

    cs = task_info_df.pop('cs')
    cs_floats = []
    for val in cs:
        cs_floats.append(str_to_floats(val))
    cs_vals_np = np.array(cs_floats)

    task_info.target_pos = target_pos_np[0]
    task_info.cs = cs_vals_np[0]
    return task_info


def skip_circles(pibb_data_df, skip=3):
    holdout = []
    y_s = pibb_data_df["y_target"]
    starts = [i for i, val in enumerate(y_s) if val == 0]
    print("Number of circles = " + str(len(starts)))

    i = 0
    circles = []
    skipped = 0

    while (i < len(starts)-1):
        start = starts[i]
        end = starts[i+1]
        # print(start, end)
        i += 2
        if skipped == 3:
            circles.append((start, end))
            skipped = 0
        else:
            holdout.append((start, end))
            skipped += 1

    keep = []
    for circle in circles:
        keep.append(pibb_data_df[circle[0]:circle[1]])
    skipped_circles_df = pd.concat(keep, ignore_index=True)
    return skipped_circles_df


def skip_k_every_n(pibb_data_df, k=100, n=5):
    holdout = []
    records = len(pibb_data_df)
    i = 0
    count = 0
    keep = []
    while (i < records):
        if count == n:
            skip = list(range(k))
            skip = [i+s for s in skip]
            holdout.extend(skip)
            i += k
            count = 0
        else:
            keep.append(pibb_data_df[i:i+n])
            i += n
            count += n
    skipped_df = pd.concat(keep, ignore_index=True)
    return skipped_df


def by_init_condition(pibb_data_df, skip_factor):
    # uniformly sample points for each of the 10 joint configs
    holdout = []

    num_init_conditions = len(pibb_data_df["init_joint_angles"].drop_duplicates(inplace=False).to_numpy())
    print(num_init_conditions)
    num_points = len(pibb_data_df.where(pibb_data_df["init_joint_angles"] == pibb_data_df["init_joint_angles"][0]).dropna(how="all"))
    print(num_points)
    
    keep = []
    for c in range(num_init_conditions):
        start = c*num_points
        end = (c+1)*num_points
        count = 0
        for i in range(start, end):
            if count%skip_factor == 0:
                keep.append(pibb_data_df[i:i+1])
            else:
                holdout.append(i)
            count += 1
    print(len(keep))
    skipped_df = pd.concat(keep, ignore_index=True)
    return skipped_df, holdout


def holdout_data(pibb_data_df, skip_factor, strat="SKIP"):
    original_df = pibb_data_df
    if strat == "SKIP":
        skipped_df, holdout = by_init_condition(pibb_data_df, skip_factor)
        return skipped_df, holdout, original_df
    raise Exception("SKIP strat only currently accepted")


def clean_data(pibb_data_df, task_info, skip_factor, planar=True):
    pibb_data_df = pibb_data_df.where(pibb_data_df["init_joint_angles"] != "0").dropna(how="all")

    if not planar:
        pibb_data_df = pibb_data_df.drop_duplicates(subset=["init_joint_angles", "x_target", "y_target", "z_target"], ignore_index=True)
    else:
        pibb_data_df = pibb_data_df.drop_duplicates(subset=["init_joint_angles", "x_target", "y_target"], ignore_index=True)

    original_df = None
    holdout = None
    if HOLDOUT is not None:
        pibb_data_df, holdout, original_df = holdout_data(pibb_data_df, skip_factor)

    delimiter = "\r\n " if platform.system() == "Windows" else "\n "

    # reshape x target
    x_target = pibb_data_df['x_target']
    x_target_np = x_target.to_numpy()
    x_target = np.reshape(x_target_np, (x_target_np.shape[0], 1))

    # rehsape y target
    y_target = pibb_data_df['y_target']
    y_target_np = y_target.to_numpy()
    y_target = np.reshape(y_target_np, (y_target_np.shape[0], 1))

    if not planar:
        # rehsape z target
        z_target = pibb_data_df['z_target']
        z_target_np = z_target.to_numpy()
        z_target = np.reshape(z_target_np, (z_target_np.shape[0], 1))

    # reshape joint angles
    joint_angles = pibb_data_df['init_joint_angles']
    joint_angle_floats = []
    for angles in joint_angles:
        joint_angle_floats.append(str_to_floats(angles))
    joint_angle_np = np.array(joint_angle_floats)
    joint_angles = joint_angle_np

    # reshape theta
    theta = pibb_data_df['Theta']
    temp_theta = np.zeros(shape=(len(pibb_data_df), task_info.B, task_info.N)) #670 for each element, 2 for x & y, 5 for gaussian basis functions
    for t in range(0, len(temp_theta)):
        if theta[t].startswith("[") and theta[t].endswith("]"):
            temp = theta[t][1:-1]
        else:
            print("Error")
        tp = temp.split(delimiter)
        for i in range(0, len(tp)):
            temp_array = str_to_floats(tp[i])
            temp_theta[t][i] = temp_array
    
    theta = temp_theta

    if not planar:
        print("Input sizes are: \n", "joint angles: ", str(joint_angles.shape), "\nx_target: ",
        str(x_target.shape), "\ny_target: ", str(y_target.shape), "\nz_target: ", str(z_target.shape))
        print("Output size is:\n theta: ", str(theta.shape))
    else:
        print("Input sizes are: \n", "joint angles: ", str(joint_angles.shape), "\nx_target: ",
        str(x_target.shape), "\ny_target: ", str(y_target.shape))
        print("Output size is:\n theta: ", str(theta.shape))

    # Concatenate all input features into a single matrix
    if not planar:
        concat_input = np.concatenate([joint_angles, x_target, y_target, z_target], axis=1)
    else:
        concat_input = np.concatenate([joint_angles, x_target, y_target], axis=1)
    flatten_theta = np.zeros(shape=(theta.shape[0], theta.shape[1]*theta.shape[2]))
    for i in range(0, len(flatten_theta)):
        flatten_theta[i] = theta[i].flatten()

    return concat_input, flatten_theta, holdout, original_df

