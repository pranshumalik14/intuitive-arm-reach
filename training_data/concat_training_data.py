import pandas as pd
from os import listdir
from os.path import isfile, join


def concat_training_data(folder_path):
    onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    
    df = None
    for file_name in onlyfiles:
        if df is None:
            df = pd.read_csv(folder_path + file_name)
        else:
            df = pd.concat([df, pd.read_csv(folder_path + file_name)], ignore_index = True)
    
    df = df[df["init_joint_angles"] != '0']
    df.to_csv(folder_path+'dummy.csv', index = False)

if __name__ == "__main__":
    folder_path = "/Users/varunsampat/ECE496/intuitive-arm-reach/src/robot/20220409_1116/"
    concat_training_data(folder_path)