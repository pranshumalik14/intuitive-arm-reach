import serial
import struct
import time
import numpy as np

def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()


def connect_arduino():
    arduino = serial.Serial(
        port = "/dev/cu.usbmodem14301", 
        baudrate = 9600, 
        timeout = 1)
    return arduino

def disconnect_arduino(arduino_cnx):
    arduino_cnx.close()
    
def set_joint_angle(joint_angles, arduino_cnx):
    """
    numpy array of joint angles
    """
    assert(len(joint_angles) == 6)
    arduino_cnx.write(struct.pack('>B', (joint_angles[0])))
    arduino_cnx.write(struct.pack('>B', (joint_angles[1])))
    arduino_cnx.write(struct.pack('>B', (joint_angles[2])))
    arduino_cnx.write(struct.pack('>B', (joint_angles[3])))
    arduino_cnx.write(struct.pack('>B', (joint_angles[4])))
    arduino_cnx.write(struct.pack('>B', (joint_angles[5])))
    time.sleep(1)
    
arduino_cnx = None

while True:  # making a loop
    try:  # used try so that if user pressed other than the given key error will not be shown
        keypress = input()
        if keypress == 'c':  # if key 'q' is pressed 
            print('CALIBRATING')
            arduino_cnx = connect_arduino()
            print(arduino_cnx)
            # A List of Items
            items = list(range(0, 120))

            # A Nicer, Single-Call Usage
            for item in progressBar(items, prefix = 'Progress:', suffix = 'Complete', length = 50):
                # Do stuff...
                time.sleep(0.1)
            print("################ ARDUINO CONNECTION: SUCCESSFUL ################")

        elif keypress == 'a':
            if arduino_cnx is None:
                print("################ CANNOT ACCEPT JOINT ANGLES W/OUT ARDUINO CNX ################")
                print("################ Press 'c' first to calibrate ################")
            else:
                print('ACCEPTING ANGLE VALS')
                """
                M1  =   base degrees. Allowed values from 0 to 180 degrees
                M2  =   shoulder degrees. Allowed values from 15 to 165 degrees
                M3  =   elbow degrees. Allowed values from 0 to 180 degrees
                M4  =   wrist vertical degrees. Allowed values from 0 to 180 degrees
                M5  =   wrist rotation degrees. Allowed values from 0 to 180 degrees
                M6  =   gripper degrees. Allowed values from 10 to 73 degrees. 10: the toungue is open, 73: the gripper is closed.
                """
                joint_angles = input("Enter 6 joint angles in degrees separated by comma (e.g: 0, 0, 0, 0, 0, 0)\n")
                joint_angles = joint_angles.strip().split(',')
                try:
                    joint_angle_ints = [int(r) for r in joint_angles]
                    assert(len(joint_angle_ints) == 6)
                    set_joint_angle(joint_angle_ints, arduino_cnx)
                    print("################ REACH TASK: SUCCESSFUL ################")
                except ValueError as ex:
                    print(str(ex))
                    print("Non integer input invalid")
                except AssertionError as ex:
                    print(str(ex))
                except Exception as ex:
                    print(str(ex))
        else:
            print('invalid key')
    except KeyboardInterrupt:
        kill = input("Kill Program (y\\n)\n").lower()
        if kill == "y":
            disconnect_arduino(arduino_cnx)
            break
        else:
            print("Not killing")
    except Exception as ex:
        print("EXCEPTION: " + str(ex))
        break  # if user pressed a key other than the given key the loop will break