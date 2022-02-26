from sre_constants import FAILURE, SUCCESS
import numpy as np
from robot_driver import BraccioRobotDriver, JOINT_DESCRIPTORS

SUCCESS = 0
FAILURE = 1

HELP_MENU = {
    "Help Menu:": "     Welcome to ECE496 PROJ353. This is the terminal software to control the Braccio Robot. The key commands to control the robot are as follows:",
    "\tPress c:": "     To connect to the Arduino (Physical Robot)",
    "\tPress a:": "     To control the joints of the robot",
    "\tPress h:": "     To go home and calibrate the camera",
    "\tPress i:": "     To print the help menu",
    "\tPress r:": "     To print the current joint angles of the robot"
}


# todo: fix float to int conversion; see bug on messenger!!!

def print_help_menu():
    for key, value in HELP_MENU.items():
        print(key + value)

# https://stackoverflow.com/a/34325723


def progressBar(iterable, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
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

    def printProgressBar(iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                         (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()


def parse_joint_angles(driver: BraccioRobotDriver, verbose=True):
    """
    read joint angles passed into terminal
    """
    print('Please provide joint angle values: ')
    for key, value in JOINT_DESCRIPTORS.items():
        print("\t" + key + ":\t" + value)

    joint_angles = input(
        "Enter up to 6 joint angles in degrees separated by comma (e.g: 0, 0, 0, 0, 0, 0)\n \
        Use * for unchanged angles\n"
    )
    joint_angles = joint_angles.strip().split(',')
    joint_angles = [angle.strip() for angle in joint_angles]
    if len(joint_angles) != 6:
        print("Not enough values provided, try again")
        return parse_joint_angles(driver)

    if "*" in joint_angles:
        existing_values = driver.read()["joint_angles"]
        for i, angle in enumerate(joint_angles):
            if angle == "*":
                joint_angles[i] = existing_values[i]

    if verbose:
        assert(len(joint_angles) == len(JOINT_DESCRIPTORS))
        for i, key in enumerate(JOINT_DESCRIPTORS):
            print("Setting " + key + " to " + str(joint_angles[i]).strip())

    if input("Continue (y/n)?\n\t") != "y":
        print("Trying again ...")
        return parse_joint_angles(driver)

    return joint_angles


def accept_joint_angles(driver: BraccioRobotDriver):
    """
    read joint angles from the user and send them to the arduino connection
    """
    joint_angles = parse_joint_angles(driver)
    try:
        joint_angle_ints = [int(r) for r in joint_angles]
        assert(len(joint_angle_ints) == 6)
        driver.set_joint_angles(joint_angle_ints)
        status = SUCCESS if driver.read(
        )["JointConstraintStatus"] == "OK" else FAILURE
        if status == FAILURE:
            driver.reset_joint_constraint_status()
        return status
    except ValueError as ex:
        print(str(ex))
        print("Non integer input invalid")
    except AssertionError as ex:
        print(str(ex))
    except Exception as ex:
        print(str(ex))


if __name__ == "__main__":
    braccio_driver = BraccioRobotDriver(
        loop_rate=1
    )

    print_help_menu()
    while True:  # making a loop
        try:  # used try so that if user pressed other than the given key error will not be shown
            keypress = input()
            if keypress == 'c':  # if key 'c' is pressed
                print("################     CALIBRATING     ################")
                port = input("Enter port number: ")
                braccio_driver.calibrate(port)
                print("################ SUCCESSFULLY CONNECTED ################")

            elif keypress == 'a':
                if not braccio_driver.is_connection_valid():
                    print(
                        "################ CANNOT ACCEPT JOINT ANGLES W/OUT ARDUINO CNX ################")
                    print(
                        "################ Press 'c' first to calibrate ################")
                else:
                    change_status = accept_joint_angles(braccio_driver)
                    if change_status == SUCCESS:
                        print(
                            "################         REACH TASK: SUCCESSFUL      ################")
                    elif change_status == FAILURE:
                        print(
                            "################ REACH TASK: FAILED JOINT CONSTRAINTS ################")

            elif keypress == 'r':
                print("################ READING FROM PORT ################")
                str_returned = braccio_driver.read()
                print(str_returned)
            elif keypress == 'h':
                print("################ GOING HOME ################")
                braccio_driver.homecoming()
            elif keypress == 'i':
                print_help_menu()
            else:
                print('invalid key. Press i for more info')
        except KeyboardInterrupt:
            kill = input("Kill Program (y\\n)\n").lower()
            if kill == "y":
                braccio_driver.disconnect()
                break
            else:
                print("Not killing")
        except Exception as ex:
            print("EXCEPTION: " + str(ex))
            break  # if user pressed a key other than the given key the loop will break
