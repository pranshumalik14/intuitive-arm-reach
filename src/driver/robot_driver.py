import struct
import time
import json
import serial

ST_WRITE_ARDUINO    = 0
ST_READ_ARDUINO     = 1
ST_GO_HOME          = 2
ST_RESET_STATUS     = 3
ST_CAMERA_CALIB     = 4

JOINT_DESCRIPTORS = {
    "M1": "Base degrees:            Allowed values from 0 to 180 degrees",
    "M2": "Shoulder degrees:        Allowed values from 15 to 165 degrees",
    "M3": "Elbow degrees:           Allowed values from 0 to 180 degrees",
    "M4": "Wrist vertical degrees:  Allowed values from 0 to 180 degrees",
    "M5": "Wrist rotation degrees:  Allowed values from 0 to 180 degrees",
    "M6": "Gripper degrees:         Allowed values from 10 to 73 degrees [10: the toungue is open, 73: the gripper is closed]"
}

class BraccioRobotDriver:
    def __init__(self, **args) -> None:
        self.cnx = None
        self.port = args.get("port")
        self.curr_joint_angles = None
        self.target_location = None
        self.loop_rate = args.get("loop_rate")

    def write(self, insts):
        if isinstance(insts, list):
            for inst in insts:
                self.cnx.write(inst)
        else:
            # TODO: check if isinstance() of whatever struct returns
            self.cnx.write(insts)
        time.sleep(self.loop_rate)

    def connect(self, port):
        self.cnx = serial.Serial(
            port = "/dev/cu.usbmodem{}".format(port), 
            baudrate = 9600, 
            timeout = 1)
    
    def disconnect(self):
        self.cnx.close()
        self.cnx = None

    def calibrate(self, port=None):
        if port:    
            self.connect(port)
        else:
            self.connect(self.port)
        time.sleep(12)

    def is_connection_valid(self):
        try:
            return self is not None
        except Exception:
            return False
    
    def set_joint_angles(self, joint_angles):
        """
        numpy array of joint angles
        """
        assert(len(joint_angles) == 6)
        print("Attempting to do the motion: " + str(joint_angles))
        insts = [
            struct.pack('>B', ST_WRITE_ARDUINO),
            struct.pack('>B', (joint_angles[0])),
            struct.pack('>B', (joint_angles[1])),
            struct.pack('>B', (joint_angles[2])),
            struct.pack('>B', (joint_angles[3])),
            struct.pack('>B', (joint_angles[4])),
            struct.pack('>B', (joint_angles[5])),
             ]
        self.write(insts)

    def is_joint_angle_valid(self):
        pass
    
    def homecoming(self):
        self.write(struct.pack('>B', ST_GO_HOME))

    def vision_calib_pose(self):
        self.write(struct.pack('>B', ST_CAMERA_CALIB))

    def read(self):
        self.cnx.reset_input_buffer()
        self.write(struct.pack('>B', ST_READ_ARDUINO))
        to_return = json.loads(self.cnx.read(1024).decode('UTF-8'))
        return to_return

    def reset_joint_constraint_status(self):
        self.write(struct.pack('>B', ST_RESET_STATUS))



