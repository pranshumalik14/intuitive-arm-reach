#include <Braccio.h>
#include <Servo.h>
#include <ArduinoJson.h>

#define OFFSET_M1 185
#define OFFSET_M2 10
#define OFFSET_M3 -4
#define OFFSET_M4 5
#define OFFSET_M5 -10

#define NO_JOINT_ANGLES 6

#define Q1_CONST_START 0
#define Q1_CONST_END 180

#define Q2_CONST_START 15
#define Q2_CONST_END 165

#define Q3_CONST_START 0
#define Q3_CONST_END 180

#define Q4_CONST_START 0
#define Q4_CONST_END 180

#define Q5_CONST_START 0
#define Q5_CONST_END 180

#define Q6_CONST_START 10
#define Q6_CONST_END 73

#define ST_WRITE_ARDUINO 0
#define ST_READ_ARDUINO 1
#define ST_GO_HOME 2
#define ST_RESET_STATUS 3
#define ST_CAMERA_CALIB 4

#define DELAY 10

Servo base;
Servo shoulder;
Servo elbow;
Servo wrist_rot;
Servo wrist_ver;
Servo gripper;

void setup()
{
  // Initialization functions and set up the initial position for Braccio

  // All the servo motors will be positioned in the "safety" position:

  // Base (M1):90 degrees
  // Shoulder (M2): 45 degrees
  // Elbow (M3): 180 degrees
  // Wrist vertical (M4): 180 degrees
  // Wrist rotation (M5): 90 degrees
  // gripper (M6): 10 degrees
  Serial.begin(9600);
  Braccio.begin();
}

int state = 0; // 0 -> set angles, 1 -> read angles
int q1 = 0;
int q2 = 30;
int q3 = 90;
int q4 = 90;
int q5 = 90;
int q6 = 10;

bool areJointConsMet = true;

void loop()
{
  /*
   Step Delay: a milliseconds delay between the movement of each servo.  Allowed values from 10 to 30 msec.
   M1=base degrees. Allowed values from 0 to 180 degrees
   M2=shoulder degrees. Allowed values from 15 to 165 degrees
   M3=elbow degrees. Allowed values from 0 to 180 degrees
   M4=wrist vertical degrees. Allowed values from 0 to 180 degrees
   M5=wrist rotation degrees. Allowed values from 0 to 180 degrees
   M6=gripper degrees. Allowed values from 10 to 73 degrees. 10: the tongue is open, 73: the gripper is closed.
  */

  // the arm is aligned upwards  and the gripper is closed
  // (step delay, M1, M2, M3, M4, M5, M6);

  if (Serial.available())
  {
    state = Serial.read();

    if (state == ST_WRITE_ARDUINO)
    {
      int q1_temp = Serial.read();
      int q2_temp = Serial.read();
      int q3_temp = Serial.read();
      int q4_temp = Serial.read();
      int q5_temp = Serial.read();
      int q6_temp = Serial.read();

      if (
          (q1_temp >= Q1_CONST_START && q1_temp <= Q1_CONST_END) &&
          (q2_temp >= Q2_CONST_START && q2_temp <= Q2_CONST_END) &&
          (q3_temp >= Q3_CONST_START && q3_temp <= Q3_CONST_END) &&
          (q4_temp >= Q4_CONST_START && q4_temp <= Q4_CONST_END) &&
          (q5_temp >= Q5_CONST_START && q5_temp <= Q5_CONST_END) &&
          (q6_temp >= Q6_CONST_START && q6_temp <= Q6_CONST_END))
      {
        areJointConsMet = true;
        q1 = q1_temp;
        q2 = q2_temp;
        q3 = q3_temp;
        q4 = q4_temp;
        q5 = q5_temp;
        q6 = q6_temp;
      }
      else
      {
        areJointConsMet = false;
      }
    }
    else if (state == ST_READ_ARDUINO)
    {
      StaticJsonDocument<1024> doc;
      doc["joint_angles"][0] = q1;
      doc["joint_angles"][1] = q2;
      doc["joint_angles"][2] = q3;
      doc["joint_angles"][3] = q4;
      doc["joint_angles"][4] = q5;
      doc["joint_angles"][5] = q6;

      doc["JointConstraintStatus"] = (areJointConsMet) ? "OK" : "NOT OK";
      serializeJson(doc, Serial);
    }
    else if (state == ST_CAMERA_CALIB)
    {
      q1 = 0;
      q2 = 30;
      q3 = 90;
      q4 = 90;
      q5 = 90;
      q6 = 10;
    }
    else if (state == ST_RESET_STATUS)
    {
      areJointConsMet = true;
    }
    else if (state == ST_GO_HOME)
    {
      q1 = 90;
      q2 = 90;
      q3 = 90;
      q4 = 90;
      q5 = 90;
      q6 = 10;
    }

    state = 0; // always wait for serial
  }

  int actual_q1 = OFFSET_M1 - q1;
  int actual_q2 = q2 + OFFSET_M2;
  int actual_q3 = q3 + OFFSET_M3;
  int actual_q4 = q4 + OFFSET_M4;
  int actual_q5 = q5 + OFFSET_M5;
  int actual_q6 = q6;

  if (areJointConsMet)
  {
    Braccio.ServoMovement(20, actual_q1, actual_q2, actual_q3, actual_q4, actual_q5, actual_q6);
  }

  delay(DELAY);
}
