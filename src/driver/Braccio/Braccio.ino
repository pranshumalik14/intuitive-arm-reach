 /*
  testBraccio90.ino

 testBraccio90 is a setup sketch to check the alignment of all the servo motors
 This is the first sketch you need to run on Braccio
 When you start this sketch Braccio will be positioned perpendicular to the base
 If you can't see the Braccio in this exact position you need to reallign the servo motors position

 Created on 18 Nov 2015
 by Andrea Martino

 This example is in the public domain.
 */

#include <Braccio.h>
#include <Servo.h>

#define OFFSET_M1 185
#define OFFSET_M2 10
#define OFFSET_M3 -4
#define OFFSET_M4 5
#define OFFSET_M5 -10

Servo base;
Servo shoulder;
Servo elbow;
Servo wrist_rot;
Servo wrist_ver;
Servo gripper;

void setup() {  
  //Initialization functions and set up the initial position for Braccio
  
  //All the servo motors will be positioned in the "safety" position:
  
  //Base (M1):90 degrees
  //Shoulder (M2): 45 degrees
  //Elbow (M3): 180 degrees
  //Wrist vertical (M4): 180 degrees
  //Wrist rotation (M5): 90 degrees
  //gripper (M6): 10 degrees
  Serial.begin(9600) ;
  Braccio.begin(); 
}

int q1 = 0;
int q2 = 15;
int q3 = 90;
int q4 = 90; 
int q5 = 90;
int q6 = 10;

void loop() {
  /*
   Step Delay: a milliseconds delay between the movement of each servo.  Allowed values from 10 to 30 msec.
   M1=base degrees. Allowed values from 0 to 180 degrees
   M2=shoulder degrees. Allowed values from 15 to 165 degrees
   M3=elbow degrees. Allowed values from 0 to 180 degrees
   M4=wrist vertical degrees. Allowed values from 0 to 180 degrees
   M5=wrist rotation degrees. Allowed values from 0 to 180 degrees
   M6=gripper degrees. Allowed values from 10 to 73 degrees. 10: the toungue is open, 73: the gripper is closed.
  */
  
  // the arm is aligned upwards  and the gripper is closed
  // (step delay, M1, M2, M3, M4, M5, M6);
  
    if(Serial.available()) {
      q1 = Serial.read();
      q2 = Serial.read();
      q3 = Serial.read();
      q4 = Serial.read();
      q5 = Serial.read();
      q6 = Serial.read();
    }

//    Serial.println("q1 :\t" + String(OFFSET_M1 - q1));
//    Serial.println("q2 :\t" + String(q2 + OFFSET_M2));
//    Serial.println("q3 :\t" + String(q3 + OFFSET_M3));
//    Serial.println("q4 :\t" + String(q4 + OFFSET_M4));
//    Serial.println("q5 :\t" + String(q5 + OFFSET_M5));
//    Serial.println("q6 :\t" + String(q6));
    
    Braccio.ServoMovement(20, OFFSET_M1 - q1, q2 + OFFSET_M2, q3 + OFFSET_M3, q4 + OFFSET_M4, q5 + OFFSET_M5, q6);  
    delay(1000);
}
