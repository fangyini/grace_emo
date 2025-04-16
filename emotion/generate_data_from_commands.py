import os
import sys
from fileinput import filename

sys.path.append(os.getcwd())
import random

import math
import time
import threading

import rospy
from hr_msgs.msg import TargetPosture, MotorStateList
from std_msgs.msg import String
from rospy_message_converter import message_converter

import cv2
import csv
import time

from grace.utils import *
from face_motors import *
from video_cap import VideoCapture
import argparse

from signal import signal, SIGINT

def handle_sigint(signalnum, frame):
    print('Key interrupeted')
    sys.exit()

class DataCollectionNode(object):
    def __init__(self, degrees=True, output='images/test/'):
        self.isError = []
        rospy.loginfo('Starting')
        self.output = output
        if not os.path.exists(output):
            os.makedirs(output)
            print(f"Directory created: {output}")
        else:
            print(f"Directory already exists: {output}")

        self.output_img = os.path.join(output, 'img/')
        if not os.path.exists(self.output_img):
            os.makedirs(self.output_img)
        self.output_verify = os.path.join(output, 'labels_verify.csv')

        motors = get_face_motors()
        self.motor_move_info = get_face_motor_move_info(motors)

        self.motors = motors
        self._motor_states = [None]*len(self.motors)
        self.degrees = degrees
        self.motor_lock = threading.Lock()
        
        self.motor_pub = rospy.Publisher('/hr/actuators/pose', TargetPosture, queue_size=1)
        rospy.Subscriber('/hr/actuators/motor_states', MotorStateList, self._capture_state)
        
        self.names = motors
        time.sleep(3.0)
        self.move_to_neutral_state()
        time.sleep(1.0)
        self.take_photo(self.output, 'neutral')
        time.sleep(3.0)

    def motor_int_to_angle(motor, position, degrees):
        if degrees:
            unit = 360
        else:
            unit = math.pi
        angle = ((position-motors_dict[motor]['init'])/4096)*unit
        return angle

    def _capture_state(self, msg):
        self._msg = msg
        for idx, motor_msg in enumerate(msg.motor_states):
            if motor_msg.name in self.names:
                idx = self.motors.index(motor_msg.name)
                self._motor_states[idx] = motor_int_to_angle(motor_msg.name, motor_msg.position, self.degrees)
                if motor_msg.errorCode != 0:
                    with self.motor_lock:
                        print('Motor error: ', motor_msg.name)
                        self.isError.append(1) 


    def take_photo(self, output_path, output_name):
        filename = os.path.join(output_path, output_name + '.png')
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        print(filename)
        try:
            image = cam.read()
            cv2.imwrite(filename, image)
        except: 
            print("Error!! Cannot take the photo. Path: ", filename)
    
    def move_specific(self, names:list, values:list):
        values = [self._check_limits(names[i],x) for i,x in enumerate(values)]
        if self.degrees: # need to convert to radians
            values = [math.radians(x) for x in values]
        args = {"names":names, "values":values}
        self.motor_pub.publish(TargetPosture(**args))

    def move_to_neutral_state(self):
        values = [0] * len(self.motors)
        self.move_specific(self.motors, values)

    def move(self, values):
        self.move_specific(self.names, values)

    def _check_limits(self, name, value):
        if value < self.motor_move_info[name]['min_degree']:
            value = self.motor_move_info[name]['min_degree']
        elif value > self.motor_move_info[name]['max_degree']:
            value = self.motor_move_info[name]['max_degree']
        return value

    def get_faces_from_testing_outcome(self, filepath):
        testing_outcomes = []
        with open(filepath, "r", encoding="utf-8", newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter="\t")
            for row in csv_reader:
                filename, label = row
                label = label[1:-1].split(', ')
                label = [float(i) for i in label]
                assert len(label) == 26
                testing_outcomes.append([filename, label])
        
        rospy.loginfo('Running predicted face')
        sleep_time = 1

        process_frame_index = 0
        deleteLastFrame = False
        with open(self.output_verify, 'a') as csv_writer2:
            writer2 = csv.writer(csv_writer2, delimiter='\t', lineterminator='\n', )
            while process_frame_index < len(testing_outcomes):
                with self.motor_lock:
                    while len(self.isError) > 0: # wait until no error
                        deleteLastFrame = True
                        print(self.isError)
                        self.isError = []
                        print('detected error, start to sleep...')
                        time.sleep(2)

                    if deleteLastFrame == True: # also discard several frames
                        deleteLastFrame = False
                        process_frame_index = max(0, process_frame_index - 2)
                        print('deleted last two frames.. starting from ', process_frame_index)
                        # warn: there might be multiple rows with same title in csv

                    self.isError = [] # clear error
                    print('isError cleared')

                time_str = testing_outcomes[process_frame_index][0]
                prediced_values = testing_outcomes[process_frame_index][1]
                time.sleep(sleep_time)
                self.move(prediced_values)
                time.sleep(sleep_time)
                with self.motor_lock:
                    states = self._motor_states
                self.take_photo(self.output_img, time_str)
                writer2.writerow([time_str, states])
                process_frame_index += 1

        print('Finished!')


if __name__ == '__main__':
    signal(SIGINT, handle_sigint)
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_gau", action='store_true', default=True)
    parser.add_argument("--output", type=str, default='dataset/generated_photos_mead/')
    parser.add_argument("--input_commands", type=str, default='/home/yini/grace_robot/dataset/best_model_lrelu/testing_results.csv')
    args = parser.parse_args()
    print(args)

    t1 = time.time()
    global cam
    cam = VideoCapture(-1)
    rospy.init_node('DataCollectionNode')
    node = DataCollectionNode(output=args.output)
    node.get_faces_from_testing_outcome(args.input_commands)
    t2 = time.time()
    print('time elapse=', str(int((t2-t1))), 'sec')
    rospy.signal_shutdown('End')
    sys.exit()
