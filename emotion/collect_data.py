import os
import sys
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
    def __init__(self, isGau, degrees=True, output='images/test/'):
        self.isGau = isGau
        print('isGau=', isGau)
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
        self.output_label = os.path.join(output, 'labels.csv')
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
        self.isError = []
        self.move_to_neutral_state()
        time.sleep(3.0)

    def _capture_state(self, msg):
        self._msg = msg
        for idx, motor_msg in enumerate(msg.motor_states):
            # todo save motor state for checking
            '''if motor_msg.name in self.names:
                    idx = self.motors.index(motor_msg.name)
                    self._motor_states[idx] = motor_int_to_angle(motor_msg.name, motor_msg.position, self.degrees)'''

            if motor_msg.name in self.names:
                if motor_msg.errorCode != 0:
                    with self.motor_lock:
                        print('Motor error: ', motor_msg.name)
                        self.isError.append(1) 
                        

    def generate_random_face(self, times=1):
        try:
            with open(self.output_label, "r", encoding="utf-8", errors="ignore") as reader:
                last_row = reader.readlines()[-1]
                starting_num = int(last_row.split('\t')[0]) + 1
        except:
            starting_num = 0
        print('starting num', str(starting_num))

        rospy.loginfo('Running random face')
        sleep_time = 1
        img_num_len = 5
        with open(self.output_label,'a') as csv_writer:
            writer=csv.writer(csv_writer, delimiter='\t',lineterminator='\n',)
            process_frame_index = starting_num
            deleteLastFrame = False
            while process_frame_index < times:
                with self.motor_lock:
                    while len(self.isError) > 0: # wait until no error
                        deleteLastFrame = True
                        print(self.isError)
                        self.isError = []
                        print('detected error, start to sleep...')
                        time.sleep(3)
                    
                    if deleteLastFrame == True: # also discard several frames
                        deleteLastFrame = False
                        process_frame_index = max(0, process_frame_index - 2)
                        print('deleted last two frames.. starting from ', process_frame_index)
                        # warn: there might be multiple rows with same title in csv

                    self.isError = [] # clear error 
                    print('isError cleared')

                time_str = '0' * (img_num_len - len(str(process_frame_index))) + str(process_frame_index)
                random_values = self.get_random_movement(gaussian=self.isGau)
                time.sleep(sleep_time)
                self.move(random_values)
                time.sleep(sleep_time)
                self.take_photo(self.output_img, time_str)

                row = [time_str, random_values]
                writer.writerow(row)
                process_frame_index += 1

        print('finised generating ', str(times-starting_num), ' frames')

    def take_photo(self, output_path, output_name):
        filename = output_path + 'img_' + output_name + '.png'
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
    
    def get_random_movement(self, names=None, gaussian=False):
        if names is None:
            names = self.motors
        values = []
        for i in names:
            min_d, max_d = math.ceil(self.motor_move_info[i]['min_degree']), math.floor(self.motor_move_info[i]['max_degree'])
            if gaussian is True:
                sampled_num = np.random.normal() # around (-3, 3)
                scale = max(max_d, abs(min_d)) / 3
                random_value = sampled_num * scale
                if max_d == 0 and random_value > 0:
                    random_value = -1 * random_value
                elif min_d == 0 and random_value < 0:
                    random_value = -1 * random_value
                # it's okay if exceed the limit, checked at a later point
                # print for debug
                '''print('min d=', min_d, ',max d=', max_d)
                print('sampled num=', sampled_num)
                print('scale=', scale)
                print('random value=', random_value)
                print('\n')'''
            else:
                random_value = random.randint(min_d, max_d)
            values.append(random_value)
        return values


if __name__ == '__main__':
    signal(SIGINT, handle_sigint)
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_gau", action='store_true', default=True)
    parser.add_argument("--times", type=int, default=100)
    parser.add_argument("--output", type=str, default='dataset/gau/')
    args = parser.parse_args()
    print(args)

    t1 = time.time()
    global cam
    cam = VideoCapture(-1)
    rospy.init_node('DataCollectionNode')
    node = DataCollectionNode(output=args.output, isGau=args.is_gau)
    node.generate_random_face(times=args.times)
    t2 = time.time()
    print('time elapse=', str(int((t2-t1))), 'sec')
    rospy.signal_shutdown('End')
    sys.exit()
    # TODO: prove: output number=gaussian distribution, motor numbers match with physical robot face,
