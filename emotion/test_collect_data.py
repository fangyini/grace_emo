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

from grace.utils import *
from face_motors import *
from video_cap import VideoCapture


class TestMotorNode(object):
    def __init__(self, degrees=True, output='images/test/'):
        rospy.loginfo('Starting')
        self.output = output
        if not os.path.exists(output):
            os.makedirs(output)
            print(f"Directory created: {output}")
        else:
            print(f"Directory already exists: {output}")

        motors = get_face_motors()
        self.motor_move_info = get_face_motor_move_info(motors)

        self.motors = motors
        self._motor_states = [None]*len(self.motors)
        self.degrees = degrees
        self.motor_lock = threading.Lock()
        
        self.is_moved_pub = rospy.Publisher('isMoved', String, queue_size=1)
        self.motor_pub = rospy.Publisher('/hr/actuators/pose', TargetPosture, queue_size=1)
        rospy.Subscriber("isMoved", String, self.take_photo_callback)
        
        self.names = motors
        time.sleep(3.0)

    def take_photo_callback(self, msg):
        self.take_photo(self.output, str(msg.data))
      
    def test_random_face(self, times=1):
        rospy.loginfo('Running random face')
        sleep_time = 1
        for i in range(times):
            '''time.sleep(sleep_time)
            # save: neutral, all_motors, half_motor_symmetric,
            self.move_to_neutral_state()
            time.sleep(sleep_time)
            self.is_moved_pub.publish(str(i)+'_neutral')'''

            random_values = self.get_random_movement(gaussian=True) # for all the face motors
            time.sleep(sleep_time)
            self.move(random_values)
            time.sleep(sleep_time)
            self.is_moved_pub.publish(str(i)+'_random_all')

        rospy.signal_shutdown('End')
        sys.exit()

    def test_symmetric_face(self, times=1):
        rospy.loginfo('Running symmetric face')
        sleep_time = 5
        for i in range(times):
            time.sleep(sleep_time)
            center_values = self.get_random_movement(self.motors[:4])
            left_values = self.get_random_movement(self.motors[4:15])
            right_values = [-x for x in left_values]
            all_values = center_values + left_values + right_values
            self.move(all_values)
            time.sleep(sleep_time)
            self.is_moved_pub.publish(str(i))

        rospy.signal_shutdown('End')
        sys.exit()

    def take_photo(self, output_path, output_name):
        filename = output_path + 'img_' + output_name + '.png'
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

    def _check_limits(self, name, value): # dont need this
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
    global cam
    cam = VideoCapture(-1)
    rospy.init_node('TestMotorNode')
    node = TestMotorNode(output='images/test/')
    node.test_random_face(times=1)
    #node.test_symmetric_face(times=10)
