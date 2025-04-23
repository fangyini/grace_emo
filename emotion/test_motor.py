# one random motor once, random degree
# random number of random motors once, random degree
# get previous motor: mean difference < 5
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
import random
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
        self.isError = []
        print('isGau=', isGau)
        rospy.loginfo('Starting')
        self.output = output
        if not os.path.exists(output):
            os.makedirs(output)
            print(f"Directory created: {output}")
        else:
            print(f"Directory already exists: {output}")

        self.output_label = os.path.join(output, 'labels.csv')
        self.output_verify = os.path.join(output, 'labels_verify.csv')

        motors = get_face_motors()
        self.motor_move_info = get_face_motor_move_info(motors)

        self.motors = motors
        self.motor_state_total = []
        self.degrees = degrees
        self.motor_lock = threading.Lock()

        self.motor_pub = rospy.Publisher('/hr/actuators/pose', TargetPosture, queue_size=1)
        rospy.Subscriber('/hr/actuators/motor_states', MotorStateList, self._capture_state)

        self.names = motors
        time.sleep(3.0)
        self.move_to_neutral_state()
        time.sleep(3.0)

    def motor_int_to_angle(motor, position, degrees):
        if degrees:
            unit = 360
        else:
            unit = math.pi
        angle = ((position - motors_dict[motor]['init']) / 4096) * unit
        return angle

    def _capture_state(self, msg):
        self._msg = msg
        _motor_states = {}
        for idx, motor_msg in enumerate(msg.motor_states):
            if motor_msg.name in self.names:
                _motor_states[motor_msg.name] = motor_int_to_angle(motor_msg.name, motor_msg.position, self.degrees)
                if motor_msg.errorCode != 0:
                    with self.motor_lock:
                        print('Motor error: ', motor_msg.name)
                        self.isError.append(1)
        self.motor_state_total.append([time.time_ns(), _motor_states])

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
        with open(self.output_label, 'a') as csv_writer:
            with open(self.output_verify, 'a') as csv_writer2:
                writer = csv.writer(csv_writer, delimiter='\t', lineterminator='\n', )
                writer2 = csv.writer(csv_writer2, delimiter='\t', lineterminator='\n', )
                process_frame_index = starting_num
                deleteLastFrame = False
                while process_frame_index < times:
                    with self.motor_lock:
                        while len(self.isError) > 0:  # wait until no error
                            deleteLastFrame = True
                            print(self.isError)
                            self.isError = []
                            print('detected error, start to sleep...')
                            time.sleep(3)

                        if deleteLastFrame == True:  # also discard several frames
                            deleteLastFrame = False
                            process_frame_index = max(0, process_frame_index - 2)
                            print('deleted last two frames.. starting from ', process_frame_index)
                            # warn: there might be multiple rows with same title in csv

                        self.isError = []  # clear error
                        print('isError cleared')

                    time_str = '0' * (img_num_len - len(str(process_frame_index))) + str(process_frame_index)
                    random_values = self.get_random_movement(gaussian=self.isGau)
                    time.sleep(sleep_time)
                    self.move(random_values)
                    time.sleep(sleep_time)
                    with self.motor_lock:
                        states = self._motor_states

                    row = [time_str, random_values]
                    writer.writerow(row)
                    writer2.writerow([time_str, states])
                    process_frame_index += 1

        print('finised generating ', str(times - starting_num), ' frames')

    def move_specific(self, names: list, values: list):
        values = [self._check_limits(names[i], x) for i, x in enumerate(values)]
        if self.degrees:  # need to convert to radians
            values = [math.radians(x) for x in values]
        args = {"names": names, "values": values}
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

    def get_random_movement(self, number_of_motor=26, gaussian=False):
        names = random.sample(self.motors, number_of_motor)
        values = []
        for i in names:
            min_d, max_d = math.ceil(self.motor_move_info[i]['min_degree']), math.floor(
                self.motor_move_info[i]['max_degree'])
            if gaussian is True:
                sampled_num = np.random.normal()  # around (-3, 3)
                scale = max(max_d, abs(min_d)) / 3
                random_value = sampled_num * scale
                if max_d == 0 and random_value > 0:
                    random_value = -1 * random_value
                elif min_d == 0 and random_value < 0:
                    random_value = -1 * random_value
            else:
                random_value = random.randint(min_d, max_d)
            values.append(random_value)
        return names, values

    def test_latency(self, number_of_motors):
        # random motor commands, get state, check if there is <5 degree, if not redo,
        while len(self.isError) > 0:
            time_start = time.time_ns()
            names, values = self.get_random_movement(number_of_motors, gaussian=False)
            self.move_specific(names, values)
            with self.motor_lock:
                for entry in self.motor_state_total:
                    final_time = entry[0]
                    motors = entry[1]
                    moved_motors = [motors[i] for i in names]
                    if abs(moved_motors - values).mean() < 5:
                        total_time = final_time - time_start
                        break
        print('Number of motors: ', number_of_motors)
        print('Total time: ', total_time)


if __name__ == '__main__':
    signal(SIGINT, handle_sigint)
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_gau", action='store_true', default=True)
    parser.add_argument("--times", type=int, default=600)
    parser.add_argument("--output", type=str, default='dataset/predicted_motors_lrelu_img')  # dataset/gau/
    args = parser.parse_args()
    print(args)

    t1 = time.time()
    global cam
    cam = VideoCapture(-1)
    rospy.init_node('DataCollectionNode')

    for x in range(26):
        for y in range(3):
            node = DataCollectionNode(output=args.output, isGau=args.is_gau)
            node.test_latency(x+1)


    t2 = time.time()
    print('time elapse=', str(int((t2 - t1))), 'sec')
    rospy.signal_shutdown('End')
    sys.exit()
