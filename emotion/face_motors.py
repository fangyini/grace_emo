import yaml
import math
def get_face_motors():
    center = ['BrowCenter', 'Jaw', 'LowerLipCenter', 'UpperLipCenter']
    left = ['BrowInnerLeft', 'BrowOuterLeft', 'CheekSquintLeft', 'FrownLeft',
            'LipsStretchLeft', 'LowerLidLeft', 
            'LowerLipLeft', 'SmileLeft', 'SneerLeft', 
            'UpperLidLeft', 'UpperLipLeft', ]
    right = ['BrowInnerRight', 'BrowOuterRight', 'CheekSquintRight', 'FrownRight',
             'LipsStretchRight', 'LowerLidRight', 
             'LowerLipRight', 'SmileRight', 'SneerRight', 
             'UpperLidRight', 'UpperLipRight', ]
    return center+left+right # [:4], [4:15], [15:26] 4+11*2

def get_face_motor_move_info(motor_list=['BrowCenter', 'Jaw', 'LowerLipCenter', 'UpperLipCenter'], 
                             motor_file='./config/head/motors.yaml'):
    info_dict = {}
    with open(motor_file) as f:
        motor_info = yaml.safe_load(f)['motors']
    for motor_name in motor_list:
        info_dict[motor_name] = {}
        info = motor_info[motor_name]
        max_rad, min_rad = info['max'], info['min']
        max_deg = max_rad * 180 / math.pi
        min_deg = min_rad * 180 / math.pi
        info_dict[motor_name]['max_degree'] = max_deg
        info_dict[motor_name]['min_degree'] = min_deg
    return info_dict

if __name__ == '__main__':
    info_dict = get_face_motor_move_info()
    print(info_dict)