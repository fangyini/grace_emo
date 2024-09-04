def get_face_motors(choice='all'):
    center = ['BrowCenter', 'Jaw', 'LowerLipCenter', 'UpperLipCenter']
    left = ['BrowInnerLeft', 'BrowOuterLeft', 'CheekSquintLeft', 'FrownLeft',
            'LipsStretchLeft', 'LowerLidLeft', 
            'LowerLipLeft', 'SmileLeft', 'SneerLeft', 
            'UpperLidLeft', 'UpperLipLeft', ]
    right = ['BrowInnerRight', 'BrowOuterRight', 'CheekSquintRight', 'FrownRight',
             'LipsStretchRight', 'LowerLidRight', 
             'LowerLipRight', 'SmileRight', 'SneerRight', 
             'UpperLidRight', 'UpperLipRight', ]
    if choice == 'all':
        return center+left+right
    elif choice == 'left':
        return left # for symmetric testing

def get_face_motor_dict():
    return