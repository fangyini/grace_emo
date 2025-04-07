import os
import glob
import numpy as np
import dlib
import cv2

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def calculate_sim(path):
    all_dist = []
    all_shapes = []
    for filename in glob.glob(os.path.join(path, '*.png')):
        image = cv2.imread(filename)
        # downsample?
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rect = detector(gray, 1)[0]
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        (x, y, w, h) = rect_to_bb(rect)
        '''cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # show the face number
        cv2.putText(image, "Face #{}".format(1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        # show the output image with the face detections + facial landmarks
        cv2.imshow("Output", image)
        cv2.waitKey(0)'''

        all_shapes.append(shape)
    for i in range(len(all_shapes)):
        for j in range(i+1, len(all_shapes)):
            dist = abs(all_shapes[i] - all_shapes[j])
            all_dist.append(dist)
    return all_dist

def calculate_sym(path):
    all_dist = []
    all_shapes = []
    for filename in glob.glob(os.path.join(path, '*.png')):
        image = cv2.imread(filename)
        # downsample?
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rect = detector(gray, 1)[0]
        shape = predictor(gray, rect)
        shape = shape_to_np(shape) # right is set, left is mirrored

        (x, y, w, h) = rect_to_bb(rect) # draw the predicted symmetric
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # show the face number
        cv2.putText(image, "Face #{}".format(1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        # show the output image with the face detections + facial landmarks
        cv2.imshow("Output", image)
        cv2.waitKey(0)

        all_shapes.append(shape)
    for i in range(len(all_shapes)):
        for j in range(i + 1, len(all_shapes)):
            dist = abs(all_shapes[i] - all_shapes[j])
            all_dist.append(dist)
    return all_dist

if __name__ == '__main__':
    image_path = '/Users/xiaokeai/Desktop/images/'
    neutral_path = ['neutral/', ]
    sym_path = ['sym/', ]
    test_path = ['test/', 'test2/', 'test3/']
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('pretrained/shape_predictor_68_face_landmarks.dat')
    '''overall_dist = []
    for folder in neutral_path:
        dist = calculate_sim(os.path.join(image_path, folder))
        overall_dist.extend(dist)
    print('neutral diff: ', np.mean(overall_dist))
    
    overall_dist = []
    for folder in test_path:
        dist = calculate_sim(os.path.join(image_path, folder))
        overall_dist.extend(dist)
    print('random movement diff: ', np.mean(overall_dist))'''

    overall_dist = []
    for folder in sym_path:
        dist = calculate_sym(os.path.join(image_path, folder))
        overall_dist.extend(dist)
    print('face symmetric diff: ', np.mean(overall_dist))


