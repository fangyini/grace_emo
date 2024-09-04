import cv2, glob

cam = cv2.VideoCapture(-1)
result, image = cam.read() 

if result: 
    cv2.imwrite("./images/test/test.png", image)
else: 
    print(result)
    print("No image detected. Please! try again")
