# import required packages
import cv2
import dlib
import time
import os


OUT_DIR = "output/aligned/"
INP_DIR = "./MsCelebV1-DevSet1/aligned/"
images = []

def apply_image_detection(detector, image, color, detector_name):
    # apply face detection (hog)
    faces = detector(image, 1)

    # loop over detected faces
    for face in faces:
        if detector_name == "HOG":
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
        else:
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right() - x
            h = face.rect.bottom() - y
        # draw box over face
        cv2.rectangle(image, (x,y), (x+w,y+h), color, 2)

    img_height, img_width = image.shape[:2]
    cv2.putText(image, detector_name, (img_width-50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    # images = os.listdir(INP_DIR)
    images = ["m.0ftqr.jpg"]

    # initialize hog + svm based face detector
    hog_face_detector = dlib.get_frontal_face_detector()
    # initialize cnn based face detector with the weights
    cnn_face_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')

    start = time.time()

    for img in images:
        print("Processing image:", INP_DIR + img)

        # load input image
        image = cv2.imread(INP_DIR + img)

        if image is None:
            print("Could not read input image")
            exit()
        
        apply_image_detection(hog_face_detector, image, (0, 255, 0), "HOG")
        apply_image_detection(cnn_face_detector, image, (0, 0, 255), "CNN")

        # save output image 
        cv2.imwrite(OUT_DIR + img, image)

    end = time.time()

main()