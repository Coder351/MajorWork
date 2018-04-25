# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np
from pprint import pprint

def preprocess(img):
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,75,20)

    # crop to smallest portion
    _,contours,hierachy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS)


    points = np.concatenate(contours)

    rotatedRect = cv2.minAreaRect(points)

    angle = rotatedRect[2]
    if angle < -45.0:
        angle += 90.0

    rotMatrix = cv2.getRotationMatrix2D(rotatedRect[0],angle,1)
    rotated = cv2.warpAffine(img,rotMatrix,img.shape[0:2][::-1],flags=cv2.INTER_CUBIC)

    if rotatedRect[2] < -45.0:
        w,h = rotatedRect[1][::-1]
    else:
        w,h = rotatedRect[1]


    cropped = cv2.getRectSubPix(rotated,(int(w),int(h)),rotatedRect[0])


    return cropped








# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image to be OCR'd")
# ap.add_argument("-p", "--preprocess", type=str, default="thresh",
#                 help="type of preprocessing to be done")
# args = vars(ap.parse_args())

# load the example image and convert it to grayscale
filepath = input("Enter filepath: ") #example1.jpg
image = cv2.imread(filepath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

preprocess(gray)

# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pytesseract.image_to_string(Image.open(filename),lang="eng")
os.remove(filename)
print(text)

# show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
