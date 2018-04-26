# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np
from pprint import pprint

#TODO:create own .trainneddata that can be used with below preprocesing for all types
def preprocess(img):
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,55,10)
    img = getCroppedImage(img)
    return img

def getCroppedImage(img):
    #getting cropped image
    coords = np.column_stack(np.where(img > 0))
    rect = cv2.minAreaRect(coords)

    rect = (rect[0][::-1],rect[1],-rect[2]-90)

    angle = rect[2]
    if angle < -45.0:
        angle += 90.0

    rotMatrix = cv2.getRotationMatrix2D(rect[0], angle, 1)
    img = cv2.warpAffine(img, rotMatrix, img.shape[0:2][::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if rect[2] < -45.0:
        w, h = rect[1][::-1]
    else:
        w, h = rect[1]

    img = cv2.getRectSubPix(img, (int(w), int(h)), rect[0])
    return img

#***********************************************************************************************************************

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
#gray = cv2.bitwise_not(gray) #TODO: get selective cases for when to apply and not apply this

preprocessedImg = preprocess(gray)

# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, preprocessedImg)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pytesseract.image_to_string(Image.open(filename),lang="eng")
os.remove(filename) #TODO: check if this works
print(text)

# show the output images
cv2.imshow("Image", gray)
cv2.imshow("Output", preprocessedImg)
cv2.waitKey(0)
