import os
import cv2

def downsample(img, num):
    return img[::num, ::num]

def batch_downsample(source_filename, destination_filename, top, left, height, width, num):
    source_img = 0
    for root, dirs, files in os.walk(source_filename):
        source_img = files
    number = 0
    for i in source_img:
        img = cv2.imread(filename=os.path.join(source_filename, img=i))
        img = img[top:top + height:num, left:left + width:num, :]
        cv2.imwrite(filename=os.path.join(destination_filename, i), img=img)
        number += 1