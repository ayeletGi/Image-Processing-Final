from http.client import CONTINUE
from turtle import shape
import pandas as pd
from projectIM2022_q1 import Q1_LOG_PATH
import cv2 
import matplotlib.pyplot as plt
import numpy as np

# read image
img_color = cv2.imread("./images/image1.jpg", cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# read log sheet and convert to list of coordinations
rect_df = pd.read_excel(Q1_LOG_PATH,  sheet_name='image1')
records = rect_df.to_records(index=False)
rect_list = list(records)
final_cntrs = []

for rect in rect_list:
    i, x1, y1, x2, y2 = rect
    pad = 70
    # go over each rectangle and find inside contour
    rect_gray = img_gray[y1-pad:y2+pad, x1-pad:x2+pad]
    rect_color = img_color[y1-pad:y2+pad, x1-pad:x2+pad]
    blur = cv2.bilateralFilter(rect_gray,
                               15, 75, 75)

    thresh = cv2.adaptiveThreshold(blur,
                                      maxValue=255,
                                      adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      thresholdType=cv2.THRESH_BINARY_INV,
                                      blockSize=651,
                                      C=-2)
    result = thresh.copy()
    contours, __ = cv2.findContours(thresh,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(result,
                         contours,
                         -1,
                         color=255,
                         thickness=cv2.FILLED)
    result = result.astype(np.uint8)

    # Find contours and draw result
    contours, __ = cv2.findContours(
        result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # delete contours that touch the rectangle
    good_cntrs = []

    for c in contours:
        rows, cols = rect_gray.shape
        (x,y,w,h) = cv2.boundingRect(c)
        if w < 20 or h < 20:
            continue
        if x != 0 and x+w != cols and y != 0 and y+h != rows:
            good_cntrs.append(c)

    for c_index, cntr in enumerate(good_cntrs):
        for p_index, point in enumerate(cntr):
            good_cntrs[c_index][p_index][0][0] += x1 - pad
            good_cntrs[c_index][p_index][0][1] += y1 - pad
    
    
    final_cntrs.append(good_cntrs)   
    
for piece in final_cntrs:
    for cntr in piece:
        cv2.drawContours(img_color, cntr, -1, (255, 0, 0), 4)
plt.imshow(img_color)
plt.show()

