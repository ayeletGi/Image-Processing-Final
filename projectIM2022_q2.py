from http.client import CONTINUE
from turtle import shape
import pandas as pd
from projectIM2022_q1 import Q1_LOG_PATH
import cv2 
import matplotlib.pyplot as plt
import numpy as np

prefix = 'image'
suffix = '.jpg'
IMAGES_PATH = './images/'
Q2_LOG_PATH = './q2_output.xlsx'
BLACK, WHITE = 0, 255

def find_piece_contours(rect_gray):
    """_summary_
    Args:
        rect_gray (_type_): _description_
    """
    # blur
    blur = cv2.bilateralFilter(rect_gray, 15, 75, 75)
    # threshold
    thresh = cv2.adaptiveThreshold(blur,
                                      maxValue=WHITE,
                                      adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      thresholdType=cv2.THRESH_BINARY_INV,
                                      blockSize=651,
                                      C=-2)
    # contour and fill
    result = thresh.copy()
    contours, __ = cv2.findContours(thresh,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(result,
                         contours,
                         -1,
                         color=WHITE,
                         thickness=cv2.FILLED)
    result = result.astype(np.uint8)

    # find final external contours
    contours, __ = cv2.findContours(
        result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # delete contours that touch the rectangle
    good_cntrs = []
    for c in contours:
        rows, cols = rect_gray.shape
        (x,y,w,h) = cv2.boundingRect(c)
        if w < 40 or h < 40:
            continue
        if x != 0 and x+w != cols and y != 0 and y+h != rows:
            good_cntrs.append(c)

    return good_cntrs


def process_image_pieces(title):
    path = IMAGES_PATH + title + suffix
    
    # read image
    img_color = cv2.imread(path, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # read log sheet and convert to list of coordinations
    rect_df = pd.read_excel(Q1_LOG_PATH,  sheet_name=title)
    records = rect_df.to_records(index=False)
    
    # init helping vars
    rect_list = list(records)
    final_cntrs = []
    
    # go over each rectangle and find inside contour
    for rect in rect_list:
        __, x1, y1, x2, y2 = rect
        pad = 100
        rect_gray = img_gray[y1-pad:y2+pad, x1-pad:x2+pad]
        good_cntrs = find_piece_contours(rect_gray)

        for c_index, cntr in enumerate(good_cntrs):
            for p_index, __ in enumerate(cntr):
                good_cntrs[c_index][p_index][0][0] += x1 - pad
                good_cntrs[c_index][p_index][0][1] += y1 - pad
        
        final_cntrs.append(good_cntrs)   
    
    # go over all pieces and draw their contour on the final image
    for piece in final_cntrs:
        for cntr in piece:
            cv2.drawContours(img_color, cntr, -1, (255, 0, 0), 4)
    
    # plot result
    plt.title(title)
    plt.imshow(img_color)
    plt.show()


if __name__ == "__main__":
    # images_numbers = range(1, 8)
    images_numbers = [3]
    
    for i in images_numbers:
        title = prefix + str(i)
        process_image_pieces(title)
