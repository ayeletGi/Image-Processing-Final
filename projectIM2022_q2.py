import pandas as pd
from projectIM2022_q1 import Q1_LOG_PATH
import cv2
import matplotlib.pyplot as plt
import numpy as np
import timeit

IMAGES_PATH = './images/'
Q2_LOG_PATH = './q2_output.xlsx'
BLACK, WHITE = 0, 255


def find_piece_contours(rect_gray):
    """ 
    Find the closest contours for a given piece using our modified algorithm.
    Args:
        rect_gray (np.asarray): source image (rect).
    Returns:
        list: closest contours points.
    """
    # blur
    blur = cv2.bilateralFilter(rect_gray, 15, 100, 100)
    # threshold
    thresh = cv2.adaptiveThreshold(blur,
                                   maxValue=WHITE,
                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   thresholdType=cv2.THRESH_BINARY_INV,
                                   blockSize=751,
                                   C=3)
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
        (x, y, w, h) = cv2.boundingRect(c)
        if w < 40 or h < 40:
            continue
        if x != 0 and x+w != cols and y != 0 and y+h != rows:
            good_cntrs.append(c)

    return good_cntrs


def process_image_pieces(title, suffix,  pad):
    """
    Process the given title image - find all the pieces the contours,
    saving them into Q2_LOG_PATH file and plotting the results.
    Args:
        title (str): image title
        suffix (str): each image name common end from the title.
    """
    print(f"starting {title}")
    start = timeit.default_timer()

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
    final_cntrs_str = []

    for rect in rect_list:
        # go over each rectangle and find inside contour
        __, point = rect
        r_x1, r_y1, r_x2, r_y2 = point.split(",")
        r_x1, r_y1, r_x2, r_y2 = int(r_x1), int(r_y1), int(r_x2), int(r_y2)
        
        rect_gray = img_gray[r_y1-pad:r_y2+pad, r_x1-pad:r_x2+pad]
        good_cntrs = find_piece_contours(rect_gray)

        # go over the piece final cntrs and add them to final vars
        piece_cntrs_str = ""
        
        for c_index, cntr in enumerate(good_cntrs):
            for l_index, line in enumerate(cntr):
                for p_index, point in enumerate(line):
                    # move the contour 
                    x,y = point
                    x += r_x1 - pad
                    y += r_y1 - pad
                   
                    # to string   
                    piece_cntrs_str += "(" + str(x) + "," + str(y) + "),"

                    # update
                    good_cntrs[c_index][l_index][p_index][0] = x
                    good_cntrs[c_index][l_index][p_index][1] = y
                    
        piece_cntrs_str = piece_cntrs_str[:-1]
        final_cntrs.append(good_cntrs)
        final_cntrs_str.append(piece_cntrs_str)
        
    # go over all pieces and draw their contour on the final image
    for piece in final_cntrs:
        for cntr in piece:
            cv2.drawContours(img_color, cntr, -1, (255, 0, 0), 4)
    
    # write the final image contours into the log file
    index = range(1, len(final_cntrs_str)+1)
    df = pd.DataFrame(zip(index, final_cntrs_str),
                        columns=['piece', 'contour'])
    df = df.set_index('piece')
    
    with pd.ExcelWriter(Q2_LOG_PATH, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=title)

    stop = timeit.default_timer()
    print(f"Pipline finished! time: {stop - start} seconds")

    # plot result
    plt.title(title)
    plt.imshow(img_color)
    plt.axis('off')
    plt.show()
    
    
def main(images_numbers, prefix, suffix):
    """
    Main method to find the pieces contours on each given image. 
    Each image path is calculates by: IMAGES_PATH + prefix + str(i) + suffix.
    Args
        images_numbers (list): images numbers to open.
        suffix (str): each image name common start up until the number.
        prefix (str): each image name common end from the number.
    """
    for i in images_numbers:
        title = prefix + str(i)
        process_image_pieces(title, suffix, pad=170)

    
if __name__ == "__main__":
    images_numbers = range(1, 8)
    main(images_numbers, prefix='image', suffix='.jpg')
