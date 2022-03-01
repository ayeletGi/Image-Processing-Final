import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import timeit
from typing import List

IMAGES_PATH = './images/'
Q1_LOG_PATH = './q1_output.xlsx'
BLACK, WHITE = 0, 255

# Generic functions

def plot_gray_imshow(img, title):
    """_summary_
    Args:
        img (_type_): _description_
        title (_type_): _description_
    """
    plt.imshow(img, cmap='gray', vmin=BLACK, vmax=WHITE)
    plt.title(title)
    plt.axis('off')


def plot_color_imshow(img, title):
    """_summary_
    Args:
        img (_type_): _description_
        title (_type_): _description_
    """
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')


class Image:
    """_summary_
    """
    org_color_key = "original color"
    org_gray_key = "original gray"
    bblur_key = "bilateral filter"
    thresh_key = "threshold"
    del_temp_key = "delete template"
    hough_lines_key = "hough lines color"
    cropping_key = "cropping frame"
    erosion_key = "erosion"
    dilation_key = "dilation"
    fcontours_key = "find and fill contours"
    mblur_key = "meddian blur"
    brect_key = "bounding rect color"


    def __init__(self, title, path) -> None:
        self.title = title
        self.path = path
        self.variations = {}
        self.pieces = []


    def plt_variations(self):
        """_summary_
        """
        # calculate rows and cols
        rows = 2
        total = len(self.variations)
        cols = (total // rows) + (total % rows > 0)

        # main title
        plt.suptitle(self.title, size=16)

        # plotting all image variations in subplots
        for i, (title, img) in enumerate(self.variations.items()):
            plt.subplot(rows, cols, i + 1)
            if "color" in title:
                plot_color_imshow(img, title)
            else:
                plot_gray_imshow(img, title)

        plt.show()


    def delete_template(self, source):
        """_summary_
        Args:
            source (_type_): _description_
        """
        # read template image
        template = cv2.imread(IMAGES_PATH + 'template.jpg', cv2.IMREAD_COLOR)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # init helping vars
        temp_h, temp_w = template.shape[::]
        org_h, __ = self.variations[self.org_gray_key].shape

        # find the template inside the original image
        detected = cv2.matchTemplate(
            self.variations[self.org_gray_key], template, cv2.TM_CCOEFF_NORMED)

        # take only matches with score above 0.5
        threshold = 0.5
        loc = np.where(detected >= threshold)

        # delete all matches by coloring them black in the image
        result = self.variations[source].copy()
        for pt in zip(*loc[::-1]):

            if pt[1] < org_h - org_h//5:
                continue

            result[pt[1]:pt[1] + temp_h, pt[0]:pt[0] + temp_w] = BLACK
            result[pt[1] + temp_h:, :] = BLACK

        self.variations[self.del_temp_key] = result


    def bilateral_blur(self, source, d, sigma_color, sigma_space):
        """_summary_
        Args:
            source (_type_): _description_
            d (_type_): _description_
            sigma_color (_type_): _description_
            sigma_space (_type_): _description_
        """
        blurred = cv2.bilateralFilter(self.variations[source],
                                      d, sigma_color, sigma_space)
        self.variations[self.bblur_key] = blurred


    def meddian_blur(self, source, kernel_size):
        """_summary_
        Args:
            source (_type_): _description_
            kernel_size (_type_): _description_
        """
        blurred = cv2.medianBlur(self.variations[source],
                                 kernel_size)
        self.variations[self.mblur_key] = blurred


    def threshold(self, source, block_size, c):
        """_summary_
        Args:
            source (_type_): _description_
            block_size (_type_): _description_
            c (_type_): _description_
        """
        thresh = cv2.adaptiveThreshold(self.variations[source],
                                       maxValue=WHITE,
                                       adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       thresholdType=cv2.THRESH_BINARY_INV,
                                       blockSize=block_size,
                                       C=c)
        self.variations[self.thresh_key] = thresh


    def hough_lines(self, source):
        """_summary_
        Args:
            source (_type_): _description_
        """
        # use canny edge detection
        edges = cv2.Canny(self.variations[source], 50, 150, apertureSize=3)

        # apply hough transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, 600)

        # iterate over points
        lines_result = self.variations[self.org_color_key].copy()
        crop_result = self.variations[source].copy()

        rows, cols, __ = lines_result.shape
        startx, endx = 0, cols
        starty, endy = 0, rows

        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            # vertical lines
            if theta > np.pi/180 * 170 or theta < np.pi/180 * 10:
                if x1 < cols / 2:
                    startx = max(startx, x1, x2)
                else:
                    endx = min(endx, x1, x2)
                cv2.line(lines_result, (x1, y1), (x2, y2), (0, 0, 255), 6)

            # horizontal lines
            if theta > np.pi/180 * 80 and theta < np.pi/180 * 100:
                if y1 < rows / 2:
                    starty = max(starty, y1, y2)
                else:
                    endy = min(endy, y1, y2)
                cv2.line(lines_result, (x1, y1), (x2, y2), (255, 0, 0), 6)

        # color black until frame coordinations
        crop_result[:starty, :] = BLACK
        crop_result[:, :startx] = BLACK
        crop_result[endy:, :] = BLACK
        crop_result[:, endx:] = BLACK

        self.variations[self.hough_lines_key] = lines_result
        self.variations[self.cropping_key] = crop_result


    def erode(self, source, struct, kernel_size, iter):
        """_summary_
        Args:
            source (_type_): _description_
            struct (_type_): _description_
            kernel_size (_type_): _description_
            iter (_type_): _description_
        """
        kernel = cv2.getStructuringElement(struct,
                                           (kernel_size, kernel_size))
        erosion = cv2.morphologyEx(self.variations[source],
                                   cv2.MORPH_ERODE,
                                   kernel,
                                   None,
                                   None,
                                   iter,
                                   cv2.BORDER_REFLECT101)
        self.variations[Image.erosion_key] = erosion


    def dilate(self, source, struct, kernel_size, iter):
        """_summary_
        Args:
            source (_type_): _description_
            struct (_type_): _description_
            kernel_size (_type_): _description_
            iter (_type_): _description_
        """
        kernel = cv2.getStructuringElement(struct,
                                           (kernel_size, kernel_size))
        dilation = cv2.morphologyEx(self.variations[source],
                                    cv2.MORPH_DILATE,
                                    kernel,
                                    None,
                                    None,
                                    iter,
                                    cv2.BORDER_REFLECT101)
        self.variations[Image.dilation_key] = dilation


    def find_and_fill_contours(self, source):
        """_summary_
        Args:
            source (_type_): _description_
        """
        size = self.variations[source].shape
        result = np.zeros(size)

        contours, __ = cv2.findContours(self.variations[source],
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(result,
                         contours,
                         -1,
                         color=WHITE,
                         thickness=cv2.FILLED)

        result = result.astype(np.uint8)
        self.variations[Image.fcontours_key] = result


    def find_bounding_rect(self, source):
        """_summary_
        Args:
            source (_type_): _description_
        """
        contours, __ = cv2.findContours(self.variations[source],
                                        cv2.RETR_CCOMP,
                                        cv2.CHAIN_APPROX_NONE)
        self.pieces = [cv2.boundingRect(c) for c in contours]


    def keep_good_pieces(self, min, max, border_size, min_gray, max_high_width_ratio, min_dark_pixels):
        """_summary_
        Args:
            min (_type_): _description_
            max (_type_): _description_
            border_size (_type_): _description_
            min_gray (_type_): _description_
        """
        # helping vars
        org_gray = self.variations[Image.org_gray_key]
        (rows, cols) = org_gray.shape
        good_pieces = []

        # remove small and huge rectangles
        self.pieces = [(x, y, w, h) for (x, y, w, h)
                       in self.pieces if (min < w < max and min < h < max)]

        # remove too long rectangles (w/h > 6)
        self.pieces = [(x, y, w, h) for (x, y, w, h)
                       in self.pieces if (abs(h//w) < max_high_width_ratio)]

        # remove if rect is too close to the borders
        self.pieces = [(x, y, w, h) for (x, y, w, h) in self.pieces if not
                       (y < border_size or y > rows - border_size or x < border_size or x > cols - border_size)]

        for rect1 in self.pieces:
            x1, y1, w1, h1 = rect1

            # check if area contains dark pixels
            area = org_gray[y1:y1 + h1, x1:x1 + w1]
            dark_pix = np.sum(area <= min_gray)
            if dark_pix <= min_dark_pixels:
                continue

            # check if the rect is completely inside another one
            parent = 0
            for rect2 in self.pieces:
                x2, y2, w2, h2 = rect2
                if rect1 == rect2:
                    continue
                if x2 <= x1 and y2 <= y1 and (x2+w2) >= (x1+w1) and (y2+h2) >= (y1+h1):
                    parent += 1

            if parent == 0:
                good_pieces.append(rect1)

        # sort by distance from origin
        good_pieces.sort(key=lambda p: math.hypot(p[0], p[1]))
        self.pieces = good_pieces


    def draw_pieces(self):
        """_summary_
        """
        result = self.variations[Image.org_color_key].copy()

        for i, rect in enumerate(self.pieces):
            x, y, w, h = rect
            cv2.rectangle(result,
                          (x, y),
                          (x + w, y + h),
                          (0, 0, 255),
                          4)
            cv2.putText(result,
                        str(i+1),
                        (x, y-15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4,
                        (0, 0, 255),
                        5)

        self.variations[Image.brect_key] = result


    def write_to_excel(self):
        """_summary_
        """
        # genereting the representing string and index for each rectangle
        coordinations = [str(x)+","+str(y)+","+str(x+w)+","+str(y+h)
                         for (x, y, w, h) in self.pieces]
        index = range(1, len(coordinations)+1)
        
        # create a dataframe
        df = pd.DataFrame(zip(index, coordinations),
                          columns=['piece', 'rectangle coordinations'])
        df = df.set_index('piece')
        
        # replace/create this image sheet
        with pd.ExcelWriter(Q1_LOG_PATH, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=self.title)


    def plt_final_result(self):
        """_summary_
        """
        # main title
        plt.suptitle(self.title, size=16)
        # plotting original + bounding rect = q1 result
        plt.subplot(1, 2, 1)
        plot_color_imshow(self.variations[Image.org_color_key], "Original")

        plt.subplot(1, 2, 2)
        plot_color_imshow(self.variations[Image.brect_key], "Result")

        plt.show()


images: List[Image] = []

def open_images(images_numbers, suffix, prefix):
    """_summary_
    Args:
        scale_percent (_type_): _description_
    """
    for i in images_numbers:
        # calculate path
        title = prefix + str(i)
        path = IMAGES_PATH + title + suffix
        
        # read image
        img_color = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_color is None:
            print(f"Failed to read image, check the path: {path}")
            
        else:
            # convert to gray
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            
            # create Image object and add original variations
            image = Image(title, path)
            image.variations[Image.org_color_key] = img_color
            image.variations[Image.org_gray_key] = img_gray
            
            # append Image to the images list
            images.append(image)


def find_pieces():
    """_summary_
    """
    print("Please be patience, this pipline may take a while..")

    for img in images:
        print(f"starting {img.title}")
        start = timeit.default_timer()

        # first step - cleaning noises
        img.bilateral_blur(source=Image.org_gray_key,
                            d=15,
                            sigma_color=45,
                            sigma_space=45)
    
        img.threshold(source=Image.bblur_key,
                        block_size=601,
                        c=2)

        img.hough_lines(source=Image.thresh_key)

        img.delete_template(source=Image.cropping_key)

        # second step - detect contours and fill
        img.find_and_fill_contours(source=Image.del_temp_key)

        img.dilate(source=Image.fcontours_key,
                    struct=cv2.MORPH_ELLIPSE,
                    kernel_size=5,
                    iter=4)

        img.erode(source=Image.dilation_key,
                    struct=cv2.MORPH_ELLIPSE,
                    kernel_size=7,
                    iter=4)

        # third step - fill holes
        img.meddian_blur(source=Image.erosion_key,
                            kernel_size=3)

        img.erode(source=Image.mblur_key,
                    struct=cv2.MORPH_CROSS,
                    kernel_size=3,
                    iter=4)

        # final step - detect and draw the bounding rect
        img.find_bounding_rect(source=Image.erosion_key)

        img.keep_good_pieces(min=40,
                                max=3000,
                                border_size=150,
                                min_gray=180,
                                max_high_width_ratio=5,
                                min_dark_pixels=10)

        img.draw_pieces()
        img.write_to_excel()

        stop = timeit.default_timer()
        print(f"Pipline finished! time: {stop - start} seconds")
        
        # img.plt_variations()
        img.plt_final_result()


def main():
    """_summary_
    """
    images_numbers = range(1, 8)
    open_images(images_numbers, prefix='image', suffix='.jpg')
    find_pieces()


if __name__ == '__main__':
    main()
