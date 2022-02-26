from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import numpy as np
import cv2
import matplotlib.pyplot as plt

IMAGES_PATH = './images/'
BLACK, WHITE = 0, 255


# Generic functions
def plot_gray_imshow(img, title):
    plt.imshow(img, cmap='Greys', vmin=BLACK, vmax=WHITE)
    plt.title(title)
    

def plot_color_imshow(img, title):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)


def out_of_bound(index, max_length):
    if 0 <= index < max_length:
        return False
    return True


class Piece:
    
    def __init__(self) -> None:
       pass
    

class Image:
    
    def __init__(self, title, path, img_gray, img_color) -> None:
        self.title = title
        self.path = path
        self.gray = img_gray
        self.color = img_color
        self.variations = {}    # dict[str, np.array]
        self.pieces = []    # list[Piece]
    
    def plt_variations(self, rows = 2):
        total = len(self.variations) + 2    # extra 2 originals
        cols = (total // rows) + (total % rows > 0)
        
        plt.suptitle(self.title, size=16)
        plt.subplot(rows, cols, 1)
        plot_color_imshow(self.color, "Original-color")

        plt.subplot(rows, cols, 2)
        plot_gray_imshow(self.gray, "Original-gray")
        
        for i, (title, img) in enumerate(self.variations.items()):
            plt.subplot(rows, cols, i + 3)
            plot_color_imshow(img, title)
        plt.show()

    def gaussian_blur(self, kernel_size):
        blur_gray = cv2.GaussianBlur(self.gray, (kernel_size, kernel_size), 0)
        self.variations["blur gray"] = blur_gray
        
    def threshold(self, value):
        # if "blur gray" in self.variations:
        #     __, result = cv2.threshold(self.variations["blur gray"], value, WHITE, cv2.THRESH_BINARY_INV)
        # else:
        __, result = cv2.threshold(self.gray, value, WHITE, cv2.THRESH_BINARY_INV)
        self.variations["threshold"] = result   


        
    def crop_frame(self):
        pass

class Process:
    
    def __init__(self, num, prefix, suffix) -> None:
        self.num = num
        self.prefix = prefix
        self.suffix = suffix
        self.images = []

    def open_images(self):
        """
        """
        for i in range(1, self.num + 1):
            title = self.prefix + str(i)
            path = IMAGES_PATH + title + self.suffix

            img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_color = cv2.imread(path, cv2.IMREAD_COLOR)

            if (img_color is None) or (img_gray is None):
                print(f"Failed to read image, check the path: {path}")
            else:
                image = Image(title, path, img_gray, img_color)
                self.images.append(image)

    def apply_threshold(self, value):
        for img in self.images:
            img.threshold(value)
    
    def apply_gaussian_blur(self, kernel_size):
        for img in self.images:
            img.gaussian_blur(kernel_size)
            
    def plt_images_variations(self):
        for img in self.images:
            img.plt_variations()

#
# def sobel_filters(img):
#     """
#     Calculate the sobel filters over the image x ax and y ax.
#     Returns the gradients and their angles.
#     """
#     Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
#     Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)
#
#     Ix = ndimage.filters.convolve(img, Kx)
#     Iy = ndimage.filters.convolve(img, Ky)
#
#     G = np.hypot(Ix, Iy)
#     G = G / G.max() * 255
#     theta = np.arctan2(Iy, Ix)
#
#     return (G, theta)


def main():
    process = Process(num=7, prefix='image', suffix='.jpg')
    process.open_images()
    process.apply_gaussian_blur(kernel_size=15)
    process.apply_threshold(value=190)
    process.plt_images_variations()


if __name__ == '__main__':
    main()

