from matplotlib import image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

images_path = './images/'  #  make sure to include / at the end of the images directory path
images_color, images_greys, titles = [], [], []
BLACK, WHITE = 0, 255


def open_images(num, images_perfix, images_suffix):
    """
    """
    if num <= 0:
        print("Wrong images number.")
        
    for i in range(1, num+1):
        image_title = images_perfix + str(i)
        image_path = images_path + image_title + images_suffix
        
        img_gray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        img_color = cv.imread(image_path, cv.IMREAD_COLOR)
        
        if (img_color is None) or (img_gray is None):
            print(f"Failed to read image, check the path: {image_path}")
        else:
            images_color.append(img_color)
            images_greys.append(img_gray)
            titles.append(image_title)


def plot_imshow(img, title):
    """
    """
    plt.imshow(img, cmap='Greys', vmin=BLACK, vmax=WHITE)
    plt.title(title)


def out_of_bound(index, max_length):
    """
    """
    if 0 <= index < max_length:
        return False
    return True
        

def plt_images(images_list, rows):
    """
    """
    images_num = len(images_list)
    cols = (images_num // rows) + (images_num % rows > 0)
    
    for i, (img, title) in enumerate(zip(images_list, titles)):
        plt.subplot(rows, cols, i+1)
        plot_imshow(img, title)
        
    plt.show()
    
    
def apply_threshold(thresh):
    thresh_images = []
    for img in images_greys:
        __, result = cv.threshold(img, thresh, WHITE, cv.THRESH_BINARY_INV)
        thresh_images.append(result)
   
    plt_images(thresh_images, rows = 2)
    return thresh_images


# def contour(gray_img, color_img):

#     # Find Canny edges
#     edged = cv.Canny(gray_img, 30, 200)
#     cv.waitKey(0)
    
#     contours, hierarchy = cv.findContours(
#         gray_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#     cv.drawContours(color_img, contours, -1, (255, 0, 0), 3)
#     cv.imwrite(images_path+'contour.jpg', color_img)

def gs_filter(img):
    """
    Calculate the gaussian filter over the image.
    """
    gs_result = cv.GaussianBlur(img, (3, 3), 0)



def sobel_filters(img):
    """
    Calculate the sobel filters over the image x ax and y ax.
    Returns the gradients and their angles.
    """
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)
        
    
def main():
    open_images(num=7, images_perfix='image', images_suffix='.jpg')
    thresh_images = apply_threshold(thresh = 195)
    contour(thresh_images[0], images_color[0])

if __name__ == '__main__':
    main()
         

# def show_image_part(index, start, size):
#     """
#     """
#     if out_of_bound(index, len(images)):
#         print("Index out of bound.")

#     # with np.printoptions(threshold=1000):  # np.inf will print the full array
#     img = images[index]
#     rows, cols = img.shape

#     if out_of_bound(start, rows) or out_of_bound(start, cols):
#         print("Start out of bound.")
#         return

#     max_row, max_col = start + size, start + size
#     img_part = img[start:max_row, start: max_col]

#     print(f'Image original shape: {rows, cols}, printing the first {max_row, max_col}')
#     print(img_part)

#     plt.subplot(1,2,1)
#     plot_imshow(img)
#     plt.subplot(1, 2, 2)
#     plot_imshow(img_part)
#     plt.show()
