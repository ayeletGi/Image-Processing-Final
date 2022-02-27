import cv2
import matplotlib.pyplot as plt
import timeit
import numpy as np

IMAGES_PATH = './images/'
BLACK, WHITE = 0, 255


# Generic functions
def plot_gray_imshow(img, title):
    plt.imshow(img, cmap='gray', vmin=BLACK, vmax=WHITE)
    plt.title(title)


def plot_color_imshow(img, title):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)


class Image:
    org_color_key = "original color"
    org_gray_key = "original gray"
    gblur_key = "gaussian blur"
    mblur_key = "meddian blur"
    sharpen_key = "sharpen"
    thresh_key = "threshold"
    lap_key = "laplacian"
    canny_key = "canny edges"
    hough_lines_key = "hough lines color"
    mopen_key = "morph open cross"
    mclose_key = "morph close cross"
    fcontours_key = "find and fill contours"
    fill_polly_key = "fill poly"
    brect_key = "bounding rect color"
    cropping_key = "cropping frame"

    def __init__(self, title, path) -> None:
        self.title = title
        self.path = path
        self.variations = {}    # dict[str, np.array]
        self.pieces = []    # list[Piece]

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

    def gaussian_blur(self, source, kernel_size):
        blurred = cv2.GaussianBlur(self.variations[source],
                                   (kernel_size, kernel_size), 0)
        self.variations[self.gblur_key] = blurred

    def meddian_blur(self, source, kernel_size):
        blurred = cv2.medianBlur(self.variations[source],
                                 kernel_size)
        self.variations[self.mblur_key] = blurred

    def sharpen(self, source):
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
        image_sharp = cv2.filter2D(src=self.variations[source],
                                   ddepth=-1,
                                   kernel=kernel)
        self.variations[self.sharpen_key] = image_sharp
        
    def threshold(self, source, block_size, c):
        thresh = cv2.adaptiveThreshold(self.variations[source],
                                       maxValue=WHITE,
                                       adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       thresholdType=cv2.THRESH_BINARY_INV,
                                       blockSize=block_size,
                                       C=c)
        self.variations[self.thresh_key] = thresh

    def laplacian_derivatives(self, source):
        laplacian = cv2.Laplacian(self.variations[source],
                                  cv2.CV_64F)
        # abs_laplacian = cv2.convertScaleAbs(laplacian)
        self.variations[Image.lap_key] = laplacian

    def canny_edges(self, source):
        canny = cv2.Canny(self.variations[source], 50, 200)
        self.variations[Image.canny_key] = canny
    
    def hough_lines(self, source):
        """_summary_
        Args:
            source (_type_): _description_
        """
        # use canny edge detection
        edges = cv2.Canny(self.variations[source], 50, 150, apertureSize=3)
        
        # apply hough transform to detect lines
        lines = cv2.HoughLines(edges,1,np.pi/180,600)
        
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
                if x1 < cols / 2 :
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
            
    def find_and_fill_contours(self, source):
        """_summary_
        Args:
            source (_type_): _description_
        """
        result = self.variations[source].copy()
        
        contours, __ = cv2.findContours(self.variations[source],
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(result,
                         contours,
                         -1,
                         color = (255, 255, 255),
                         thickness=cv2.FILLED, 
                         lineType=cv2.LINE_AA)
        
        self.variations[Image.fcontours_key] = result
    
    def fill_poly(self, source):
        """_summary_
        Args:
            source (_type_): _description_
        """
        result = self.variations[source].copy()

        contours, __ = cv2.findContours(self.variations[source],
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_NONE)
        cv2.fillPoly(result,
                     contours,
                     color=(255, 255, 255))

        self.variations[Image.fill_polly_key] = result

    def bounding_rect(self, source, min, max):
        """_summary_
        Args:
            source (_type_): _description_
            rect_range (_type_): _description_
        """
        result = self.variations[Image.org_color_key].copy()

        contours, __ = cv2.findContours(self.variations[source],
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_NONE)
        for c in contours:
            # finding smallest blocking rectangle
            x, y, w, h = cv2.boundingRect(c)
            # really small rectangles
            if w < min and h < min:
                continue
            # too big
            if w > max and h > max:
                continue
            # drawing
            cv2.rectangle(result,
                            (x, y),
                            (x + w, y + h),
                            (0, 0, 255),
                            4)
        self.variations[Image.brect_key] = result
        
    def morphological_transform(self, source, struct, kernel_size, method, iter):
        """_summary_
        Args:
            source (_type_): _description_
            struct (_type_): _description_
            kernel_size (_type_): _description_
            method (_type_): _description_
            iter (_type_): _description_
        """
        kernel = cv2.getStructuringElement(struct,
                                           (kernel_size, kernel_size))
        result = cv2.morphologyEx(self.variations[source],
                                   method,
                                   kernel,
                                   iterations=iter)
        if method == cv2.MORPH_OPEN:
            self.variations[Image.mopen_key] = result
        else:
            self.variations[Image.mclose_key] = result

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

    
class Process:

    def __init__(self, images_numbers, prefix, suffix) -> None:
        self.images_numbers = images_numbers
        self.prefix = prefix
        self.suffix = suffix
        self.images = []

    def open_images(self, scale_percent = -1):
        """_summary_
        Args:
            scale_percent (_type_): _description_
        """
        for i in self.images_numbers:
            # calculate path
            title = self.prefix + str(i)
            path = IMAGES_PATH + title + self.suffix
            # read image
            img_color = cv2.imread(path, cv2.IMREAD_COLOR)
            
            if img_color is None:
                print(f"Failed to read image, check the path: {path}")
            else:
                if scale_percent > 0:
                    # resize image
                    width = int(img_color.shape[1] * scale_percent / 100)
                    height = int(img_color.shape[0] * scale_percent / 100)
                    dim = (width, height)

                    img_color = cv2.resize(img_color,
                                        dim,
                                        interpolation=cv2.INTER_AREA)
                # convert to gray
                img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
                # create Image object and add original variations
                image = Image(title, path)
                image.variations[Image.org_color_key] = img_color
                image.variations[Image.org_gray_key] = img_gray
                # append Image to the images list
                self.images.append(image)

    def find_pieces(self):
        """_summary_
        """
        print("Please be patience, this pipline may take a while..")
        start = timeit.default_timer()

        for img in self.images:   
            # img=Image(img)
            # first step - cleaning noises
            img.meddian_blur(source=Image.org_gray_key,
                             kernel_size=11)
            img.gaussian_blur(source=Image.mblur_key,
                              kernel_size=11)
            img.sharpen(source=Image.gblur_key)
            img.threshold(source=Image.sharpen_key,
                          block_size=601,
                          c=5)
            img.hough_lines(source=Image.thresh_key)
            
            # second step - detect contours and fill
            img.find_and_fill_contours(source=Image.cropping_key)
            img.morphological_transform(source=Image.fcontours_key,
                                        struct=cv2.MORPH_CROSS,
                                        kernel_size=5,
                                        method=cv2.MORPH_CLOSE,
                                        iter=2)
            img.morphological_transform(source=Image.mclose_key,
                                        struct=cv2.MORPH_CROSS,
                                        kernel_size=7,
                                        method=cv2.MORPH_OPEN,
                                        iter=2)
            
            # third step - fill holes
            img.fill_poly(source=Image.mopen_key)
            img.morphological_transform(source=Image.fill_polly_key,
                                        struct=cv2.MORPH_RECT,
                                        kernel_size=5,
                                        method=cv2.MORPH_CLOSE,
                                        iter=2)

            # final step - detect bounding rect
            img.bounding_rect(source=Image.mclose_key, min=5, max=2000)

        stop = timeit.default_timer()
        print(f"Pipline finished! time: {stop - start} seconds")
        
    def plot_images_varaitions(self):
        for img in self.images:
            img.plt_variations()
    
    def plot_images_results(self):
        for img in self.images:
            img.plt_final_result()

        
def main():
    # images_numbers = range(1, 8)
    images_numbers = [6]
    process = Process(images_numbers, prefix='image', suffix='.jpg')
    process.open_images()
    process.find_pieces()
    # process.plot_images_varaitions()
    process.plot_images_results()


if __name__ == '__main__':
    main()
