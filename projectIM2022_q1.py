import cv2
import matplotlib.pyplot as plt
import timeit

IMAGES_PATH = './images/'
BLACK, WHITE = 0, 255


# Generic functions
def plot_gray_imshow(img, title):
    plt.imshow(img, cmap='gray', vmin=BLACK, vmax=WHITE)
    plt.title(title)


def plot_color_imshow(img, title):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)


# class Piece:

#     def __init__(self) -> None:
#         pass


class Image:
    org_color_key = "original color"
    org_gray_key = "original gray"
    gblur_key = "gaussian blur"
    mblur_key = "meddian blur"
    thresh_key = "threshold"
    lap_key = "laplacian"
    canny_key = "canny edges"
    mopen_cross_key = "morph open cross"
    mopen_rect_key = "morph open rect"
    mclose_cross_key = "morph close cross"
    mclose_rect_key = "morph close rect"
    fcontours_key = "find and fill contours"
    fill_polly_key = "fill poly"
    brect_key = "bounding rect color"

    
    def __init__(self, title, path) -> None:
        self.title = title
        self.path = path
        self.variations = {}    # dict[str, np.array]
        # self.pieces = []    # list[Piece]

    def plt_variations(self, rows=2):
        total = len(self.variations)
        cols = (total // rows) + (total % rows > 0)

        plt.suptitle(self.title, size=16)
        
        for i, (title, img) in enumerate(self.variations.items()):
            plt.subplot(rows, cols, i + 1)
            if "color" in title:
                plot_color_imshow(img, title)
            else:
                plot_gray_imshow(img, title)
                
        plt.show()

    def gaussian_blur(self, kernel_size, source):
        blurred = cv2.GaussianBlur(self.variations[source],
                                   (kernel_size, kernel_size), 0)
        self.variations[self.gblur_key] = blurred

    def meddian_blur(self, kernel_size, source):
        blurred = cv2.medianBlur(self.variations[source],
                                 kernel_size)
        self.variations[self.mblur_key] = blurred

    def threshold(self, source, block_size):
        thresh = cv2.adaptiveThreshold(self.variations[source],
                                       maxValue=WHITE,
                                       adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       thresholdType=cv2.THRESH_BINARY_INV,
                                       blockSize=block_size,
                                       C=10)
        self.variations[self.thresh_key] = thresh

    def laplacian_derivatives(self, source):
        laplacian = cv2.Laplacian(self.variations[source],
                                  cv2.CV_64F)
        # abs_laplacian = cv2.convertScaleAbs(laplacian)
        self.variations[Image.lap_key] = laplacian

    def canny_edges(self, source):
        canny = cv2.Canny(self.variations[source], 50, 200)
        self.variations[Image.canny_key] = canny
        
    def find_and_fill_contours(self, source):
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
    
    def morph_rect_open(self, source, kernel_size):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (kernel_size, kernel_size))
        opening = cv2.morphologyEx(self.variations[source],
                                   cv2.MORPH_OPEN,
                                   kernel,
                                   iterations=1)
        self.variations[Image.mopen_rect_key] = opening

    def morph_cross_open(self, source, kernel_size):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,
                                           (kernel_size, kernel_size))
        opening = cv2.morphologyEx(self.variations[source],
                                   cv2.MORPH_OPEN,
                                   kernel,
                                   iterations=1)
        self.variations[Image.mopen_cross_key] = opening

    def morph_cross_close(self, source, kernel_size):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,
                                           (kernel_size, kernel_size))
        closing = cv2.morphologyEx(self.variations[source],
                                   cv2.MORPH_CLOSE,
                                   kernel,
                                   iterations=2)
        self.variations[Image.mclose_cross_key] = closing

    def morph_rect_close(self, source, kernel_size):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (kernel_size, kernel_size))
        closing = cv2.morphologyEx(self.variations[source],
                                   cv2.MORPH_CLOSE,
                                   kernel,
                                   iterations=2)
        self.variations[Image.mclose_rect_key] = closing
        
    def fill_poly(self, source):
        result = self.variations[source].copy()
        
        contours, __ = cv2.findContours(self.variations[source],
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_NONE)
        cv2.fillPoly(result,
                     contours,
                     color=(255, 255, 255))
        
        self.variations[Image.fill_polly_key] = result
    
    def bounding_rect(self, source, min, max):
        result = self.variations[Image.org_color_key].copy()
        
        contours, __ = cv2.findContours(self.variations[source],
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_NONE)

        for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                # really small rectangles
                if w < min and h < min:
                    continue
                # too big 
                if w > max and h > max:
                    continue
                cv2.rectangle(result,
                            (x, y),
                            (x + w, y + h),
                            (0, 0, 255),
                            4)
            
        self.variations[Image.brect_key] = result

    def plt_final_result(self):
        plt.suptitle(self.title, size=16)
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

    def open_images(self):
        """
        """
        for i in self.images_numbers:
            title = self.prefix + str(i)
            path = IMAGES_PATH + title + self.suffix

            img_color = cv2.imread(path, cv2.IMREAD_COLOR)

            if img_color is None:
                print(f"Failed to read image, check the path: {path}")
            else:
                img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
                image = Image(title, path)
                image.variations[Image.org_color_key] = img_color
                image.variations[Image.org_gray_key] = img_gray
                self.images.append(image)

    def plt_images_variations(self):
        for img in self.images:
            img.plt_variations()
            
    def apply_gaussian_blur(self, kernel_size, source):
            for img in self.images:
                img.gaussian_blur(kernel_size, source)

    def apply_meddian_blur(self, kernel_size, source):
        for img in self.images:
            img.meddian_blur(kernel_size, source)

    def apply_threshold(self, source, block_size):
        for img in self.images:
            img.threshold(source, block_size)

    def apply_laplacian_derivatives(self, source):
        for img in self.images:
            img.laplacian_derivatives(source)

    def apply_canny_edges(self, source):
        for img in self.images:
            img.canny_edges(source)
            
    def apply_find_and_fill_contours(self, source):
        for img in self.images:
            img.find_and_fill_contours(source)
    
    def apply_morph_rect_open(self, source, kernel_size):
        for img in self.images:
            img.morph_rect_open(source, kernel_size)
        
    def apply_morph_cross_open(self, source, kernel_size):
        for img in self.images:
            img.morph_cross_open(source, kernel_size)
            
    def apply_morph_cross_close(self, source, kernel_size):
        for img in self.images:
            img.morph_cross_close(source, kernel_size)

    def apply_morph_rect_close(self, source, kernel_size):
        for img in self.images:
            img.morph_rect_close(source, kernel_size)
            
    def apply_fill_poly(self, source):
        for img in self.images:
            img.fill_poly(source)
                    
    def apply_bounding_rect(self, source, min, max):
        for img in self.images:
            img.bounding_rect(source, min, max)
        
    def plt_images_final_results(self):
        for img in self.images:
            img.plt_final_result()
        
        
def main():
    print("Please be patience, this pipline may take a while..")
    start = timeit.default_timer()

    images_numbers = range(1, 8)
    # images_numbers = [5]
    process = Process(images_numbers, prefix='image', suffix='.jpg')
    process.open_images()
    
    process.apply_meddian_blur(source=Image.org_gray_key, kernel_size=15)
    process.apply_threshold(source=Image.mblur_key, block_size=601)
    
    process.apply_find_and_fill_contours(source=Image.thresh_key)
    process.apply_morph_cross_close(source=Image.fcontours_key, kernel_size= 11)
    process.apply_morph_cross_open(source=Image.mclose_cross_key, kernel_size=3)
    
    process.apply_find_and_fill_contours(source=Image.mopen_cross_key)
    process.apply_morph_cross_close(source=Image.fcontours_key, kernel_size=11)
    process.apply_morph_cross_open(source=Image.mclose_cross_key, kernel_size=3)
    
    process.apply_bounding_rect(source=Image.mopen_cross_key, min = 40, max = 3000)
    
    stop = timeit.default_timer()
    print(f"Pipline finished! time: {stop - start} seconds")
   
    # process.plt_images_final_results()
    process.plt_images_variations()


if __name__ == '__main__':
    main()
