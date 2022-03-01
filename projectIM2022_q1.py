import cv2
import timeit
from image_class import *
from typing import List

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
    start = timeit.default_timer()

    for img in images:
        print(f"starting {img.title}")

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

        print(f"finished {img.title}")
        
        # img.plt_variations()
        img.plt_final_result()

    stop = timeit.default_timer()
    print(f"Pipline finished! time: {stop - start} seconds")


def main():
    """_summary_
    """
    images_numbers = range(1, 8)
    open_images(images_numbers, prefix='image', suffix='.jpg')
    find_pieces()


if __name__ == '__main__':
    main()
