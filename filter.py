"""
LINK THAM KHẢO:
- https://www.analyticsvidhya.com/blog/2021/07/an-interesting-opencv-application-creating-filters-like-
instagram-and-picsart/?fbclid=IwAR1-gvPGQZqG00iCCN1M-8eELe9nbaDW4aiyJKSczlfm4p5p987ghn68Zoo
1. GreyScale Filter                             8. Summer Effect Filter
2. Sharp Effect                                 9. Winter Effect Filter
3. Sepia Filter                                 --- K nằm trong link trên---
4. Pencil Sketch Effect: GreyScale              10. Cartoon
5. Pencil Sketch Effect: Colour                 11.
6. HDR effect
7. Invert Filter

- kernel work?
  ~ link: https://www.geeksforgeeks.org/python-opencv-filter2d-function/
  ~ link: https://en.wikipedia.org/wiki/Kernel_(image_processing)
"""

import cv2
import numpy as np
import scipy
from scipy.interpolate import UnivariateSpline
import sys


LIST_FILTER = ["None", "greyscale_filter", "sharpen", "sepia", "pencil_sketch_grey",
               "pencil_sketch_col", "HDR", "invert", "summer", "winter", "cartoon","gaussian_blur"]


def LookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))


def greyscale_filter(image):
    """
    GreyScale Filter
    - The greyscale filter is used to give a Black and White effect to images.
    Basically coloured components from the image are removed.
    - Using cv2.cvtColor():
        ~ Công thức chuyển sang xám: "RGB [A] to Gray: Y ← 0.299⋅R + 0.587⋅G + 0.114⋅B"
        ~ Link tham khảo: https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor
    """
    greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.merge((greyscale, greyscale, greyscale))


def sharpen(img):
    """
    Sharp Effect
    - The kernel for the sharpening effect will be : [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]
    *ngoài ra có 1 số kernel khác có thể dùng như: [[0, -1, 0],[-1, 5,-1],[0, -1, 0]]
    """
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_sharpen = cv2.filter2D(img, -1, kernel)
    return img_sharpen


def sepia(img):
    """
    Sepia Filter
    - Sepia is one of the most commonly used filters in image editing. Sepia adds a warm brown effect to photos.
    A vintage, calm and nostalgic effect is added to images.
    - Công thức:
        newR = (R * 0.393 + G * 0.769 + B * 0.189)
        newG = (R * 0.349 + G * 0.686 + B * 0.168)
        newB = (R * 0.272 + G * 0.534 + B * 0.131)
    - link tham khao: https://www.yabirgb.com/sepia_filter/

    """
    # converting to float to prevent loss
    img_sepia = np.array(img, dtype=np.float64)
    # multipying image with special sepia matrix
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                                    [0.349, 0.686, 0.168],
                                                    [0.393, 0.769, 0.189]]))
    # normalizing values greater than 255 to 255
    img_sepia[np.where(img_sepia > 255)] = 255
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    return img_sepia


def pencil_sketch_grey(img):
    """
    Pencil Sketch Effect: GreyScale
    - cv2.pencilSketch():
    """
    # inbuilt function to create sketch effect in colour and greyscale
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    return cv2.merge((sk_gray, sk_gray, sk_gray))


def pencil_sketch_col(img):
    """
    Pencil Sketch Effect: Color
    - tương tự trên
    """
    # inbuilt function to create sketch effect in colour and greyscale
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    return sk_color


def HDR(img):
    """
    HDR effect:
    - The HDR effect is used a lot as it increases the level of detail in images.
    I shall use the cv2.detailEnhance() to implement this.
    - cv2.detailEnhance():
      ~ https://stackoverflow.com/questions/45142120/the-meaning-of-sigma-s-and-sigma-r-in-detailenhance
      -function-on-opencv
      ~ https://docs.opencv.org/3.0-beta/modules/photo/doc/npr.html#detailenhance
    """
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr


def invert(img):
    """
    Invert Filter
    - Đảo màu: lấy 255-giá trị pixel
    - sử dụng: cv2.bitwise_not()
    *Assume the value of a pixel is 200(10)=11001000(2); then the bitwise_not of that simply is 00110111b=55
    """
    inv = cv2.bitwise_not(img)
    return inv


def summer(img):
    """
    Summer Effect Filter
    - Tăng kênh đỏ, giảm kênh blue
    """
    increase_lookup_table = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decrease_lookup_table = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel, red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increase_lookup_table).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decrease_lookup_table).astype(np.uint8)
    sum = cv2.merge((blue_channel, green_channel, red_channel))
    return sum


def winter(img):
    """
    Summer Effect Filter
    - Tăng kênh blue, giảm kênh đỏ
    """
    increase_lookup_table = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decrease_lookup_table = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel, red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decrease_lookup_table).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increase_lookup_table).astype(np.uint8)
    win = cv2.merge((blue_channel, green_channel, red_channel))
    return win


def cartoon(img):
    """
    INFO:
    - link 1: https://datahacker.rs/002-opencv-projects-how-to-cartoonize-an-image-with-opencv-in-python/
    - link 2: https://datahacker.rs/007-color-quantization-using-k-means-clustering/#Color-quantization
    *Su dung cach o link 2
    """
    K = 9
    # Defining input data for clustering
    data = np.float32(img).reshape((-1, 3))
    # Defining criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # Applying cv2.kmeans function
    ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result


def brightness_contrast(img, gamma):
    """
    Điều chỉnh độ sáng ảnh: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
    """
    look_up_table = np.empty((1, 256), np.uint8)
    for i in range(256):
        look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(img, look_up_table)
    return res


def vignette(img, v_value):
    """
    What is Vignette filter?
    - The Vignette filter is generally used to focus viewer attention on certain parts of the image without hiding
    other parts completely. Generally focused part has higher brightness and saturation and brightness and saturation
    decrease as we go radially out from center to periphery.
    - link:
        https://www.geeksforgeeks.org/create-a-vignette-filter-using-python-opencv/
        https://www.youtube.com/watch?v=9AaRXGFP-C0
    """

    if v_value is None:
        return img

    # Extracting the height and width of an image
    rows, cols = img.shape[:2]
    value = 1
    mask = np.zeros((int(rows * (value * 0.1 + 1)), int(cols * (value * 0.1 + 1))))
    # generating vignette mask using Gaussian
    # resultant_kernels
    # generating vignette mask using Gaussian kernels
    kernel_x = cv2.getGaussianKernel(int(cols * (0.1 * value + 1)), v_value)
    kernel_y = cv2.getGaussianKernel(int(rows * (0.1 * value + 1)), v_value)
    kernel = kernel_y * kernel_x.T

    # Normalizing the kernel
    kernel = kernel / np.linalg.norm(kernel)

    # Genrating a mask to image
    mask = 255 * kernel
    output = np.copy(img)
    # applying the mask to each channel in the input image
    mask_imposed = mask[int(0.1 * value * rows):, int(0.1 * value * cols):]
    for i in range(3):
        output[:, :, i] = output[:, :, i] * mask_imposed
    return output


def gaussian_blur(image, k_value):
    """
    Làm mờ ảnh:
    - https://docs.opencv.org/3.4/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
    - https://www.geeksforgeeks.org/python-opencv-smoothing-and-blurring/
    - https://www.geeksforgeeks.org/what-is-image-blurring/
    """
    if k_value is None:
        return image
    image = cv2.GaussianBlur(image, (2 * k_value + 1, 2 * k_value + 1), 0)
    return image


def change_saturation(img, saturation):
    """
    Chỉnh độ bão hòa (saturation)
    - https://www.programmerall.com/article/5684321533/
    """
    MAX_VALUE = 100
    # Load picture Read the color image normalized and converted to floating point
    image = img.astype(np.float32) / 255.0
    # Color space conversion BGR to HLS
    hlsImg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hlsImg[:, :, 2] = (1.0 + saturation / float(MAX_VALUE)) * hlsImg[:, :, 2]
    hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
    # HLS2BGR
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
    lsImg = lsImg.astype(np.uint8)
    return lsImg


"""BLENDING IMAGE"""


def blending_image(image, flag):
    h, w = image.shape[:2]
    s_path = sys.path[0]
    img_path = None
    check = True
    if flag == 0:
        return image
    if flag % 2 == 0:
        img_path = s_path + "\\resource\old" + str(int(flag/2)) + "n.jpg"
    else:
        img_path = s_path + "\\resource\old" + str(int(flag/2) + 1) + ".jpg"
        check = False

    img = cv2.imread(img_path)
    print(img)
    img = cv2.resize(img, (int(w), int(h)))
    if check:
        add = cv2.add(img, image)
        return add
    else:
        add = cv2.addWeighted(image, 0.7, img, 0.3, 0)
        return add


# import sys
# s = sys.path[0] + "\\resource\OIP.jpg"
# img = cv2.imread(s)
# img = cv2.GaussianBlur(img, (5, 5), 0)
# cv2.imshow("haha", img)
# cv2.waitKey(0)
# cv2.destroyAllWindow()

#
#
# def changeRadius(value):
#     global radius
#     radius = value
#
#
# # For changing the focus of the mask
# def changeFocus(scope):
#     global value
#     value = scope
#
#
# # Reading the image and getting properties
# img = cv2.imread(s)
# rows, cols = img.shape[:2]
# value = 1
# radius = 130
# mask = np.zeros((int(rows * (value * 0.1 + 1)), int(cols * (value * 0.1 + 1))))
#
# cv2.namedWindow('Trackbars')
# cv2.createTrackbar('Radius', 'Trackbars', 130, 500, changeRadius)
# cv2.createTrackbar('Focus', 'Trackbars', 1, 10, changeFocus)
#
# while (True):
#     # generating vignette mask using Gaussian kernels
#     kernel_x = cv2.getGaussianKernel(int(cols * (0.1 * value + 1)), radius)
#     kernel_y = cv2.getGaussianKernel(int(rows * (0.1 * value + 1)), radius)
#     kernel = kernel_y * kernel_x.T
#
#     # Normalizing the kernel
#     kernel = kernel / np.linalg.norm(kernel)
#
#     # Genrating a mask to image
#     mask = 255 * kernel
#     output = np.copy(img)
#     # applying the mask to each channel in the input image
#     mask_imposed = mask[int(0.1 * value * rows):, int(0.1 * value * cols):]
#     for i in range(3):
#         output[:, :, i] = output[:, :, i] * mask_imposed
#     cv2.imshow('Original', img)
#     cv2.imshow('Vignette', output)
#     key = cv2.waitKey(50)
