import cv2

HEIGHT = 360
WIDTH = 540


def check_size(img_width, img_height):
    if img_width > WIDTH and img_height <= HEIGHT:
        return 1
    if img_width <= WIDTH and img_height > HEIGHT:
        return 2
    if img_width > WIDTH and img_height > HEIGHT:
        return 3
    return 0


def resize(image):
    w, h = float(image.shape[1]), float(image.shape[0])
    aspect_ratio = w / h
    s = 9
    while s != 0:
        s = check_size(w, h)
        if s == 1:
            w = WIDTH
            h = w/aspect_ratio
        if s == 2 or s == 3:
            h = HEIGHT
            w = h*aspect_ratio
    image = cv2.resize(image, (int(w), int(h)))
    return image


# def ismodulefunction(module, member):
#     module_member = getattr(module, member)
#     member_type = type(module_member).__name__
#     return member_type == "function" or member_type == "builtin_function_or_method"
