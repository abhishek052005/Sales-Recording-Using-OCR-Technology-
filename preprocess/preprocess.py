import cv2
import numpy as np
import os

def deskew(image):
    """Automatically straighten the image"""

    coords = np.column_stack(np.where(image < 255))

    if len(coords) == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = 90 + angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated



def preprocess_image(input_path, output_path):
    image = cv2.imread(input_path)
    if image is None:
        raise Exception(f"cannot read image:{input_path}")

    # Resize
    image = cv2.resize(
        image,None,fx=2,fy=2,interpolation=cv2.resize
    )
    #gray
    gray = cv2.cvtcolor(image, cv2.COLOR_BGR2GRAY)

    #CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0, titleGridSize=(8,8)
    )
    gray = clahe.apply(gray)

    # Denoise
    gray = cv2.fastNlMeansDenoising(
        gray,None,h=12
    )


    # Sharpen
    kernel = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])
    gray = cv2.filter2D(gray,-1,kernel)

    # Threshold
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8
    )

    # Morphology
    kernel = np.ones((2,2),np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    thresh = cv2.dilate(
        thresh,
        kernel,
        iterations=1
    )

    
    # Deskew
    thresh = deskew(thresh)
    gray = deskew(gray)



    # File Names

    base = os.path.splitext(output_path)[0]

    threshold_path = base + "_thresh.png"
    gray_path = base + "_gray.png"

    cv2.imwrite(threshold_path, thresh)
    cv2.imwrite(gray_path, gray)

    print("Gray Image :", gray_path)
    print("Threshold :", threshold_path)

    return {
        "gray": gray_path,
        "threshold": threshold_path,
        "original": input_path
    }
input_folder = "bill_image"
output_folder = "image_cleaning_one_folder"

if __name__ == "__main__":
    preprocess_image(input_folder,output_folder)