import os
import cv2 as cv
import numpy as np

class PencilFilter:
    def __init__(self, dilatation_size=2, dilation_shape=cv.MORPH_ELLIPSE):
        self.dilatation_size = dilatation_size
        self.dilation_shape = dilation_shape

    def apply(self, img):
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        dilated = self.dilatation(gray_img.copy())
        gray_img_32_bit = gray_img.copy().astype(np.uint32)
        dilated_my = ((gray_img_32_bit * 255) / dilated).astype(np.uint8)
        penciled = np.where(np.isnan(dilated_my), 255, dilated_my).astype(np.uint8)
        penciled_rgb = cv.cvtColor(penciled, cv.COLOR_GRAY2RGB)
        return penciled_rgb

    def dilatation(self, img):
        element = cv.getStructuringElement(
            self.dilation_shape,
            (2 * self.dilatation_size + 1, 2 * self.dilatation_size + 1),
            (self.dilatation_size, self.dilatation_size)
        )
        dilatation_dst = cv.dilate(img, element)
        return dilatation_dst

def apply_pencil_filter_to_folder(folder_path):
    subfolders = ['test', 'train', 'valid']
    filter = PencilFilter()

    for subfolder in subfolders:
        input_folder = os.path.join(folder_path, subfolder, 'images')
        output_folder = os.path.join(folder_path, subfolder, 'pencil_images')
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)

                img = cv.imread(input_path)
                if img is not None:
                    penciled_img = filter.apply(img)
                    cv.imwrite(output_path, penciled_img)
                else:
                    print(f"Could not read image {input_path}")

# Example usage
folder_path = '../../yolo_dataset/0807'
apply_pencil_filter_to_folder(folder_path)
