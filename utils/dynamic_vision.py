import cv2
import numpy as np
import os
from shutil import copy

# TODO: Abseil flags for command line tool.


def create_dvs_directory(input_dir, output_dir, copy_non_img=True, thresh=40):
    """
    Create a dynamic vision simulation image for every image in the input
    directory and write it to the output directory. Use the given threshold
    for the dvs simulation.

    :author: Joschi
    :param input_dir: Path of the input directory, string.
    :param output_dir: Path of  the output directory, string.
    :param copy_non_img: Whether to copy all non-image files, such as txt or
        JSON, to the output directory. Boolean, default=True.
    :param thresh: Threshold between setting detected motion to max or min
        value. Integer between 0 and 255, default=40.
    :return: None
    """
    img_paths = []
    other_paths = []

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # filter all files into image and non-image objects
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if file.endswith(".png") or file.endswith(".jpg"):
            img_paths.append(file_path)
        else:
            other_paths.append(file_path)
    img_paths.sort()
    # create and save dvs images
    n_images = len(img_paths)
    for i in range(0, n_images):
        img_path_1 = img_paths[i]
        # last image has no successor, use predecessor instead
        if i != n_images - 1:
            img_path_2 = img_paths[i+1]
        else:
            img_path_2 = img_paths[i-1]
        filename = os.path.basename(img_path_1)
        dvs_path = os.path.join(output_dir, filename)
        create_dvs(img_path_1, img_path_2, dvs_path, thresh)

    if copy_non_img:
        for file in other_paths:
            filename = os.path.basename(file)
            output_file = os.path.join(output_dir, filename)
            copy(file, output_file)


def create_dvs(image_path_1, image_path_2, dvs_path, thresh=40):
    """
    Create a dynamic vision simulation based on two images given by their paths
    and a threshold. The threshold decides for each pixel, if the motion is set
    to a maximum or minimum value. This motion map replaces the value channel of
    the images in the HSV format. The resulting dynamic vision simulation is
    saved in the given dvs_path.

    :author: Marvin
    :param image_path_1: Path of the first image, string.
    :param image_path_2: Path of the second image, string.
    :param dvs_path: Path of the output, string.
    :param thresh: Threshold between setting detected motion to max or min
        value. Integer between 0 and 255, default=40.
    :return: None
    """
    img1 = cv2.imread(image_path_1)
    img2 = cv2.imread(image_path_2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray1 = np.array(gray1, dtype=np.int16)
    gray2 = np.array(gray2, dtype=np.int16)
    dvs = np.zeros_like(gray1, dtype=np.int16)
    dvs = abs(gray1 - gray2)
    dvs[dvs > thresh] = 255
    dvs[dvs <= thresh] = 0
    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    value[dvs == 0] = value[dvs == 0] * 0.3
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(dvs_path, bgr)


if __name__ == '__main__':
    input_paths = [
        "/home/joschi/cvmlfs/data/insect_camera_trap/labeled/raw/Bilder_V5",
    ]

    for i in range(34, 54):
        path = "/home/joschi/cvmlfs/data/insect_camera_trap/labeled/raw/Bilder_V"+str(i)
        input_paths.append(path)

    suffixes = [
        "P1_seq1",
        "P2_seq1",
        "P2_seq2",
        "P2_seq3",
        "P2_seq4",
        "P2_seq5",
        "P3_seq1",
        "P4_seq1",
        "P5_seq1",
        "P5_seq2",
        "P5_seq3",
        "P5_seq4",
        "P6_seq1",
        "P6_seq2",
        "P7",
        "P8_seq1",
        "P8_seq2",
        "P9_seq1",
        "P9_seq2",
        "P9_seq3",
        "P9_seq4",
        "P9_seq5",
        "P10_seq1",
        "P10_seq2",
        "P10_seq3",
        "P11_seq1",
        "P11_seq2",
        "P11_seq3",
        "P12_seq1",
        "P12_seq2",
        "P12_seq3",
        "P12_seq4",
        "P13_seq1",
        "P13_seq2",
        "P14_seq1",
        "P14_seq2",
        "P15_seq1",
        "P15_seq2",
        "P15_seq3",
        "P15_seq4",
        "P15_seq5",
        "P15_seq6",
        "P15_seq7",
        "P15_seq8",
        "P15_seq9",
        "P15_seq10",
        "P15_seq11",
        "P15_seq12",
        "P15_seq13",
        "P15_seq14",
        "P15_seq15",
        "P15_seq16",
        "P16_seq1",
        "P16_seq2",
        "P16_seq3",
        "P16_seq4",
        "P16_seq5",
        "P16_seq6",
        "P16_seq7",
        "P16_seq8",
        "P16_seq9",
        "P16_seq10",
        "P17_seq1",
        "P17_seq2",
        "P17_seq3",
        "P17_seq4"
    ]

    for suffix in suffixes:
        path = "/home/joschi/cvmlfs/data/insect_camera_trap/labeled/raw/" + suffix
        input_paths.append(path)

    print(input_paths)

    for i in range(len(input_paths)):
        input_path = input_paths[i]
        output_path = input_path.replace("/raw/", "/dynamic_vision/")
        print(input_path, output_path)
        create_dvs_directory(input_path, output_path)
