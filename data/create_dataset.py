import os
import random
from pathlib import Path


def split_dataset(path, val_size=0.1, test_size=0.25):
    """
    Splits the dataset given by the path into test, train, validation sets and a
    set with all unlabeled images. Images are split based on their scenes, so
    all images of the same scene also end up in the same set.

    :param path: String, path of a directory
    :param val_size: Float, default=0.10
        The amount of scenes of labeled data that are used as validation set.
    :param test_size: Float, default=0.25
        The amount of scenes of labeled data that are used as test set.
    :return: None
    """
    labeled, unlabeled = order_labeled(path)

    n_val = round(val_size * len(labeled))
    n_test = round(test_size * len(labeled))

    random.shuffle(labeled)
    val_dirs = labeled[:n_val]
    test_dirs = labeled[n_val:(n_val+n_test)]
    train_dirs = labeled[(n_val+n_test):]

    print(len(val_dirs))
    print(len(test_dirs))
    print(len(train_dirs))
    print(val_dirs)

    train_filename = os.path.join(path, "train.txt")
    val_filename = os.path.join(path, "val.txt")
    test_filename = os.path.join(path, "test.txt")
    unlabeled_filename = os.path.join(path, "unlabeled.txt")

    write_img_list(train_dirs, train_filename)
    write_img_list(val_dirs, val_filename)
    write_img_list(test_dirs, test_filename)
    write_img_list(unlabeled, unlabeled_filename)


def order_labeled(path):
    """
    Filters subdirectories of the given path on whether they contain labeled or
    unlabeled images.

    :param path: String, path of a directory
    :return: (filtered, unfiltered), both are lists of strings containing
        subdirectories of the given path.
    """
    dir_list = [os.path.join(path, x) for x in os.listdir(path)]
    subdirs = [x for x in dir_list if os.path.isdir(x)]

    labeled = [x for x in subdirs if is_labeled(x)]
    unlabeled = [x for x in subdirs if x not in labeled]

    return labeled, unlabeled


def is_labeled(path):
    """
    Tests whether a path contains labeled or unlabeled images. Decides based on
    whether or not the first found image has a .txt file containing its label.

    :param path: String, path of a directory
    :return: Boolean, True if the images are labeled, false if not.
    """
    files = os.listdir(path)
    image_found = False
    index = 0
    filename = None

    while (not image_found) and index < len(files):
        filename = Path(files[index])
        extension = filename.suffix
        if extension == '.png' or extension == '.jpg':
            image_found = True
        else:
            index += 1

    if image_found:
        filename_label = filename.with_suffix('.txt')
        path_label = os.path.join(path, filename_label)
        if os.path.isfile(path_label):
            return True
    return False


def write_img_list(dir_list, filename):
    """
    Writes the paths of all images in the list of directories into the new text
    file. The paths have to be relative to the file that is used to train the
    network and we assume the data set shares a parent directory with this file.

    :param dir_list: List of strings, containing the directories with images
        whose paths should be saved in the specified file.
    :param filename: String, path of the new .txt file containing the image
        paths.
    :return: None
    """
    # todo rewrite deep indentation
    with open(filename, 'w') as img_paths:
        for dir in dir_list:
            for file in os.listdir(dir):
                # set path relative to yolo by removing the first two
                # directories and append parent directory by '..'
                # todo: find a nicer way to do this
                file = Path(os.path.join(dir, file))
                extension = file.suffix
                relative_path = Path(*file.parts[3:])
                relative_path = os.path.join('..', relative_path)
                if extension == '.png' or extension == '.jpg':
                    img_paths.write(relative_path+'\n')


if __name__ == '__main__':
    #path_unlabeled = '/home/joschi/cvmlfs/data/insect_camera_trap/labeled/dynamic_vision/P16_seq8/'
    #path_labeled = '/home/joschi/cvmlfs/data/insect_camera_trap/labeled/dynamic_vision/Bilder_V15/'

    #print(is_labeled(path_labeled))
    #print(is_labeled(path_unlabeled))

    path = '/home/joschi/cvmlfs/data/insect_camera_trap/labeled/dynamic_vision/'

    #labeled, unlabeled = order_labeled(path)
    #print(len(labeled))
    #print(len(unlabeled))

    split_dataset(path)
