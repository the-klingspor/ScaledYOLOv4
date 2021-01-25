import os
from pathlib import Path


def split_dataset(path, val_size=0.1, test_size=0.2):
    """
    Splits the dataset given by the path into test, train, validation sets and a
    set with all unlabeled images. Images are split based on their scenes, so
    all images of the same scene also end up in the same set.

    :param path: String, path of a directory
    :param val_size: Float, default=0.1
        The amount of scenes of labeled data that are used as validation set.
    :param test_size: Float, default=0.2
        The amount of scenes of labeled data that are used as test set.
    :return: None
    """
    labeled, unlabeled = order_labeled(path)

    val_size = round(val_size)
    test_size = round(test_size)

    train_dirs = []
    val_dirs = []
    test_dirs = []


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
        print(filename)
        extension = filename.suffix
        if extension == '.png' or extension == '.jpg':
            image_found = True
        else:
            index += 1

    if image_found:
        filename_label = filename.with_suffix('.txt')
        path_label = os.path.join(path, filename_label)
        if os.path.isfile(path_label):
            print(True)
            return True
    print(False)
    return False


if __name__ == '__main__':
    path_unlabeled = '/home/joschi/cvmlfs/data/insect_camera_trap/labeled/dynamic_vision/P16_seq8/'
    path_labeled = '/home/joschi/cvmlfs/data/insect_camera_trap/labeled/dynamic_vision/Bilder_V15/'

    print(is_labeled(path_labeled))
    print(is_labeled(path_unlabeled))

    path = '/home/joschi/cvmlfs/data/insect_camera_trap/labeled/dynamic_vision/'

    labeled, unlabeled = order_labeled(path)
    print(len(labeled))
    print(len(unlabeled))


