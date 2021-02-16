import os
import random

from pathlib import Path
from shutil import copy2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def split_dataset(path, val_size=0.1, test_size=0.25, split_mode="seq",
                  shuffle_img=True, seed=None):
    """
    Splits the dataset given by the path into test, train, validation sets and a
    set with all unlabeled images. There are two modes to split images:
    1) By scene: Images are split based on their scenes, so all images of the
        same scene also end up in the same set.
    2) Sequential: Every scene is split into contiguous sets based on their
        split ratios. The first part is used for training, second for validation
        and the last one for testing.

    :param path: String, path of a directory
    :param val_size: Float, default=0.10
        The amount of scenes of labeled data that are used as validation set.
    :param test_size: Float, default=0.25
        The amount of scenes of labeled data that are used as test set.
    :param split_mode: String, default="seq"
        Split mode to decide whether to split the data into contigous sequences
        ("seq") or based on complete scenes ("scene")
    :param shuffle_img: Boolean, default=True
        Whether or not to shuffle the img paths.
    :param seed: Int, default=None
        The seed to use for shuffling.
    :return: None
    """
    labeled, unlabeled = order_labeled(path)

    train_filename = os.path.join(path, "train.txt")
    val_filename = os.path.join(path, "val.txt")
    test_filename = os.path.join(path, "test.txt")
    unlabeled_filename = os.path.join(path, "unlabeled.txt")

    write_img_list(unlabeled, unlabeled_filename)
    if split_mode == "scene":
        n_val = round(val_size * len(labeled))
        n_test = round(test_size * len(labeled))

        random.shuffle(labeled)
        test_dirs = labeled[:n_test]
        val_dirs = labeled[n_test:(n_test + n_val)]
        train_dirs = labeled[(n_val + n_test):]

        write_img_list(train_dirs, train_filename)
        write_img_list(val_dirs, val_filename)
        write_img_list(test_dirs, test_filename)
    elif split_mode == "seq":
        train_list = []
        val_list = []
        test_list = []
        for dir in labeled:
            img_list = []
            for file in sorted(os.listdir(dir)):
                # set path relative to yolo by removing the first two
                # directories and append parent directory by '..'
                # todo: find a nicer way to do this, refactor with
                file = Path(os.path.join(dir, file))
                extension = file.suffix
                relative_path = Path(*file.parts[-7:])
                relative_path = os.path.join('..', relative_path)
                if extension == '.png' or extension == '.jpg':
                    img_list.append(relative_path)
            n_img = len(img_list)
            index_val = int((1.0 - (val_size + test_size)) * n_img)  # starts after the last train img
            index_test = index_val + int(val_size * n_img)  # starts after the last test img

            train_list.extend(img_list[:index_val])
            val_list.extend(img_list[index_val:index_test])
            test_list.extend(img_list[index_test:])

        if shuffle_img:
            train_list = shuffle(train_list, random_state=seed)
            val_list = shuffle(val_list, random_state=seed)
            test_list = shuffle(test_list, random_state=seed)

        print("Train images: {0}, validation images: {1}, test images: {2}"
              .format(len(train_list), len(val_list), len(test_list)))

        # todo in Funktion auslagern
        with open(train_filename, 'w') as train_file:
            for img in train_list:
                train_file.write(img+'\n')
        with open(val_filename, 'w') as val_file:
            for img in val_list:
                val_file.write(img+'\n')
        with open(test_filename,  'w') as test_file:
            for img in test_list:
                test_file.write(img+'\n')
    else:
        raise ValueError("Split mode must be 'seq' or 'scene', but was {}"
                         .format(split_mode))


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
                relative_path = Path(*file.parts[:-7])
                relative_path = os.path.join('..', relative_path)
                if extension == '.png' or extension == '.jpg':
                    img_paths.write(relative_path+'\n')


def change_paths(input, output, to_replace, replacement):
    """
    Takes paths to an input and to an output path and replaces the 'to_replace'
    part from the img paths of input with the 'replacement' and write the
    modified paths into the output txt file.

    :param input: String, path to a txt file with img paths
    :param output: String, path
    :param to_replace: String, part of the img paths of input that are to be
        replaced.
    :param replacement: String, part of the new img paths that replace the
        'to_replace' parts
    :return: None
    """
    with open(input) as input_file:
        paths = input_file.readlines()

    # change 'to_replace' parts to the specified replacement, e.g.
    # '.../dynamic_vision/...' to '.../raw/...'
    paths = [x.replace(to_replace, replacement) for x in paths]

    with open(output, 'w') as output_file:
        for path in paths:
            output_file.write(path)


if __name__ == '__main__':
    path = '/home/joschi/Documents/Studium/Masterarbeit/cvmlfs/data/insect_camera_trap/labeled/dynamic_vision/'
    split_dataset(path, split_mode='seq', shuffle_img=True)

    """
    # Load broken files from remote network
    source_prefix = "/home/joschi/cvmlfs/data/insect_camera_trap/labeled/dynamic_vision/"
    dest_prefix = "/home/joschi/Documents/Studium/Masterarbeit/cvmlfs/data/insect_camera_trap/labeled/dynamic_vision/"

    path_list = []
    broken_files = "/home/joschi/Documents/Studium/Masterarbeit/cvmlfs/data/insect_camera_trap/labeled/dynamic_vision/broken.txt"
    with open(broken_files) as file_broken:
        for line in file_broken:
            path_list.append(str.rstrip(line))

    source_list = [os.path.join(source_prefix, path) for path in path_list]
    dest_list = [os.path.join(dest_prefix, path) for path in path_list]

    for i in range(len(source_list)):
        print(dest_list[i])
        if os.path.isfile(dest_list[i]):
            os.remove(dest_list[i])
        copy2(source_list[i], dest_list[i])
    """


