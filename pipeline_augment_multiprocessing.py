import os
import cv2
import numpy as np
import albumentations as A
import argparse
import yaml
import shutil
import concurrent.futures
import time
from distutils.dir_util import copy_tree
from tqdm import tqdm

import gc


def format_bboxes(annotations):
    """
    Process the annotations.
    Input -> [category_id x_centre y_centre width height]
    Output -> [category_id] [x_centre y_centre width height]
    """

    for coords in annotations:
        annot = np.array(coords).astype(np.float32)

        if annot.size == 0:
            pass
        else:
            old_bboxes = np.delete(annot, 0, axis = 1)
            category_ids = np.delete(annot, [1, 2, 3, 4], axis = 1)

            bboxes = []

            for arr in old_bboxes:
                x_centre = arr[0]
                y_centre = arr[1]
                width = arr[2]
                height = arr[3]

                bboxes.append(x_centre)
                bboxes.append(y_centre)
                bboxes.append(width)
                bboxes.append(height)
            
            bboxes = np.array(bboxes).reshape(-1, 4)
            category_ids = np.array(category_ids).astype(int)

            bboxes = bboxes.tolist()
            category_ids = category_ids.ravel()
            category_ids = category_ids.tolist()

        return bboxes, category_ids



def augment(args, directory = "./dataset/."):
    """
    Augment images randomly from the list of transformations
    Input -> number of augmentations per image
    Output -> Augmented images an corresponding annotations
    """

    # Create lists of images to augment and to store the annotations
    imgs_list = []
    annotations = []
    
    for dir in next(os.walk(directory))[1]:
        imgs_list.clear()
        annotations.clear()

        for i in tqdm(os.listdir(f"./dataset/{dir}/images"), desc = f"Directory: {dir}"):   # Read the images in the directory
            imgs_list.append(i)     # Store the images list in the list defined above
            img_name = os.path.splitext(i)      # Remove the extension of the image

            with open(f"./dataset/{dir}/labels/{img_name[0]}.txt") as f:    # Open the annotations file with the same filename
                lines = [line.split() for line in f]
                annotations.append(lines)       

            bboxes, category_ids = format_bboxes(annotations)   # Format the Bounding boxes

            if dir == "train":      # Augment images only in the train directory, skip the others
                for img in imgs_list:
                    image = cv2.imread(f"./dataset/{dir}/images/" + img)    # Read the image as an array

                transform = A.Compose(transforms, bbox_params = A.BboxParams(format = 'yolo', label_fields = ['category_ids']))     # Define the augmentations, annotations format and the labels

                transformed = transform(image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB), bboxes = bboxes, category_ids = category_ids)   # Transform the images

                for aug in range(args):     # Create n transformations per image and save them
                    cv2.imwrite(f"./dataset/{dir}/images/" + str(aug) + "_" + i, transformed["image"])

                    # Format the augmented bounding boxes correctly and save them in the labels folder
                    transformed_bbox = np.array(transformed["bboxes"]).astype(np.float32)
                    transformed_ids = np.array(transformed["category_ids"]).reshape(-1, 1).astype(str)

                    transformed_annotations = np.hstack((transformed_ids, transformed_bbox))

                    with open(f"./dataset/{dir}/labels/" + str(aug) + "_" + os.path.splitext(i)[0] + ".txt", "w") as txt:
                        for line in transformed_annotations:
                            txt.write(" ".join(line) + "\n")
            else:
                pass



# List of transformations to apply
transforms = [A.Resize(width = 416, height = 416),
              A.RandomCrop(width = 416, height = 416),
              A.Rotate(limit = 40, p = 0.9, border_mode = cv2.BORDER_CONSTANT),
              A.HorizontalFlip(p = 0.5),
              A.VerticalFlip(p = 0.1),
              A.RGBShift(r_shift_limit = 25, g_shift_limit = 25, b_shift_limit = 25, p = 0.9)]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type = int, default = 3, help = "Number of augmentation per image")
    args = parser.parse_args()

    # Check if diretory exists
    if not os.path.isdir("./dataset"):
        os.mkdir("./dataset")

    # If directory is not empty, skip, else copy all files from the resized dataset
    if len(os.listdir("./dataset")) == 0:
        copy_tree("./resized_dataset", "./dataset")
        print("Successfully copied all files to the Dataset folder")
    else:
        print("Directory is not empty. Skipping...")
        pass
    print("\n")

    # Count the number of images in the resized dataset (images before augmentation)
    print("Before Augmentation:")
    for path, dirs, files in os.walk("./resized_dataset/train/images"):
        print(f"Number of images in the train folder: {len(files)}")
    for path, dirs, files in os.walk("./resized_dataset/valid/images"):
        print(f"Number of images in the valid folder: {len(files)}")
    for path, dirs, files in os.walk("./resized_dataset/test/images"):
        print(f"Number of images in the test folder: {len(files)}")
    print("\n")

    # Read the yaml file and convert the class labels from list to dictionary
    labels_file = [_ for _ in os.listdir("./dataset") if _.endswith(".yaml")]

    yaml_file = open(f"./dataset/{labels_file[0]}")
    file = yaml.load(yaml_file, Loader = yaml.FullLoader)
    classes = file["names"]
    print("yaml file read successfully!\n")

    labels = {i: classes[i] for i in range(0, len(classes))}
    print("Labels and Indices: ")
    print(labels)
    print("\n")

    #augment(args.n)

    start = time.perf_counter()

    # Try multiprocessing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(augment, args.n)

    end = time.perf_counter()

    # Count the number of images in the dataset after augmentation
    print("\nAugmentation Complete!\n")
    print("After Augmentation:")
    for path, dirs, files in os.walk("./dataset/train/images"):
        print(f"Number of images in the train folder: {len(files)}")
    for path, dirs, files in os.walk("./dataset/valid/images"):
        print(f"Number of images in the valid folder: {len(files)}")
    for path, dirs, files in os.walk("./dataset/test/images"):
        print(f"Number of images in the test folder: {len(files)}")

    # Remove the resized dataset directory, as it is now redundant
    print("\nRemoving redundant directories...")
    shutil.rmtree("./resized_dataset")
    print("Redundant directories removed")


    print(f"Total time taken: {round(end - start, 4)} seconds")