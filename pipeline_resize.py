import os
import cv2
import argparse
from distutils.dir_util import copy_tree
from tqdm import tqdm



def resize(path, size = 416):
    """
    Simple function to resize the image to the specified size
    input -> image path
    output -> array of resized image
    """
    im = cv2.imread(path)
    img = cv2.resize(im, (size, size))

    return img



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Resize all images in all folders, retaining the structure")

    parser.add_argument("--dir", type = str, required = True, help = "Path to the input image directory")
    parser.add_argument("--size", type = int, default = 416, help = "Size of the resized image")
    
    args = parser.parse_args()

    # Check if directory exists
    if not os.path.isdir("./resized_dataset"):
        os.mkdir("./resized_dataset")
        print("Created ./resized_dataset directory")

    # If the directory is empty, copy the files from the original dataset, else skip
    if  len(os.listdir("./resized_dataset")) == 0:
        copy_tree(args.dir, "./resized_dataset")
        print("Successfully copied all files to the resized_dataset folder")
    else:
        print("Directory is not empty")
        pass

    # Resize image
    print("Resizing images...")
    for root, dirs, files in os.walk("./resized_dataset"):
        for name in tqdm(files):
            filepath = root + os.sep + name

            if filepath.endswith(".jpg"):
                resized_img = resize(path = os.path.join(root, name), size = args.size)
                cv2.imwrite(os.path.join(root, name), resized_img)
    print("Datset resized!")