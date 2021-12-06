import os
import argparse
import numpy as np
import cv2
import logging
import time
from distutils.dir_util import copy_tree
from imutils import paths
from multiprocessing import Pool, cpu_count
from functools import partial




def resize_image(image: str, size: int = 416):
    resized = cv2.resize(image, (size, size))
    return resized


def make_batch(imgs_list: list, batch_size: int):
    for i in range(0, len(imgs_list), batch_size):
        yield imgs_list[i: i + batch_size]


def process_batch(payload: list, size: int):
        logger.info(f"starting process {payload['id']}")

        for imagePath in payload["input_paths"]:
            image = cv2.imread(imagePath)
            resized_image = resize_image(image, size)
            cv2.imwrite(imagePath, resized_image)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Resize images, retaining folder structure")
    parser.add_argument("--dir", required = True, type = str, help = "Path to the dataset")
    parser.add_argument("--size", type = int, default = 416, help = "Size to resize the image to")
    parser.add_argument("--num-procs", type = int, default = cpu_count()-1, help = f"Number of threads to run in parallel. Keep less than {cpu_count()}")

    args = parser.parse_args()
    
    logging.basicConfig(filename = "resize.log", filemode = "w", format = '%(asctime)s | %(name)s | %(message)s', datefmt = '%Y-%m-%d  %H:%M:%S')
    logger = logging.getLogger('Resize Dataset')
    logger.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(message)s', datefmt = '%Y-%m-%d::%H:%M:%S')
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)


    start_time = time.perf_counter()

    # Check if directory exists
    if not os.path.exists("./temp"):
        os.mkdir("./temp")
        logger.info("Created temp folder to store images")
    else:
        logger.info("temp folder already exists.")

    # If directory, is not empty, copy files from the original dataset
    if len(os.listdir("./temp")) == 0:
        logger.info("Copying files from original dataset folder...")
        copy_tree(args.dir, "./temp")
        logger.info("All files successfully copied.")
    else:
        logger.info("Directory is not empty. Skipping...")
        pass

    processes = args.num_procs if args.num_procs > 0 else cpu_count()
    process_id = list(range(0, processes))

    logger.info("Grabbing image...")
    
    path_images = sorted(list(paths.list_images("./temp")))
    num_images_per_process = int(np.ceil(len(path_images) / float(processes)))

    logger.info(f"Resizing {int(len(path_images))} images...")
    logger.info(f"Processing images in batches of {num_images_per_process}")

    chunked_paths = list(make_batch(path_images, num_images_per_process))

    payloads = []

    for i, image_paths in enumerate(chunked_paths):
        data = {"id": i, "input_paths": image_paths}
        payloads.append(data)

    logger.info(f"Launching {processes} processes...")
    pool = Pool(processes = processes)
    pool.map(partial(process_batch, size = args.size), payloads)

    logger.info("Processing...")
    pool.close()
    pool.join()
    logger.info("Dataset Resized!")

    end_time = time.perf_counter()

    logger.info("=" * 40)
    logger.info(f"Time taken: {end_time - start_time} seconds")
    logger.info("=" * 40)
