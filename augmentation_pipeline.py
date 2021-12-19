import os
import cv2
import glob
import yaml
import time
import logging
import argparse
import numpy as np
import albumentations as A
from imutils import paths
from functools import partial
from distutils.dir_util import copy_tree
from multiprocessing import Pool, cpu_count





def make_batch(images_path_list, num_images_per_process):
    for i in range(0, len(images_path_list), num_images_per_process):
        yield images_path_list[i: i + num_images_per_process]


def resize_images(image, size: int = 416):
    resized = cv2.resize(image, (size, size))
    return resized


def format_data(filename):
    annotations = []

    with open(f"./augmented_dataset/train/labels/{filename}.txt") as f:
        lines = [line.split() for line in f]
        annotations.append(lines)

    for coordinates in annotations:
        annot = np.array(coordinates).astype(np.float32)

    if annot.size == 0:
        pass
    else:
        old_bboxes = np.delete(annot, 0, axis = 1)
        category_ids = np.delete(annot, [1, 2, 3, 4], axis = 1)

        bboxes = []
        
        for arr in old_bboxes:
            bboxes.extend((arr[0], arr[1], arr[2], arr[3]))     # x_centre, y_centre, width, height

        bboxes = np.array(bboxes).reshape(-1, 4).tolist()
        category_ids = np.array(category_ids).astype(int).ravel().tolist()
    
    return bboxes, category_ids


def resize_dataset(payload: list, size: int):
    logger.info(f"starting process {payload['id']}")

    for path in payload["input_paths"]:
        image = cv2.imread(path)
        resized_image = resize_images(image, size)
        cv2.imwrite(path, resized_image)


def augment_dataset(payload: list, n_augmentations: int = 3):
    logger.info(f"starting process {payload['id']}")

    for path in payload["input_paths"]:
        bboxes, category_ids = format_data(path)

        image = cv2.imread(f"./augmented_dataset/train/images/{path}.jpg")

        transform = A.Compose(transforms,
                               bbox_params = A.BboxParams(format = 'yolo', label_fields = ['category_ids'])
                               )

        transformed = transform(image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB), bboxes = bboxes, category_ids = category_ids)
        
        for aug in range(n_augmentations):
            cv2.imwrite(f"./augmented_dataset/train/images/{path}_{str(aug)}.jpg", transformed["image"])

            transformed_bboxes = np.array(transformed["bboxes"]).astype(np.float32)
            transformed_ids = np.array(transformed["category_ids"]).reshape(-1, 1).astype(str)

            transformed_annotations = np.hstack((transformed_ids, transformed_bboxes))

            with open(f"./augmented_dataset/train/labels/{path}_{aug}.txt", "w") as txt:
                for line in transformed_annotations:
                    txt.write(" ".join(line) + "\n")


transforms = [A.Resize(width = 416, height = 416),      
              A.RandomCrop(width = 416, height = 416),
              A.Rotate(limit = 40, p = 0.9, border_mode = cv2.BORDER_CONSTANT),
              A.HorizontalFlip(p = 0.5),
              A.VerticalFlip(p = 0.1),
              A.RGBShift(r_shift_limit = 25, g_shift_limit = 25, b_shift_limit = 25, p = 0.9)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Augment dataset: Resize -> Transform")
    parser.add_argument("--dir", required = True, type = str, help = "Path to the dataset")
    parser.add_argument("--size", type = int, default = 416, help = "New image size")
    parser.add_argument("--n", type = int, default = 3, help = "Number of images to generate per original image")
    parser.add_argument("--num-procs", type = int, default = cpu_count()-1, help = f"Number of processes to run in parallel. Keep less than {cpu_count()}")

    args = parser.parse_args()

    logging.basicConfig(filename = "pipeline.log", filemode = "w", format = '%(asctime)s | %(name)s | %(message)s', datefmt = '%Y-%m-%d  %H:%M:%S')
    logger = logging.getLogger('Resize Dataset ')
    logger.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(message)s', datefmt = '%Y-%m-%d::%H:%M:%S')
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.info("Resizing dataset...")

    start_time = time.perf_counter()

    # Check if directory exists
    if not os.path.exists("./augmented_dataset"):
        os.mkdir("./augmented_dataset")
        logger.info("Created augmented_dataset folder to store images")
    else:
        logger.info("augmented_dataset folder already exists.")

    # If directory, is not empty, copy files from the original dataset
    if len(os.listdir("./augmented_dataset")) == 0:
        logger.info("Copying files from original dataset folder...")
        copy_tree(args.dir, "./augmented_dataset")
        logger.info("All files successfully copied.")
    else:
        logger.info("Directory is not empty. Skipping...")
        pass

    processes = args.num_procs if args.num_procs > 0 else cpu_count()
    process_id = list(range(0, processes))

    logger.info("Grabbing image paths...")
    
    path_images = sorted(list(paths.list_images("./augmented_dataset")))
    num_images_per_process = int(np.ceil(len(path_images) / float(processes)))

    logger.info(f"Resizing {int(len(path_images))} images...")
    logger.info(f"Assigning {num_images_per_process} images to each process")

    chunked_paths = list(make_batch(path_images, num_images_per_process))

    payloads = []

    for i, image_paths in enumerate(chunked_paths):
        data = {"id": i, "input_paths": image_paths}
        payloads.append(data)

    logger.info(f"Launching {processes} processes...")
    pool = Pool(processes = processes)
    logger.info("Processing...")

    pool.map(partial(resize_dataset, size = args.size), payloads)
    pool.close()
    pool.join()
    logger.info("Dataset Resized!")
    
    logger.info("\n")
    logger = logging.getLogger('Augment Dataset')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(message)s', datefmt = '%Y-%m-%d  %H:%M:%S')
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.info("Augmenting dataset...")

    config_file = yaml.load(open("./augmented_dataset/data.yaml"), Loader = yaml.FullLoader)
    logger.info("Config file read successfully!")
    logger.info(f"Number of classes: {config_file['nc']}")
    logger.info(config_file["names"])

    imgs_path = glob.glob("./augmented_dataset/train/images/*")
    annots_path = glob.glob("./augmented_dataset/train/labels/*")

    imgs_annots_pairs = list(set([os.path.splitext(img.split(os.sep)[-1])[0] for img in imgs_path]).intersection([os.path.splitext(annot.split(os.sep)[-1])[0] for annot in annots_path]))

    logger.info("Grabbing image paths...")

    path_images = sorted(imgs_annots_pairs)
    num_images_per_process = int(np.ceil(len(path_images) / float(processes)))

    logger.info(f"Augmenting {int(len(path_images))} images...")
    logger.info(f"Assigning {num_images_per_process} per process")
    logger.info(f"Generating {args.n} images per image.")

    chunked_paths = list(make_batch(path_images, num_images_per_process))
    
    payloads = []

    for i, image_paths in enumerate(chunked_paths):
        data = {"id": i, "input_paths": image_paths}
        payloads.append(data)

    logger.info(f"Launching {processes} processes...")
    pool = Pool(processes = processes)

    logger.info("Processing...")
    pool.map(partial(augment_dataset, n_augmentations = args.n), payloads)
    pool.close()
    pool.join()
    logger.info("Augmentation complete!")

    end_time = time.perf_counter()

    logger.info(f"Train folder contains {int(len(path_images))} images before augmentation")
    logger.info(f"Train folder contains {len(glob.glob('./augmented_dataset/train/images/*'))} after augmentation")
    logger.info(f"Data saved in the '/augmented_dataset' folder")

    logger.info("=" * 40)
    logger.info(f"Time taken: {end_time - start_time} seconds")
    logger.info("=" * 40)
