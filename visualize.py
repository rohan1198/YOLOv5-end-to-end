import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
from PIL import Image


BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=1):
    """
    Visualizes bounding boxes on the images (Just as a sanity check for augmented images
    """
    x_centre, y_centre, w, h = bbox

    x_min = np.ceil((x_centre - (w / 2)) * 416).astype(int)
    x_max = np.ceil((x_centre + (w / 2)) * 416).astype(int)
    y_min = np.ceil((y_centre - (h / 2)) * 416).astype(int)
    y_max = np.ceil((y_centre + (h / 2)) * 416).astype(int)
    
    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, 
                text = class_name, 
                org = (x_min, y_min - int(0.3 * text_height)),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.35, 
                color = TEXT_COLOR, 
                lineType = cv2.LINE_AA,
                )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    return img


if __name__ == "__main__":
    images_list = os.listdir("./dataset/train/images")[:16]     # Display the first 16 images
    annotations = []
    augmented_images = []
    
    for img in images_list:
        filename = os.path.splitext(img)[0]
        
        with open(f"./dataset/train/labels" + "/" + filename + ".txt") as f:
            lines = [line.split() for line in f]
            annotations.append(lines)

        # Format Bounding Box Coordinates
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
        
        image = cv2.imread(f"./dataset/train/images/{img}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        yaml_file = open("./dataset/data.yaml")
        file = yaml.load(yaml_file, Loader = yaml.FullLoader)
        labels_lines = file["names"]

        labels = {i: labels_lines[i] for i in range(0, len(labels_lines))}

        #visualize(image, bboxes, category_ids, labels)
        augmented_images.append(visualize(image, bboxes, category_ids, labels))

    # Plot and save the image
    for num, x in enumerate(augmented_images):
        img = Image.fromarray(np.uint8(x)).convert('RGB')
        plt.subplot(4, 4, num+1)
        plt.axis('off')
        plt.imshow(img)
    plt.savefig("augmented_images.png", dpi = 500)