import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def color_clusters(image_path):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binary mask for flood filling
    mask = np.zeros((gray.shape[0]+2, gray.shape[1]+2), np.uint8)
    red_regions = np.zeros_like(image)

    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            if mask[y+1, x+1] == 0:
                # Using floodFill to find similar colored regions with a more lenient threshold
                _, _, _, rect = cv2.floodFill(
                    gray.copy(), mask, (x, y), None, 20, 20, flags=8)

                # Extract the region
                roi = mask[rect[1]+1:rect[1]+rect[3] +
                           1, rect[0]+1:rect[0]+rect[2]+1]

                # Check for isolated pixels
                kernel = np.ones((3, 3), np.uint8)
                erosion = cv2.erode(roi.astype(np.uint8), kernel, iterations=1)
                isolated_pixels = np.sum(roi) - np.sum(erosion)

                # If the region is larger than 50 pixels and has no isolated pixels, color it red
                if np.sum(roi) > 50 and isolated_pixels == 0:
                    red_regions[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]][roi == 1] = [0, 0, 255]

    # Overlay the red regions onto the original image with 50% opacity
    overlayed_image = cv2.addWeighted(image, 0.5, red_regions, 0.5, 0)

    # Save the outputs
    filename = os.path.basename(image_path).split('.')[0]
    cv2.imwrite(f"outputUniform5/{filename}_overlayed.png", overlayed_image)


if not os.path.exists("outputUniform5"):
    os.makedirs("outputUniform5")

for img_file in os.listdir("./images"):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        color_clusters(f"./images/{img_file}")
