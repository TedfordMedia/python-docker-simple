import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def color_clusters(image_path):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to separate lighter and darker regions
    # This value may need adjustments
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # Find connected components in the thresholded image
    num_labels, labels = cv2.connectedComponents(thresh)

    red_regions = np.zeros_like(image)
    for label in range(1, num_labels):
        mask = (labels == label).astype(np.uint8)
        area = np.sum(mask)

        # If the area of the region is greater than 50 pixels, consider it for coloring
        if area > 50:
            red_regions[mask == 1] = [0, 0, 255]

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
