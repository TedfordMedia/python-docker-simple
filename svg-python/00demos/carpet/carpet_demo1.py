import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def analyze_image(image_path):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Isolate light grey pixels
    lower_bound = 140  # Adjusted value
    upper_bound = 180  # Adjusted value
    light_grey_mask = cv2.inRange(gray, lower_bound, upper_bound)

    # Morphological operations to clean and connect light grey regions
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(light_grey_mask, cv2.MORPH_CLOSE, kernel)

    # Detecting connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)

    # Highlighting detected spaces in the original image
    highlighted = image.copy()
    for i in range(1, num_labels):
        highlighted[labels == i] = [0, 0, 255]

    # Image showing only the detected spaces, colored red
    red_spaces = np.zeros_like(image)
    for i in range(1, num_labels):
        red_spaces[labels == i] = [0, 0, 255]

    # Pixel statistics
    total_pixels = gray.size
    gray_pixels = total_pixels - cv2.countNonZero(light_grey_mask)
    white_pixels = total_pixels - gray_pixels
    island_pixels = np.sum(stats[1:, cv2.CC_STAT_AREA])

    # Save the outputs
    filename = os.path.basename(image_path).split('.')[0]
    cv2.imwrite(f"outputCarpet1/{filename}_highlighted.png", highlighted)
    cv2.imwrite(f"outputCarpet1/{filename}_red_spaces.png", red_spaces)

    # Print the statistics
    print(f"\nStatistics for {image_path}:")
    print(f"Total Pixels: {total_pixels}")
    print(f"Gray Pixels (Outside the range of light grey): {gray_pixels}")
    print(f"Light Gray Pixels (Within the range): {white_pixels}")
    print(f"Island Pixels (Connected light gray regions): {island_pixels}")


if not os.path.exists("outputCarpet1"):
    os.makedirs("outputCarpet1")

for img_file in os.listdir("./images"):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        analyze_image(f"./images/{img_file}")
