import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def edge_detection(image_path):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Finding contours in the edge-detected image
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled_image = np.zeros_like(image)
    total_filled_area = 0

    # Filling the large regions enclosed by contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # reduced threshold for detecting more green areas
            # Filling the contour
            cv2.drawContours(
                filled_image, [contour], 0, (0, 255, 0), thickness=-1)
            total_filled_area += area

    # Overlay the filled regions onto the original image with 50% opacity
    overlayed_image = cv2.addWeighted(image, 0.5, filled_image, 0.5, 0)

    # Save the outputs
    filename = os.path.basename(image_path).split('.')[0]
    cv2.imwrite(
        f"outputUniform4/{filename}_only_edges_filled.png", filled_image)
    cv2.imwrite(f"outputUniform4/{filename}_overlayed.png", overlayed_image)

    # Print statistics
    total_pixels = gray.shape[0] * gray.shape[1]
    edge_pixels = np.sum(edges == 255)

    print(f"Statistics for {image_path}:")
    print(f"Total Pixels: {total_pixels}")
    print(f"Edge Pixels: {edge_pixels}")
    print(f"Filled Pixels: {total_filled_area}")

    # Save histogram to file for visual analysis
    plt.hist(edges.ravel(), 256, [0, 256])
    plt.title(f'Edge Intensity Histogram for {image_path}')
    histogram_filename = os.path.join(
        "outputUniform4", f"{filename}_histogram.png")
    plt.savefig(histogram_filename)
    plt.close()


if not os.path.exists("outputUniform4"):
    os.makedirs("outputUniform4")

for img_file in os.listdir("./images"):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        edge_detection(f"./images/{img_file}")
