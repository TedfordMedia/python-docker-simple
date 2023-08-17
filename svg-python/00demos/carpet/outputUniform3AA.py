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

    # Image showing only the detected edges
    only_edges = np.zeros_like(image)
    only_edges[edges == 255] = [0, 0, 255]

    # Binary mask for flood filling
    mask = np.zeros((gray.shape[0]+2, gray.shape[1]+2), np.uint8)
    filled_image = only_edges.copy()

    total_filled_area = 0

    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            if mask[y+1, x+1] == 0 and only_edges[y, x][2] == 0:
                area = cv2.floodFill(filled_image, mask,
                                     (x, y), (0, 255, 0))[0]
                if area < 300:  # reduced threshold for detecting gaps
                    filled_image[mask[1:-1, 1:-1] == 1] = [0, 0, 0]
                else:
                    total_filled_area += area

    # Save the outputs
    filename = os.path.basename(image_path).split('.')[0]
    cv2.imwrite(
        f"outputUniform3AA/{filename}_only_edges_filled.png", filled_image)

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
        "outputUniform3AA", f"{filename}_histogram.png")
    plt.savefig(histogram_filename)
    plt.close()


if not os.path.exists("outputUniform3AA"):
    os.makedirs("outputUniform3AA")

for img_file in os.listdir("./images"):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        edge_detection(f"./images/{img_file}")
