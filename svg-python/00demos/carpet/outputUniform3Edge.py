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

    # Image with original and highlighted edges
    highlighted = image.copy()
    highlighted[edges == 255] = [0, 0, 255]

    # Detecting connected components of the edges
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges)

    # Image showing only the isolated edge islands
    edge_islands = np.zeros_like(image)
    for i in range(1, num_labels):
        # adjust these values based on the desired size of islands
        if 200 < stats[i, cv2.CC_STAT_AREA] < 5000:
            edge_islands[labels == i] = [0, 0, 255]

    # Save the outputs
    filename = os.path.basename(image_path).split('.')[0]
    cv2.imwrite(f"outputUniform3/{filename}_only_edges.png", only_edges)
    cv2.imwrite(
        f"outputUniform3/{filename}_highlighted_edges.png", highlighted)
    cv2.imwrite(f"outputUniform3/{filename}_edge_islands.png", edge_islands)

    print(f"Edge detection completed for {image_path}.")

    # Show histogram for visual analysis
    plt.hist(edges.ravel(), 256, [0, 256])
    plt.title(f'Edge Intensity Histogram for {image_path}')
    plt.show()


if not os.path.exists("outputUniform3"):
    os.makedirs("outputUniform3")

for img_file in os.listdir("./images"):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        edge_detection(f"./images/{img_file}")
