import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def uniform_color_highlight(image_path):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define kernel for uniform color checking
    kernel = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ])

    # Convolve the image with the kernel
    response = cv2.filter2D(gray, -1, kernel)

    # Apply morphological operation to emphasize detected regions
    morphed = cv2.morphologyEx(
        response, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Threshold the response to get regions of uniform color
    _, uniform_regions = cv2.threshold(
        morphed, 0.01, 255, cv2.THRESH_BINARY_INV)

    # Create an output image where uniform regions are highlighted in red
    highlighted = image.copy()
    highlighted[uniform_regions == 255] = [0, 0, 255]

    # Create an output image that only shows the detected areas in red
    only_red = np.zeros_like(image)
    only_red[uniform_regions == 255] = [0, 0, 255]

    # Save the outputs
    filename = os.path.basename(image_path).split('.')[0]
    cv2.imwrite(
        f"outputUniform/{filename}_highlighted_extreme.png", highlighted)
    cv2.imwrite(f"outputUniform/{filename}_only_red.png", only_red)

    print(f"Highlighting completed for {image_path}.")

    # Show histogram for visual analysis
    plt.hist(response.ravel(), 256, [0, 256])
    plt.title(f'Histogram for {image_path}')
    plt.show()


if not os.path.exists("outputUniform"):
    os.makedirs("outputUniform")

for img_file in os.listdir("./images"):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        uniform_color_highlight(f"./images/{img_file}")
