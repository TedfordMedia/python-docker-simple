import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import sys
import os


def lbp_analysis(image_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        return

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # If the image is not loaded successfully, notify the user and exit
    if image is None:
        print("Error loading the image. Please check the file format or the path.")
        return

    # Equalize the image
    equalized_image = cv2.equalizeHist(image)

    # Compute LBP
    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(
        equalized_image, n_points, radius, method="uniform")

    # Normalize LBP for visualization
    normalized_lbp = cv2.normalize(
        lbp_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Create an output folder if it doesn't exist
    output_directory = "output2"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Extract filename from the input path
    filename = os.path.basename(image_path)
    filename_without_extension = os.path.splitext(filename)[0]

    # Construct the output path
    output_path = os.path.join(
        output_directory, f"{filename_without_extension}_lbp1.png")
    cv2.imwrite(output_path, normalized_lbp)

    print(f"LBP analysis completed. Result saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lbp_analysis.py <path_to_image>")
        sys.exit()

    image_path = sys.argv[1]
    lbp_analysis(image_path)
