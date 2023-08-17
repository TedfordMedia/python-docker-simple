import cv2
import numpy as np
import os


def fill_shapes(image_path):
    # Load the hardcoded image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Cannot load image from {image_path}")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the detected edges
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask
    mask = np.zeros_like(gray)

    # Fill in the contours
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Overlay the filled shapes on the original image
    filled_image = image.copy()
    filled_image[mask == 255] = [0, 0, 255]  # Color the filled shapes in red

    # Save the outputs in the temp11test folder
    if not os.path.exists("temp11test"):
        os.makedirs("temp11test")

    cv2.imwrite("temp11test/temp11_filled_shapes.png", filled_image)
    cv2.imwrite("temp11test/temp11_only_shapes.png", mask)


# Call the function
fill_shapes("temp11.png")
