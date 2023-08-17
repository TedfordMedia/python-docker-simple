import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

OUTPUT_DIR = "svy-python/00demos/informational/info_outputs"
IMAGES_DIR = "./images/mazes"


def extract_temperature_from_filename(filename):
    try:
        temp_str = filename.split('_')[1][2:].split('.')[0]
        return int(temp_str)
    except IndexError:
        logging.warning(f"Failed to extract temperature from {filename}.")
        return None


# Data structures for aggregated data
color_data_aggregated = {}
channel_histogram_data = {'b': [], 'g': [], 'r': []}


def analyze_image(image_path):
    filename = os.path.basename(image_path).split('.')[0]
    logging.info(f"Processing file: {filename}")

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        logging.warning(f"Failed to read image: {image_path}")
        return


def extract_temperature_from_filename(filename):
    """Extract the temperature value from filename"""
    try:
        temp_str = filename.split('_')[1][2:].split('.')[0]
        return int(temp_str)
    except (IndexError, ValueError):
        logging.warning(f"Failed to extract temperature from {filename}.")
        return None

    # Flatten and get unique color counts
    colors, counts = np.unique(
        image.reshape(-1, 3), axis=0, return_counts=True)

    # Sorting the counts to get the most frequent colors
    sorted_indices = np.argsort(counts)[::-1]
    colors = colors[sorted_indices]
    counts = counts[sorted_indices]

    # Aggregating the top color for each temperature
    top_color = tuple(colors[0])
    color_data_aggregated[temperature] = top_color

    # Plotting histograms for each color channel
    for i, col in enumerate(('b', 'g', 'r')):
        histogram, _ = np.histogram(image[:, :, i].ravel(), bins=256)
        channel_histogram_data[col].append((temperature, histogram))

    # The rest of your original analysis code
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lower_bound = 140
    upper_bound = 180
    light_grey_mask = cv2.inRange(gray, lower_bound, upper_bound)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(light_grey_mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    highlighted = image.copy()
    for i in range(1, num_labels):
        highlighted[labels == i] = [0, 0, 255]
    red_spaces = np.zeros_like(image)
    for i in range(1, num_labels):
        red_spaces[labels == i] = [0, 0, 255]
    total_pixels = gray.size
    gray_pixels = total_pixels - cv2.countNonZero(light_grey_mask)
    white_pixels = total_pixels - gray_pixels
    island_pixels = np.sum(stats[1:, cv2.CC_STAT_AREA])

    # Save individual image outputs
    cv2.imwrite(os.path.join(
        OUTPUT_DIR, f"{filename}_highlighted.png"), highlighted)
    cv2.imwrite(os.path.join(
        OUTPUT_DIR, f"{filename}_red_spaces.png"), red_spaces)

    # Print statistics
    print(f"\nStatistics for {image_path}:")
    print(f"Total Pixels: {total_pixels}")
    print(f"Gray Pixels (Outside the range of light grey): {gray_pixels}")
    print(f"Light Gray Pixels (Within the range): {white_pixels}")
    print(f"Island Pixels (Connected light gray regions): {island_pixels}")


# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    logging.info(f"Created output directory: {OUTPUT_DIR}")

# Analyze all images
for img_file in os.listdir(IMAGES_DIR):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        analyze_image(os.path.join(IMAGES_DIR, img_file))

# Combined bar chart for top color at each temperature
temperatures = list(color_data_aggregated.keys())
top_colors = [color_data_aggregated[temp] for temp in temperatures]

plt.figure(figsize=(12, 6))
for i, temp in enumerate(temperatures):
    plt.barh(i, 1, color=np.array(top_colors[i])/255.0, label=f"{temp}C")
plt.yticks(range(len(temperatures)), [f"{temp}C" for temp in temperatures])
plt.xlabel('Temperature (Celsius)')
plt.title('Top Color per Temperature')
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "combined_color_chart.png"))
plt.close()

# Combined histograms for each color channel
for col in ('b', 'g', 'r'):
    plt.figure(figsize=(12, 6))
    for data in channel_histogram_data[col]:
        plt.plot(data[1], label=f"{data[0]}C")
    plt.title(f"{col.upper()} Channel Histograms")
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"combined_{col}_histograms.png"))
    plt.close()
