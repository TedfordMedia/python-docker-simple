import os
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(CURRENT_DIR, "../../images/mazes")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "info6_outputs")

# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Filename for storing the results
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "file_list.txt")

# Regular expression pattern to match the temperature
temp_pattern = re.compile(r"_c_(\d+)\.png$")

all_pixel_data = {}
temperatures = []

with open(OUTPUT_FILE, "w") as outfile:
    for img_file in os.listdir(IMAGES_DIR):
        file_path = os.path.join(IMAGES_DIR, img_file)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        unique_pixels, counts = np.unique(image, return_counts=True)
        other_count = np.sum(counts[counts < 10])
        unique_pixels = unique_pixels[counts >= 10]
        counts = counts[counts >= 10]
        match = temp_pattern.search(img_file)
        if match:
            temperature = match.group(1)
            temperatures.append(temperature)
            outfile.write(
                f"Filename: {img_file}, Temperature: {temperature}°C\n")
            all_pixel_data[temperature] = (unique_pixels, counts, other_count)
        else:
            outfile.write(
                f"Filename: {img_file}, Temperature information not found\n")

# Plotting
fig, ax = plt.subplots()
for temperature, data in all_pixel_data.items():
    unique_pixels, counts, other_count = data
    ax.bar(unique_pixels, counts, label=f"{temperature}°C", alpha=0.5, width=1)
ax.legend()
ax.set_xlabel("Gray Shade")
ax.set_ylabel("Pixel Count")
ax.set_title("Pixel Shade Distribution by Temperature")
plt.tight_layout()

# Save to PNG with white background
fig.patch.set_facecolor('white')
plt.gca().set_facecolor('white')
temperatures_str = "_".join(temperatures)
output_graph_file = os.path.join(
    OUTPUT_DIR, f"pixel_distribution_{temperatures_str}_bar_chart.png")
plt.savefig(output_graph_file, bbox_inches='tight',
            facecolor=fig.get_facecolor(), transparent=False)

print(f"Graph saved to {output_graph_file}")

# Save the graph again for overlaying but with transparent background
plt.savefig(output_graph_file.replace('.png', '_transparent.png'),
            bbox_inches='tight', transparent=True)

graph_image = cv2.imread(output_graph_file.replace(
    '.png', '_transparent.png'), cv2.IMREAD_UNCHANGED)

for img_file in os.listdir(IMAGES_DIR):
    file_path = os.path.join(IMAGES_DIR, img_file)
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized_graph = cv2.resize(
        graph_image, (colored_image.shape[1], colored_image.shape[0]))
    if resized_graph.shape[2] == 4:
        alpha_channel = resized_graph[:, :, 3] / 255.0
        rgb_channels = resized_graph[:, :, :3]
        colored_image = (1 - alpha_channel).reshape(alpha_channel.shape[0], alpha_channel.shape[1], 1) * colored_image + alpha_channel.reshape(
            alpha_channel.shape[0], alpha_channel.shape[1], 1) * rgb_channels
    else:
        colored_image = cv2.addWeighted(
            colored_image, 0.6, resized_graph, 0.4, 0)

    text_position = (10, colored_image.shape[0] - 10)
    cv2.putText(colored_image, img_file, text_position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    overlayed_image_file = os.path.join(OUTPUT_DIR, f"overlay_{img_file}")
    cv2.imwrite(overlayed_image_file, colored_image)
    print(f"Overlay image saved to {overlayed_image_file}")

# Define a function to add images below the chart


def add_images_below_chart(fig, image_paths):
    num_images = len(image_paths)
    for idx, img_path in enumerate(image_paths):
        img = plt.imread(img_path)
        ax = fig.add_subplot(2, num_images, num_images + idx + 1)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(os.path.basename(img_path))


# Plotting
# Adjusting figure size for taller output
fig, ax = plt.subplots(figsize=(10, 10))
for temperature, data in all_pixel_data.items():
    unique_pixels, counts, other_count = data
    ax.bar(unique_pixels, counts, label=f"{temperature}°C", alpha=0.5, width=1)

# Setting space for original images
fig.subplots_adjust(bottom=0.3)  # adjust this value for optimal spacing

# Add original images below the chart
image_paths = [os.path.join(IMAGES_DIR, img_file)
               for img_file in os.listdir(IMAGES_DIR)]
add_images_below_chart(fig, image_paths)

ax.legend()
ax.set_xlabel("Gray Shade")
ax.set_ylabel("Pixel Count")
ax.set_title("Pixel Shade Distribution by Temperature")
plt.tight_layout()

# Save to PNG with white background
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
output_graph_file_with_images = os.path.join(
    OUTPUT_DIR, f"pixel_distribution_with_images_{temperatures_str}_bar_chart.png")
plt.savefig(output_graph_file_with_images, bbox_inches='tight',
            facecolor=fig.get_facecolor(), transparent=False)

print(f"Graph with images saved to {output_graph_file_with_images}")
