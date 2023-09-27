import os
import cv2
import numpy as np
import re
from skimage import feature  # for LBP
from PIL import Image
import matplotlib.pyplot as plt

# Print start message
print("Starting image analysis...")

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(CURRENT_DIR, "../../../images/mazes")
OUTPUT_DIR_EXTRAS = os.path.join(CURRENT_DIR, "adjacents_outputs_extras")

# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR_EXTRAS):
    os.makedirs(OUTPUT_DIR_EXTRAS)

# Regular expression pattern to match the temperature
temp_pattern = re.compile(r"_c_(\d+)\.png$")

# Function to create a summary image of all clusters


def create_summary_image(clusters, img_shape, output_dir, filename_prefix):
    summary_img = np.zeros(img_shape, dtype=np.uint8)
    for idx, (_, cluster) in enumerate(clusters):
        color = int(255 * (idx + 1) / len(clusters))
        # Debugging line
        for x, y in cluster:
            summary_img[x, y] = color

    # Save the image
    output_path = os.path.join(output_dir, f"{filename_prefix}_summary.png")
    Image.fromarray(summary_img, 'L').save(output_path)

# Function to plot histogram of cluster sizes


def plot_cluster_size_histogram(clusters, output_dir, filename_prefix):
    cluster_sizes = [len(cluster) for _, cluster in clusters]
    plt.hist(cluster_sizes, bins=30, alpha=0.7,
             color='blue', edgecolor='black')
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster Size')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    # Save the plot
    output_path = os.path.join(
        output_dir, f"{filename_prefix}_cluster_size_histogram.png")
    plt.savefig(output_path)
    plt.close()

# Function to save clusters as images


# Function to save clusters as images
def save_clusters_as_images(clusters, output_dir, filename_prefix, img_shape):
    # Create a subfolder for each image's clusters
    subfolder = os.path.join(output_dir, f"{filename_prefix}_clusters")
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    # Initialize an empty summary image
    summary_img = np.zeros(img_shape, dtype=np.uint8)

    for idx, (color, cluster) in enumerate(clusters):
        # Only save clusters of a significant size (e.g., more than 10 pixels)
        if len(cluster) > 10:
            # Initialize an empty image
            img = np.zeros(img_shape, dtype=np.uint8)

            # Fill in the cluster pixels
            for x, y in cluster:
                img[x, y] = color

            # Add this cluster to the summary image
            summary_img = np.maximum(summary_img, img)

            # Save the individual cluster image
            output_path = os.path.join(
                subfolder, f"{filename_prefix}_cluster_{idx}.png")
            # 'L' indicates it's a grayscale image
            img_pil = Image.fromarray(img, 'L')
            img_pil.save(output_path)

    # Save the summary image
    summary_path = os.path.join(
        output_dir, f"{filename_prefix}_summary_try2.png")
    summary_img_pil = Image.fromarray(summary_img, 'L')
    summary_img_pil.save(summary_path)


# Function to save LBP as an image


def save_lbp_as_image(lbp_image, output_dir, filename_prefix):
    output_path = os.path.join(output_dir, f"{filename_prefix}_lbp.png")
    Image.fromarray(lbp_image.astype('uint8')).save(output_path)


# Function to save clusters and LBP as text files
def save_data_as_text(clusters, lbp_data, output_dir, filename_prefix):
    # Save clusters
    clusters_path = os.path.join(output_dir, f"{filename_prefix}_clusters.txt")
    with open(clusters_path, 'w') as f:
        for color, cluster in clusters:
            f.write(f"Color: {color}, Cluster: {cluster}\\n")

    # Save LBP data
    lbp_path = os.path.join(output_dir, f"{filename_prefix}_lbp.txt")
    np.savetxt(lbp_path, lbp_data, fmt='%d')


# Function to find pixel clusters

def find_clusters(image):
    visited = set()
    clusters = []

    def dfs(x, y, color, cluster):
        if (x, y) in visited:
            return
        if x < 0 or y < 0 or x >= image.shape[0] or y >= image.shape[1]:
            return
        if image[x, y] != color:
            return

        visited.add((x, y))
        cluster.append((x, y))

        dfs(x+1, y, color, cluster)
        dfs(x-1, y, color, cluster)
        dfs(x, y+1, color, cluster)
        dfs(x, y-1, color, cluster)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if (x, y) not in visited:
                color = image[x, y]
                cluster = []
                dfs(x, y, color, cluster)
                if cluster:
                    clusters.append((color, cluster))

    return clusters

# Function to calculate LBP


def local_binary_pattern(image, P=8, R=1):
    lbp_image = feature.local_binary_pattern(image, P, R, method="uniform")
    return lbp_image


all_pixel_clusters = {}
all_lbp_data = {}
temperatures = []
# Loop over each image file in the IMAGES_DIR
for img_file in os.listdir(IMAGES_DIR):
    print(f"Processing {img_file}...")
    file_path = os.path.join(IMAGES_DIR, img_file)
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    match = temp_pattern.search(img_file)
    if match:
        temperature = match.group(1)
        temperatures.append(temperature)

        # Find clusters of adjacent pixels
        clusters = find_clusters(image)
        all_pixel_clusters[temperature] = clusters

        # Plot histogram of cluster sizes
        plot_cluster_size_histogram(clusters, OUTPUT_DIR_EXTRAS, img_file)

        # Create and save a summary image containing all clusters
        create_summary_image(clusters, image.shape,
                             OUTPUT_DIR_EXTRAS, img_file)

        # Save clusters as images
        save_clusters_as_images(
            clusters, OUTPUT_DIR_EXTRAS, f"{img_file}", image.shape)

        # Calculate Local Binary Patterns
        lbp_image = local_binary_pattern(image)
        all_lbp_data[temperature] = lbp_image

        # Save LBP as an image
        save_lbp_as_image(lbp_image, OUTPUT_DIR_EXTRAS, f"{img_file}")

        # Save clusters and LBP data as text files
        save_data_as_text(clusters, lbp_image,
                          OUTPUT_DIR_EXTRAS, f"{img_file}")

print("Image analysis and data saving complete.")
