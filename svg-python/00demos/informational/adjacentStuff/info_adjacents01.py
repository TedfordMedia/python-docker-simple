from PIL import ImageOps
import os
import cv2
import numpy as np
import re
from skimage import feature  # for LBP
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import matplotlib.cm as cm
import sys
sys.setrecursionlimit(10000)

# Print start message
print("Starting image analysis...")

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(CURRENT_DIR, "../../../images/mazes")
OUTPUT_DIR_EXTRAS = os.path.join(CURRENT_DIR, "adjacents_outputs_extras")

# Delete the existing output folder if it exists
if os.path.exists(OUTPUT_DIR_EXTRAS):
    shutil.rmtree(OUTPUT_DIR_EXTRAS)

# Create a new empty output folder
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
    summary_img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    summary_img_alpha = np.zeros(
        (img_shape[0], img_shape[1], 4), dtype=np.uint8)

    for idx, (_, cluster) in enumerate(clusters):
        # Only save clusters of a significant size (e.g., more than 10 pixels)
        if len(cluster) > 10:
            # Initialize an empty image
            img = np.zeros(img_shape, dtype=np.uint8)

            # Fill in the cluster pixels
            for x, y in cluster:
                img[x, y] = 255  # Set to white
                summary_img[x, y] = [0, 0, 255]  # Set to red
                summary_img_alpha[x, y] = [
                    0, 0, 255, 255]  # Set to red with alpha

            # Save the individual cluster image
            output_path = os.path.join(
                subfolder, f"{filename_prefix}_cluster_{idx}.png")
            img_pil = Image.fromarray(img, 'L')
            img_pil.save(output_path)

    # Save the summary image
    summary_path = os.path.join(
        output_dir, f"{filename_prefix}_summary_try2.png")
    summary_img_pil = Image.fromarray(summary_img, 'RGB')
    summary_img_pil.save(summary_path)

    # Create and save the overlay image
    original_img = Image.open(os.path.join(
        IMAGES_DIR, filename_prefix)).convert('RGBA')
    overlay_img = Image.fromarray(summary_img_alpha, 'RGBA')
    overlay_img = ImageOps.fit(
        overlay_img, original_img.size, method=0, bleed=0.0, centering=(0.5, 0.5))
    combined_img = Image.alpha_composite(original_img, overlay_img)

    overlay_path = os.path.join(
        output_dir, f"{filename_prefix}_summary_try2_overlay.png")
    combined_img.save(overlay_path)
    num_files = len(os.listdir(subfolder))
    print(
        f"Number of cluster files: {num_files}, Number of clusters: {len(clusters)}")

# Function to save LBP as an image


def save_lbp_as_image(lbp_image, output_dir, filename_prefix):
    lbp_image = cm.jet(lbp_image / lbp_image.max())  # Use the jet color map
    lbp_image = (lbp_image[:, :, :3] * 255).astype(np.uint8)
    output_path = os.path.join(output_dir, f"{filename_prefix}_lbp.png")
    Image.fromarray(lbp_image, 'RGB').save(output_path)


def plot_number_of_clusters(cluster_count, output_dir):
    image_names = list(cluster_count.keys())
    num_clusters = list(cluster_count.values())

    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.bar(image_names, num_clusters, color='green', edgecolor='black')
    plt.xlabel('Image Name')
    plt.ylabel('Number of Clusters (Files)')
    plt.title('Number of Clusters (Files) Per Image')
    plt.xticks(rotation=90)  # Rotate x labels for better visibility
    plt.tight_layout()  # Adjust layout for better visibility

    output_path = os.path.join(output_dir, "number_of_clusters_per_image.png")
    plt.savefig(output_path)
    plt.close()


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

def find_clusters(image, color_tolerance=5):
    visited = set()
    clusters = []

    def dfs_iterative(start_x, start_y, color):
        stack = [(start_x, start_y)]
        cluster = []

        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            if x < 0 or y < 0 or x >= image.shape[0] or y >= image.shape[1]:
                continue
            if abs(image[x, y] - color) > color_tolerance:
                continue

            visited.add((x, y))
            cluster.append((x, y))

            stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])

        return cluster

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if (x, y) not in visited:
                color = image[x, y]
                cluster = dfs_iterative(x, y, color)
                if cluster:
                    clusters.append((color, cluster))

    return clusters


def plot_cluster_count(cluster_count, output_dir):
    image_names = list(cluster_count.keys())
    num_clusters = list(cluster_count.values())

    plt.bar(image_names, num_clusters)
    plt.xlabel('Image Name')
    plt.ylabel('Number of Clusters')
    plt.title('Number of Clusters per Image')
    plt.xticks(rotation=90)  # Rotate x labels for better visibility

    output_path = os.path.join(output_dir, "cluster_count_per_image.png")
    plt.savefig(output_path)
    plt.close()


def plot_aggregated_pixel_count_in_clusters(clusters, output_dir, filename_prefix, bin_size=10):
    cluster_sizes = [len(cluster) for _, cluster in clusters]
    max_size = max(cluster_sizes)

    # Create bins
    bins = list(range(0, max_size + bin_size, bin_size))

    # Initialize the bin counts to zero
    bin_counts = [0] * len(bins)

    # Populate the bin counts
    for size in cluster_sizes:
        index = size // bin_size
        bin_counts[index] += 1

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(bins, bin_counts, marker='o')
    plt.xlabel('Cluster Size')
    plt.ylabel('Number of Clusters')
    plt.title(f'Aggregated Pixel Count in Clusters for {filename_prefix}')

    # Save the plot
    output_path = os.path.join(
        output_dir, f"{filename_prefix}_aggregated_pixel_count_in_clusters.png")
    plt.savefig(output_path)
    plt.close()

# Function to calculate LBP


def local_binary_pattern(image, P=8, R=1):
    lbp_image = feature.local_binary_pattern(image, P, R, method="uniform")
    return lbp_image


all_pixel_clusters = {}
all_lbp_data = {}
temperatures = []
cluster_count_per_image = {}

# Loop over each image file in the IMAGES_DIR
for img_file in os.listdir(IMAGES_DIR):
    print(f"Processing {img_file}...")
    file_path = os.path.join(IMAGES_DIR, img_file)
    # image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)
    match = temp_pattern.search(img_file)
    if match:
        temperature = match.group(1)
        temperatures.append(temperature)

        # Find clusters of adjacent pixels
        clusters = find_clusters(image, color_tolerance=10)
        significant_clusters = [
            cluster for cluster in clusters if len(cluster[1]) > 10]

        all_pixel_clusters[temperature] = significant_clusters
        cluster_count_per_image[img_file] = len(significant_clusters)
        # Plot histogram of cluster sizes
        plot_cluster_size_histogram(
            significant_clusters, OUTPUT_DIR_EXTRAS, img_file)

        # Create and save a summary image containing all clusters
        create_summary_image(clusters, image.shape,
                             OUTPUT_DIR_EXTRAS, img_file)

        # Save clusters as images
        save_clusters_as_images(
            clusters, OUTPUT_DIR_EXTRAS, f"{img_file}", image.shape)

        small_clusters = [c for c in clusters if len(c[1]) <= 10]
        print(f"Number of small clusters: {len(small_clusters)}")

        # Calculate Local Binary Patterns
        lbp_image = local_binary_pattern(image)
        all_lbp_data[temperature] = lbp_image

        # Save LBP as an image
        save_lbp_as_image(lbp_image, OUTPUT_DIR_EXTRAS, f"{img_file}")

        # Save clusters and LBP data as text files
        save_data_as_text(clusters, lbp_image,
                          OUTPUT_DIR_EXTRAS, f"{img_file}")

plot_cluster_count(cluster_count_per_image, OUTPUT_DIR_EXTRAS)
plot_number_of_clusters(cluster_count_per_image, OUTPUT_DIR_EXTRAS)
plot_aggregated_pixel_count_in_clusters(
    significant_clusters, OUTPUT_DIR_EXTRAS, img_file)

print("Image analysis and data saving complete.")
