from scipy import stats
from PIL import ImageOps
import os
import cv2
import numpy as np
import re
from skimage import feature  # for LBP
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import seaborn as sns
import matplotlib.cm as cm
import sys
sys.setrecursionlimit(90000)

# Print start message
print("Starting image analysis...")

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(CURRENT_DIR, "../../../../images/mazes")
OUTPUT_DIR_EXTRAS = os.path.join(CURRENT_DIR, "lbp_outputs")

# Delete the existing output folder if it exists
if os.path.exists(OUTPUT_DIR_EXTRAS):
    shutil.rmtree(OUTPUT_DIR_EXTRAS)

# Create a new empty output folder
os.makedirs(OUTPUT_DIR_EXTRAS)

# Regular expression pattern to match the temperature
temp_pattern = re.compile(r"_c_(\d+)\.png$")

# Function to write statistics to a text file


def write_statistics_to_file(image_statistics, output_dir):
    output_stats_file_path = os.path.join(output_dir, "image_statistics.txt")

    # Open the file in write mode
    with open(output_stats_file_path, 'w') as f:
        for img, stats in image_statistics.items():
            f.write(f"Statistics for {img}:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")  # Add a new line to separate each image's statistics


def plot_lbp_histogram(lbp_image, output_dir, filename_prefix):
    lbp_values, counts = np.unique(lbp_image.ravel(), return_counts=True)
    plt.bar(lbp_values, counts, color='blue', edgecolor='black')
    plt.xlabel('LBP Value')
    plt.ylabel('Frequency')
    plt.title('LBP Histogram')

    output_path = os.path.join(
        output_dir, f"{filename_prefix}_lbp_histogram.png")
    plt.savefig(output_path)
    plt.close()


def save_lbp_as_image(lbp_image, output_dir, filename_prefix):
    lbp_image = cm.jet(lbp_image / lbp_image.max())  # Use the jet color map
    lbp_image = (lbp_image[:, :, :3] * 255).astype(np.uint8)
    output_path = os.path.join(output_dir, f"{filename_prefix}_lbp.png")
    Image.fromarray(lbp_image, 'RGB').save(output_path)


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

    def dfs(x, y, color, cluster):
        if (x, y) in visited:
            return
        if x < 0 or y < 0 or x >= image.shape[0] or y >= image.shape[1]:
            return
        if abs(image[x, y] - color) > color_tolerance:
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


def local_binary_pattern(image, P=8, R=1):
    lbp_image = feature.local_binary_pattern(image, P, R, method="uniform")
    return lbp_image


all_pixel_clusters = {}
all_lbp_data = {}
temperatures = []
cluster_count_per_image = {}
image_statistics = {}

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
        clusters = find_clusters(image, color_tolerance=10)
        significant_clusters = [cluster for cluster in clusters if len(cluster[1]) > 10]

        all_pixel_clusters[temperature] = significant_clusters
        cluster_count_per_image[img_file] = len(significant_clusters)
        # Plot histogram of cluster sizes
        # plot_cluster_size_histogram(significant_clusters, OUTPUT_DIR_EXTRAS, img_file)

        small_clusters = [c for c in clusters if len(c[1]) <= 10]
        print(f"Number of small clusters: {len(small_clusters)}")

        # Calculate Local Binary Patterns
        lbp_image = local_binary_pattern(image)
        all_lbp_data[temperature] = lbp_image
        # Now let's calculate the basic statistics
        lbp_values = lbp_image.ravel()
        lbp_mean = np.mean(lbp_values)
        lbp_median = np.median(lbp_values)
        lbp_std_dev = np.std(lbp_values)
        lbp_skewness = stats.skew(lbp_values)
        lbp_kurtosis = stats.kurtosis(lbp_values)

        # Basic and Advanced Statistics
        lbp_values = lbp_image.ravel()
        lbp_mean = np.mean(lbp_values)
        lbp_median = np.median(lbp_values)
        lbp_std_dev = np.std(lbp_values)
        lbp_skewness = stats.skew(lbp_values)
        lbp_kurtosis = stats.kurtosis(lbp_values)
        lbp_entropy = stats.entropy(lbp_values)
        lbp_range = np.ptp(lbp_values)
        lbp_variance = np.var(lbp_values)

  # Save the statistics for this image in the dictionary
        image_statistics[img_file] = {
            "Mean": lbp_mean,
            "Median": lbp_median,
            "StdDev": lbp_std_dev,
            "Skewness": lbp_skewness,
            "Kurtosis": lbp_kurtosis,
            "Entropy": lbp_entropy,
            "Range": lbp_range,
            "Variance": lbp_variance
        }

        # plt.figure()
        # plt.hist(lbp_values, bins=50, density=True, cumulative=True, label='CDF',
        #          histtype='step', alpha=0.8, color='purple')
        # plt.title('Cumulative Distribution Function (CDF)')
        # plt.xlabel('LBP Value')
        # plt.ylabel('Likelihood of occurrence')
        # plt.show()

        # # Boxplot
        # plt.figure()
        # plt.boxplot(lbp_values)
        # plt.title('Boxplot of LBP Values')
        # plt.show()

        # # Heatmap
        # plt.figure()
        # sns.heatmap(lbp_image, cmap='viridis')
        # plt.title('Heatmap of LBP')
        # plt.show()
        # Save LBP as an image
        save_lbp_as_image(lbp_image, OUTPUT_DIR_EXTRAS, f"{img_file}")

        # Place the call to plot_lbp_histogram here
        plot_lbp_histogram(lbp_image, OUTPUT_DIR_EXTRAS, f"{img_file}")

        # Save clusters and LBP data as text files
        save_data_as_text(clusters, lbp_image, OUTPUT_DIR_EXTRAS, f"{img_file}")

# Call the function to write statistics to a file
write_statistics_to_file(image_statistics, OUTPUT_DIR_EXTRAS)

print("Image analysis and data saving complete.")
