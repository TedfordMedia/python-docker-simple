from scipy import stats
import os
import cv2
import numpy as np
from skimage import feature  # for LBP
import matplotlib.pyplot as plt
import shutil
import matplotlib.cm as cm
import seaborn as sns

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

# Function to write statistics to a text file


def write_statistics_to_file(image_statistics, output_dir):
    output_stats_file_path = os.path.join(output_dir, "image_statistics.txt")
    with open(output_stats_file_path, 'w') as f:
        for img, stats in image_statistics.items():
            f.write(f"Statistics for {img}:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")


def plot_lbp_histogram(lbp_image, output_dir, filename_prefix):
    lbp_values, counts = np.unique(lbp_image.ravel(), return_counts=True)
    plt.bar(lbp_values, counts, color='blue', edgecolor='black')
    plt.xlabel('LBP Value')
    plt.ylabel('Frequency')
    plt.title('LBP Histogram')
    output_path = os.path.join(output_dir, f"{filename_prefix}_lbp_histogram.png")
    plt.savefig(output_path)
    plt.close()


def save_lbp_as_image(lbp_image, output_dir, filename_prefix):
    lbp_image = cm.jet(lbp_image / lbp_image.max())
    lbp_image = (lbp_image[:, :, :3] * 255).astype(np.uint8)
    output_path = os.path.join(output_dir, f"{filename_prefix}_lbp.png")
    cv2.imwrite(output_path, lbp_image)


def local_binary_pattern(image, P=8, R=1):
    lbp_image = feature.local_binary_pattern(image, P, R, method="uniform")
    return lbp_image


image_statistics = {}

# Loop over each image file in the IMAGES_DIR
for img_file in os.listdir(IMAGES_DIR):
    print(f"Processing {img_file}...")
    file_path = os.path.join(IMAGES_DIR, img_file)
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Calculate Local Binary Patterns
    lbp_image = local_binary_pattern(image)
    lbp_values = lbp_image.ravel()

    # Basic and Advanced Statistics
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

   # Cumulative Distribution Function (CDF) plot
    plt.figure()
    plt.hist(lbp_values, bins=50, density=True, cumulative=True, label='CDF',
             histtype='step', alpha=0.8, color='purple')
    plt.title('Cumulative Distribution Function (CDF)')
    plt.xlabel('LBP Value')
    plt.ylabel('Likelihood of occurrence')
    plt.show()

    # Boxplot
    plt.figure()
    plt.boxplot(lbp_values)
    plt.title('Boxplot of LBP Values')
    plt.show()

    # Heatmap
    plt.figure()
    sns.heatmap(lbp_image, cmap='viridis')
    plt.title('Heatmap of LBP')
    plt.show()

    save_lbp_as_image(lbp_image, OUTPUT_DIR_EXTRAS, f"{img_file}")
    plot_lbp_histogram(lbp_image, OUTPUT_DIR_EXTRAS, f"{img_file}")

# Call the function to write statistics to a file
write_statistics_to_file(image_statistics, OUTPUT_DIR_EXTRAS)

print("Image analysis and data saving complete.")
