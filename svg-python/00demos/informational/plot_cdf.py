# plot_cdf.py

import numpy as np
import matplotlib.pyplot as plt


def plot_pixel_intensity_cdf(all_pixel_data, output_directory, temperatures_str):
    """
    Plot CDF of pixel intensities for given image data.

    Args:
    - all_pixel_data (dict): Dictionary containing pixel data for images.
    - output_directory (str): Directory to save the CDF plot.
    - temperatures_str (str): String representation of temperatures for naming.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    for temperature, data in all_pixel_data.items():
        unique_pixels, counts, _ = data

        # Calculate CDF
        cdf = np.cumsum(counts)
        cdf = cdf / cdf[-1]

        ax.plot(unique_pixels, cdf, label=f"{temperature}°C", alpha=0.7)

    ax.legend()
    ax.set_xlabel("Gray Shade")
    ax.set_ylabel("CDF")
    ax.set_title("Pixel Intensity Cumulative Distribution by Temperature")
    plt.tight_layout()

    output_cdf_file = f"{output_directory}/pixel_intensity_cdf2_{temperatures_str}.png"
    plt.savefig(output_cdf_file, bbox_inches='tight')
    plt.close()

    print(f"CDF graph saved to {output_cdf_file}")
