import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def advanced_pixel_plotting(all_pixel_data, output_directory, temperatures_str):
    # CDF of pixel intensities
    fig, ax = plt.subplots(figsize=(10, 6))
    for temperature, data in all_pixel_data.items():
        unique_pixels, counts, _ = data
        cdf = np.cumsum(counts)
        cdf = cdf / cdf[-1]
        ax.plot(unique_pixels, cdf, label=f"{temperature}°C", alpha=0.7)
    ax.set_xticks([0, 255])
    ax.set_xticklabels(['Black', 'White'])
    ax.legend()
    ax.set_xlabel("Gray Shade")
    ax.set_ylabel("CDF")
    ax.set_title("Pixel Intensity Cumulative Distribution by Temperature")
    plt.tight_layout()
    output_cdf_file = f"{output_directory}/pixel_intensity_cdf_{temperatures_str}.png"
    plt.savefig(output_cdf_file, bbox_inches='tight')
    plt.close()
    print(f"CDF graph saved to {output_cdf_file}")

    # Histogram of pixel intensities
    fig, ax = plt.subplots(figsize=(10, 6))
    for temperature, data in all_pixel_data.items():
        unique_pixels, counts, _ = data
        ax.bar(unique_pixels, counts,
               label=f"{temperature}°C", alpha=0.5, width=1)
    ax.set_xticks([0, 255])
    ax.set_xticklabels(['Black', 'White'])
    ax.legend()
    ax.set_xlabel("Gray Shade")
    ax.set_ylabel("Pixel Count")
    ax.set_title("Pixel Shade Distribution by Temperature")
    plt.tight_layout()
    output_hist_file = f"{output_directory}/pixel_intensity_histogram_{temperatures_str}.png"
    plt.savefig(output_hist_file, bbox_inches='tight')
    plt.close()
    print(f"Histogram graph saved to {output_hist_file}")

    # ... (rest of the code remains the same)


# Additional setups for seaborn (if you choose to use the violin plot)
sns.set(style="whitegrid")
