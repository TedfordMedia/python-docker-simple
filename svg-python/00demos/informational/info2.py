import os
import re  # Import the regular expression module

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(CURRENT_DIR, "../../images/mazes")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "info_outputs")

# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Filename for storing the results
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "file_list.txt")

# Regular expression pattern to match the temperature
temp_pattern = re.compile(r"_c_(\d+)\.png$")

with open(OUTPUT_FILE, "w") as outfile:
    for img_file in os.listdir(IMAGES_DIR):
        # Match the pattern to the filename
        match = temp_pattern.search(img_file)

        # If a match is found, extract the temperature
        if match:
            temperature = match.group(1)
            print(f"Filename: {img_file}, Temperature: {temperature}°C")
            outfile.write(
                f"Filename: {img_file}, Temperature: {temperature}°C\n")
        else:
            print(f"Filename: {img_file}, Temperature information not found")
            outfile.write(
                f"Filename: {img_file}, Temperature information not found\n")

print(f"File names and temperatures written to {OUTPUT_FILE}")
