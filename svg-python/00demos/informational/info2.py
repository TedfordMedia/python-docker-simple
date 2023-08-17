import os
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
IMAGES_DIR = os.path.join(CURRENT_DIR, "../../images/mazes")
OUTPUT_DIR = "./info_outputs"
OUTPUT_FILENAME = "file_list.txt"


# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Open the output file for writing
with open(os.path.join(OUTPUT_DIR, OUTPUT_FILENAME), 'w') as output_file:
    # Iterate over all files in the directory
    for img_file in os.listdir(IMAGES_DIR):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            # Write the filename to the output file
            output_file.write(img_file + '\n')
            # Print the filename to the console
            print(img_file)

print(f"File names written to {os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)}")
