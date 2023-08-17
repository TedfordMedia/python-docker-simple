import cairosvg
import os
import sys
print(sys.executable)


def convert_svg_to_png(svg_file_path, png_directory):
    png_file_name = os.path.basename(svg_file_path).replace('.svg', '.png')
    png_file_path = os.path.join(png_directory, png_file_name)

    cairosvg.svg2png(url=svg_file_path, write_to=png_file_path)


# Specify your SVG directory
svg_dir = "images/tridentFull"
svg_files = [os.path.join(svg_dir, f)
             for f in os.listdir(svg_dir) if f.endswith('.svg')]

# Create a directory for PNGs
png_dir = "images/tridentFull_pngs"
if not os.path.exists(png_dir):
    os.makedirs(png_dir)

print("Starting SVG to PNG conversion...")

for svg_file in svg_files:
    convert_svg_to_png(svg_file, png_dir)

print(f"Converted {len(svg_files)} SVG files to PNG.")
