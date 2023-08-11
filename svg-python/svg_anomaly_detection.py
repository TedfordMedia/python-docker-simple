import os
from lxml import etree


def extract_svg_features(svg_file_path):
    with open(svg_file_path, 'r') as file:
        parser = etree.XMLParser(recover=True)
        tree = etree.parse(file, parser)
        paths = tree.xpath(
            '//svg:path', namespaces={'svg': 'http://www.w3.org/2000/svg'})

        num_paths = len(paths)
        total_path_length = sum([len(p.attrib['d']) for p in paths])
        avg_path_length = total_path_length / num_paths if num_paths != 0 else 0

        return [num_paths, avg_path_length, total_path_length]


print("Starting feature extraction...")

# Adjusted the path to the directory where your SVGs are stored
svg_dir = "images/tridentFull"
svg_files = [os.path.join(svg_dir, f)
             for f in os.listdir(svg_dir) if f.endswith('.svg')]

features = []  # Initializing the features list
total_files = len(svg_files)
for i, svg in enumerate(svg_files, 1):
    print(f"Processing file {i} out of {total_files}...")
    features.append(extract_svg_features(svg))

print("Feature extraction completed!")
