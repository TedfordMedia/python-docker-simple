import os
import json
from lxml import etree
from sklearn.ensemble import IsolationForest
# import cairosvg


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


# Specify your SVG directory
svg_dir = "images/tridentFull"
svg_files = [os.path.join(svg_dir, f)
             for f in os.listdir(svg_dir) if f.endswith('.svg')]

print("Starting feature extraction...")
features = [extract_svg_features(svg) for svg in svg_files]
print(f"Extracted features from {len(svg_files)} SVG files.")

# Model Training
print("Starting model training...")
# Assuming around 5% of the SVGs are anomalies
model = IsolationForest(contamination=0.01)
model.fit(features)

# Detect anomalies
predictions = model.predict(features)
anomalous_files = [file for file, pred in zip(
    svg_files, predictions) if pred == -1]

print(f"Detected {len(anomalous_files)} anomalous SVG files:")

# Create analysisResults directory

script_dir = os.path.dirname(os.path.abspath(__file__))
# Create analysisResults directory
analysis_dir = os.path.join(script_dir, "analysisResults")
if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)

# Save anomalous results to a JSON file
anomalous_results = [{"filename": os.path.basename(
    file), "number": idx+1} for idx, file in enumerate(anomalous_files)]

with open(os.path.join(analysis_dir, 'anomalousResults.json'), 'w') as json_file:
    json.dump(anomalous_results, json_file, indent=4)

# Create a directory to save the PNG images
output_dir = "anomalous_pngs"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for file in anomalous_files:
    print(file)
#     # Convert SVG to PNG
#     png_file_path = os.path.join(
#         output_dir, os.path.basename(file).replace('.svg', '.png'))
#     cairosvg.svg2png(url=file, write_to=png_file_path)

print("Process completed!")
