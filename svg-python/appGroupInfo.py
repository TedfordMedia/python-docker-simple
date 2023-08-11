from flask import Flask
from xml.etree import ElementTree as ET
from svgpathtools import parse_path

app = Flask(__name__)


@app.route('/')
def hello():
    tree = ET.parse('images/tridentFull/trident_167.svg')
    root = tree.getroot()

    # XML namespaces in SVGs can be tricky. This helps grab the correct elements
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}

    groups = root.findall('.//svg:g', namespaces)

    messages = []

    for idx, group in enumerate(groups):
        paths_in_group = group.findall('.//svg:path', namespaces)
        path_lengths = []

        for p in paths_in_group:
            d = p.get('d')
            path_obj = parse_path(d)
            path_lengths.append(path_obj.length())

        shortest_path = min(path_lengths) if path_lengths else 0
        longest_path = max(path_lengths) if path_lengths else 0
        average_path_length = sum(path_lengths) / \
            len(path_lengths) if path_lengths else 0

        group_message = [
            f"Group {idx + 1} (data-param-set-id='{group.get('data-param-set-id')}'):",
            f"  - Paths in Group: {len(paths_in_group)}",
            f"  - Shortest Path Length: {shortest_path:.2f}",
            f"  - Longest Path Length: {longest_path:.2f}",
            f"  - Average Path Length: {average_path_length:.2f}"
        ]

        messages.extend(group_message)

    return "<br>".join(messages)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
