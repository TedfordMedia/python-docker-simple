from flask import Flask
from svgpathtools import svg2paths

app = Flask(__name__)


@app.route('/')
def hello():
    # Load the SVG and extract the paths
    paths, _ = svg2paths('images/tridentFull/trident_167.svg')

    # Calculate path lengths
    path_lengths = [path.length() for path in paths]

    # Determine the shortest, longest, and average path lengths
    shortest_path = min(path_lengths) if path_lengths else None
    longest_path = max(path_lengths) if path_lengths else None
    average_path_length = sum(path_lengths) / \
        len(path_lengths) if path_lengths else None

    # Create the message
    message = [
        f"Total Paths: {len(paths)}",
        # f"First Path: {paths[0] if paths else 'None'}",
        f"Shortest Path Length: {shortest_path}",
        f"Longest Path Length: {longest_path}",
        f"Average Path Length: {average_path_length}"
    ]

    return "<br>".join(message)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
