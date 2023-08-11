from flask import Flask
from svgpathtools import svg2paths

app = Flask(__name__)


@app.route('/')
def hello():
    # Load the SVG and extract the paths
    paths, _ = svg2paths('images/tridentFull/trident_0.svg')

    # For demonstration purposes, let's say you want to return the first path (if exists)
    if paths:
        return f"The first path is: {paths[0]}"
    else:
        return "No paths found in the SVG."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
