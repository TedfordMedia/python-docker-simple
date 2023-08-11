# Your main application file
from flask import Flask
from path_analysis import analyze_paths
from group_messages import generate_group_messages

app = Flask(__name__)


@app.route('/')
def hello():
    analysis_results = analyze_paths('images/tridentFull/trident_167.svg')
    messages = generate_group_messages(analysis_results)
    return "<br>".join(messages)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
