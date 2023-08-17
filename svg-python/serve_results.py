from flask import Flask, send_from_directory

app = Flask(__name__)


@app.route('/results')
def serve_results():
    return send_from_directory('analysisResults', 'anomalousResults.json')


if __name__ == "__main__":
    app.run(debug=True, port=5000)
