from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, Goodbye again!'


if __name__ == '__main__':
    print("Starting the Flask app...")
    app.run(host='0.0.0.0', port=5001, debug=True)
