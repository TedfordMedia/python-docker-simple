# Your main application file
from flask import Flask, render_template_string
from path_analysis import analyze_paths
from group_messages import generate_group_messages
import matplotlib.pyplot as plt
import io
from base64 import b64encode

app = Flask(__name__)


@app.route('/')
def hello():
    analysis_results = analyze_paths('images/tridentFull/trident_167.svg')

    # Extract data for plotting
    group_ids = [result['group_id'] for result in analysis_results]
    total_lengths = [result['total_length'] for result in analysis_results]

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(group_ids, total_lengths, color='skyblue')
    plt.xlabel('Group ID')
    plt.ylabel('Total Path Length')
    plt.title('Total Length of Paths by Group')

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = b64encode(img.getvalue()).decode('utf8')

    # Generate group messages
    messages = generate_group_messages(analysis_results)

    # Render the image and messages in the HTML response
    template = """
    <img src="data:image/png;base64,{{ plot_url }}">
    <br>
    {% for message in messages %}
        {{ message }}<br>
    {% endfor %}
    """
    return render_template_string(template, plot_url=plot_url, messages=messages)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
