# group_messages.py
from flask import Markup


def generate_group_messages(analysis_results):
    messages = []

    for result in analysis_results:
        group_message = [
            Markup(
                f"<strong>data-param-set-id='{result['group_id']}'</strong>"),
            Markup(f"  - Paths in Group: {result['num_paths']}"),
            Markup(f"  - Shortest: {result['shortest_path']:.2f}"),
            Markup(f"  - Longest: {result['longest_path']:.2f}"),
            Markup(f"  - Average: {result['average_path']:.2f}"),
            Markup(f"  - Mean Length: {result['mean']:.2f}"),
            Markup(f"  - Median Length: {result['median']:.2f}"),
            # Use Markup for mode only if it exists
            # Markup(f"  - Mode Length(s): {', '.join([f'{mode:.2f}' for mode in result['mode']])}") if result.get('mode') else Markup("No mode"),
            Markup(f"  - Standard Deviation: {result['std_dev']:.2f}"),
            Markup(
                f"  - Total Length of Paths in Group: {result['total_length']:.2f}")
        ]
        messages.extend(group_message)
        messages.append("")

    return messages
