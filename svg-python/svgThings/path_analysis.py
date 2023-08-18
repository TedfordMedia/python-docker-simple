from xml.etree import ElementTree as ET
from svgpathtools import parse_path
import statistics
import numpy as np


def analyze_paths(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}
    groups = root.findall('.//svg:g', namespaces)
    analysis_results = []

    for idx, group in enumerate(groups):
        paths_in_group = group.findall('.//svg:path', namespaces)
        path_lengths = [parse_path(p.get('d')).length()
                        for p in paths_in_group]

        # Basic stats
        shortest_path = min(path_lengths) if path_lengths else 0
        longest_path = max(path_lengths) if path_lengths else 0
        average_path_length = sum(path_lengths) / \
            len(path_lengths) if path_lengths else 0

        # Advanced stats
        mean_length = statistics.mean(path_lengths) if path_lengths else 0
        median_length = statistics.median(path_lengths) if path_lengths else 0
        mode_length = statistics.multimode(
            path_lengths) if path_lengths else []
        std_dev = statistics.stdev(path_lengths) if len(
            path_lengths) > 1 else 0
        total_length = sum(path_lengths)

        analysis_results.append({
            'group_id': group.get('data-param-set-id'),
            'num_paths': len(paths_in_group),
            'shortest_path': shortest_path,
            'longest_path': longest_path,
            'average_path': average_path_length,
            'mean': mean_length,
            'median': median_length,
            'mode': mode_length,
            'std_dev': std_dev,
            'total_length': total_length
        })

    return analysis_results
