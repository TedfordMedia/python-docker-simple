import os
from path_analysis import analyze_paths


def main():
    dir_path = 'tridentFull'
    svg_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.svg')])

    all_results = {}

    for svg_file in svg_files:
        full_path = os.path.join(dir_path, svg_file)

        # Get the results for each SVG file
        analysis_results = analyze_paths(full_path)

        file_results = {}
        for result in analysis_results:
            group_id = result['group_id']
            file_results[group_id] = result

        all_results[svg_file] = file_results

    # Print the results (or process further as needed)
    for svg, results in all_results.items():
        print(f"Results for {svg}:")
        for group_id, group_result in results.items():
            print(f"Group {group_id}: {group_result}")
        print("\n")


if __name__ == '__main__':
    main()
