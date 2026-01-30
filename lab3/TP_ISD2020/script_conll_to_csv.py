import csv
import argparse
import sys

def convert_conll_to_csv(input_file, output_file, columns, separator, no_header):
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', newline='', encoding='utf-8') as outfile:

            writer = csv.writer(outfile)

            # Write a generic header unless disabled
            if not no_header:
                header = [f"Mot" if str(idx).isdigit() else "Tag" for idx in columns]
                writer.writerow(header)

            line_count = 0

            for line in infile:
                # Skip empty lines (sentence boundaries)
                if not line.strip():
                    continue

                # Split the line based on the user-defined separator
                # If separator is None, split() handles mixed whitespace (spaces/tabs)
                parts = line.split(separator) if separator else line.split()

                row_data = []
                try:
                    for idx in columns:
                        # Handle negative indices (e.g., -1 for the last element)
                        actual_idx = int(idx)
                        row_data.append(parts[actual_idx])

                    writer.writerow(row_data)
                    line_count += 1

                except IndexError:
                    print(f"Warning: Line skipped (not enough columns): {line.strip()}")

            print(f"Success! Processed {line_count} lines. Saved to '{output_file}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert specific CoNLL columns to CSV.")

    # Required parameters
    parser.add_argument("input_file", help="Path to the input .conll file")
    parser.add_argument("output_file", help="Path to save the output .csv file")

    # Optional parameters
    parser.add_argument("--cols", nargs='+', default=[1, -1],
                        help="List of column indices to extract (0-based). Default is '1 -1' (2nd and Last).")

    parser.add_argument("--sep", default=None,
                        help="Delimiter used in input file (e.g., '\\t' for tab). Default is whitespace.")

    parser.add_argument("--no-header", action="store_true",
                        help="Flag to disable writing the header row in CSV.")

    args = parser.parse_args()

    # Run the conversion
    convert_conll_to_csv(args.input_file, args.output_file, args.cols, args.sep, args.no_header)