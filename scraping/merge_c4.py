import argparse
import json
from pathlib import Path

def merge_jsons(input_folder, output_file):
    # Create an empty list to store the data from all the JSON files
    all_data = []
    
    # Find all the .json files in the subfolders of the input folder using pathlib
    json_files = list(input_folder.glob("dataset_part_*/*c4_sparql_*.json*"))
    
    print(f"Found {len(json_files)} JSON files. Merging them into {output_file}...")

    # Read each file and append its contents to all_data
    for json_file in json_files:
        print(f"Processing {json_file}...")
        with open(json_file, "r", encoding="utf-8") as f:
            # Each line in the JSON file represents one example in the dataset
            for line in f:
                all_data.append(json.loads(line.strip()))  # Read each line and parse as JSON

    # Write all collected data into the output JSONL file
    with open(output_file, "w", encoding="utf-8") as fout:
        for item in all_data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")  # Write each entry as a line

    print(f"Successfully merged {len(all_data)} examples into {output_file}")

def main(args):
    # Call the merge_jsons function with the provided arguments
    merge_jsons(Path(args.input_folder), args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Argument for the input folder containing the subfolders with JSON files
    parser.add_argument(
        "-i", "--input-folder", type=str, required=True, help="Path to the folder containing subfolders with JSON files"
    )
    
    # Argument for the output file where the merged JSONL will be saved
    parser.add_argument(
        "-o", "--output-file", type=str, required=True, help="Path to the output JSONL file"
    )

    main(parser.parse_args())
