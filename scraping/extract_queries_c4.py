import json
import re
import argparse
from pathlib import Path

# Refined extraction logic with the re module imported
def refined_extract_sparql_queries(input_file, output_file):
    total_files = 0
    total_queries = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            text = data.get('text', '')
            
            # Regex to match full PREFIX definitions
            prefix_pattern = r"PREFIX\s+(\w+):\s+<([^>]+)>"
            # Find all PREFIX declarations
            prefixes = re.findall(prefix_pattern, text)
            text = re.sub(prefix_pattern, '', text)
            text = re.sub(r"\n+", "\n", text)
            # Construct a full prefix declaration block
            prefix_block = '\n'.join([f"PREFIX {prefix[0]}: <{prefix[1]}>" for prefix in prefixes])
            
            # Improved regex pattern to capture both prefixes and the query body
            query_pattern = r"(\b(?:SELECT|CONSTRUCT|DESCRIBE|ASK)\s.*?WHERE\s.*?})"

            # Normalize the text (remove HTML tags but keep < > for URIs)
            # Remove HTML tags but leave URIs in < > intact
            text = re.sub(r'<(?!\/?[\w\s]+>)[^>]*>', '', text)  # Remove HTML tags excluding < > URIs

            # Find all SPARQL queries, including the query body
            found_queries = re.findall(query_pattern, text, flags=re.DOTALL)

            queries = []
            placeholder_text = text
            for idx, query in enumerate(found_queries):
                # Full query includes both the prefixes and the body
                full_query = prefix_block + '\n' + query
                queries.append(full_query)
                placeholder_text = placeholder_text.replace(query, f"<Q{idx}>") # Replace the queries with placeholders

            # Create the new row with the required fields
            new_row = {
                'queries': queries,
                'context': placeholder_text,
                'timestamp': data.get('timestamp', ''),
                'url': data.get('url', '')
            }

            total_files += 1
            total_queries += len(queries)

            # Write the new row to the output file
            json.dump(new_row, outfile)
            outfile.write('\n')
        
        print(f"Successfully extracted {total_queries} queries from {total_files} files.")

def main(args):
    # Get paths from arguments
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Run the improved extraction function
    refined_extract_sparql_queries(input_path, output_path)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Extract SPARQL queries and replace with placeholders.")
    
    # Add necessary arguments with abbreviated names
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to the output JSONL file.")

    args = parser.parse_args()
    main(args)
