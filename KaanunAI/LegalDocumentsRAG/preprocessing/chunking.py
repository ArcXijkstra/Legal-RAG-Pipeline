import json
import os

# Define input and output directories
input_dir = '../Data/Parsed'  # Directory containing the JSON files
output_dir = '../Data/Chunked'  # Directory to save processed files

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each JSON file in the input directory
def process_json_file(input_path, output_path):
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            # Construct full file paths
            input_path = os.path.join(input_dir, filename)
            output_filename = f"{os.path.splitext(filename)[0]}_chunk.json"
            output_path = os.path.join(output_dir, output_filename)

            # Load and process the JSON file
            with open(input_path, 'r') as f:
                data = json.load(f)

            sections = []
            current_section = None

            for element in data:
                if element['type'] == 'Title':
                    if current_section:
                        sections.append(current_section)
                    current_section = {
                        'section_title': element['text'],
                        'content': []
                    }
                elif element['type'] == 'NarrativeText' and current_section:
                    current_section['content'].append(element['text'])

            # Add the last section if it exists
            if current_section:
                sections.append(current_section)

            # Save the processed data
            with open(output_path, 'w') as f:
                json.dump(sections, f, indent=4)

            print(f"Processed data saved to: {output_path}")

if __name__ == "__main__":
    process_json_file(input_dir, output_dir)