import json

# Load the JSON file
with open('Parsed Data 1/eta2063.json', 'r') as f:
    data = json.load(f)

# Initialize variables
sections = []
current_section = None

# Iterate through the elements
for element in data:
    if element['type'] == 'Title':
        # If we encounter a new title, save the current section and start a new one
        if current_section:
            sections.append(current_section)
        current_section = {
            'section_title': element['text'],
            'content': []
        }
    elif element['type'] == 'NarrativeText':
        # Add narrative text to the current section
        if current_section:
            current_section['content'].append(element['text'])

# Don't forget to add the last section
if current_section:
    sections.append(current_section)

# Save the structured sections to a new JSON file
with open('Parsed Data 1/eta2063_chunk.json', 'w') as f:
    json.dump(sections, f, indent=4)

print("Chunked data saved to 'eta2063_chunk.json'.")