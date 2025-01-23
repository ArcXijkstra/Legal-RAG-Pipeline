from unstructured.partition.api import partition_via_api
from unstructured.staging.base import elements_to_json
import os

# Set up API credentials
os.environ["UNSTRUCTURED_API_KEY"] = "usWIjZXkr1uW2C2GYfUdVDi6LbJOIs"
os.environ["UNSTRUCTURED_API_URL"] = "https://api.unstructured.io/general/v0/general"

# Input and output directories
input_path = "/home/manjil/Desktop/Arcxen/Data"
output_dir = "/home/manjil/Desktop/Arcxen/Parsed Data 1"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each file in the input directory
for filename in os.listdir(input_path):
    if filename.endswith(".pdf"):  # Process only PDF files
        file_path = os.path.join(input_path, filename)
        print(f"Processing file: {file_path}")

        # Partition the PDF using the Unstructured API
        elements = partition_via_api(
            filename=file_path,
            api_key=os.environ["UNSTRUCTURED_API_KEY"],
            api_url=os.environ["UNSTRUCTURED_API_URL"],
            strategy="hi_res",
            additional_partition_args={
                "split_pdf_page": True,
                "split_pdf_allow_failed": True,
                "split_pdf_concurrency_level": 15,
            },
        )

        # Save the partitioned elements as JSON
        output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(elements_to_json(elements))

        print(f"Saved partitioned data to: {output_file}")