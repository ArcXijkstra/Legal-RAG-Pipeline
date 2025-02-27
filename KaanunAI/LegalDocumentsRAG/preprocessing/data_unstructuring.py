from unstructured.partition.api import partition_via_api
from unstructured.staging.base import elements_to_json
import os
from dotenv import load_dotenv

load_dotenv()

# Set up API credentials

os.environ["UNSTRUCTURED_API_URL"] = "https://api.unstructured.io/general/v0/general"


# Input and output directories
input_path = "../Data/"
output_dir = "../Data/Parsed/"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each file in the input directory
def process_file():
    for filename in os.listdir(input_path):
        json_path = os.path.splitext(filename)[0] + ".json"
        if filename.endswith(".pdf") and json_path not in os.listdir("../Data/Parsed/"):  # Process only PDF files
            file_path = os.path.join(input_path, filename)
            print(f"Processing file: {file_path}")

            # Partition the PDF using the Unstructured API
            elements = partition_via_api(
                filename=file_path,
                api_key=os.getenv("UNSTRUCTURED_API_KEY2"),
                api_url=os.getenv("UNSTRUCTURED_API_URL"),
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

if __name__ == "__main__":
    process_file()
        
        