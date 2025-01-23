import json
import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
import ollama

# Initialize Qdrant client
try:
    qdrant_client = QdrantClient(
        url="https://3318b213-e9d8-4338-9a77-b73c5fd8d79e.eu-west-1-0.aws.cloud.qdrant.io:6333",
        api_key="E3ldrxtzRA9hAGNkCF7-BeLjGaE4Of-KZVTyisARjyMukR-Q4Bc0eg",
    )
    print("Connected to Qdrant Cloud successfully!")
except Exception as e:
    print(f"Failed to connect to Qdrant Cloud: {e}")
    exit(1)

# Create a Qdrant collection for titles and content
collection_name = "legal_documents"

# Check if collection exists
if qdrant_client.collection_exists(collection_name):
    print(f"Collection '{collection_name}' exists. Deleting it...")
    qdrant_client.delete_collection(collection_name)

# Create a new collection
print(f"Creating collection '{collection_name}'...")
qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=768,  # Embedding size for Nomic Embed Text
        distance=models.Distance.COSINE  # Use cosine similarity
    )
)

# Directory containing the chunked JSON files
input_dir = "../Parsed Data 1/chunked_json_files"
output_dir = "../Parsed Data 1/metadata_qdrant"
os.makedirs(output_dir, exist_ok=True)

# Initialize lists to store metadata
metadata = []

# Load all chunked JSON files
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r") as f:
            chunked_data = json.load(f)

        # Debug: Print filename and number of sections
        print(f"Processing file: {filename} (sections: {len(chunked_data)})")

        # Process each section in the chunked data
        for section_id, section in enumerate(chunked_data):
            title = section.get("section_title", "").strip()  # Get title and remove whitespace
            content = " ".join(section.get("content", [])).strip()  # Get content and remove whitespace

            # Skip if title or content is empty
            if not title or not content:
                print(f"Skipping section {section_id} in file {filename} due to empty title or content")
                continue

            # Generate title embedding
            title_embedding = ollama.embeddings(model="nomic-embed-text", prompt=title)["embedding"]
            if not title_embedding or len(title_embedding) != 768:  # Ensure embedding is valid and has the correct size
                print(f"Failed to generate valid embedding for title: {title}")
                continue

            # Generate content embedding
            content_embedding = ollama.embeddings(model="nomic-embed-text", prompt=content)["embedding"]
            if not content_embedding or len(content_embedding) != 768:  # Ensure embedding is valid and has the correct size
                print(f"Failed to generate valid embedding for content: {content}")
                continue

            # Debug: Print embedding lengths
            print(f"Title Embedding Length: {len(title_embedding)}")
            print(f"Content Embedding Length: {len(content_embedding)}")

            # Store metadata
            metadata.append({
                "section_id": section_id,
                "section_title": title,
                "content": content,
                "filename": filename
            })

            # Upload embeddings to Qdrant
            qdrant_client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=section_id,  # Unique ID for the section
                        vector=title_embedding,  # Title embedding
                        payload={
                            "type": "title",
                            "section_title": title,
                            "content": content,
                            "filename": filename
                        }
                    ),
                    models.PointStruct(
                        id=section_id + 100000,  # Unique ID for content (offset to avoid conflicts)
                        vector=content_embedding,  # Content embedding
                        payload={
                            "type": "content",
                            "section_title": title,
                            "content": content,
                            "filename": filename
                        }
                    )
                ]
            )

# Save the metadata
with open(os.path.join(output_dir, "metadata_qdrant.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("Embeddings uploaded to Qdrant and metadata saved successfully!")