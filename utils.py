from typing import List

import cv2
import uuid
import numpy as np
# import torch
import os
from io import BytesIO
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
import torch
import faiss
from loguru import logger
import boto3
from botocore.exceptions import ClientError

load_dotenv()   

# Generate a presigned S3 POST URL
s3_client = boto3.client('s3', 
                            aws_access_key_id=os.getenv('AWS_SERVER_PUBLIC_KEY'), 
                            aws_secret_access_key=os.getenv('AWS_SERVER_SECRET_KEY'), 
                            region_name=os.getenv('AWS_REGION_NAME')
)




# # Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def create_presigned_post(bucket_name, object_name,
                          fields=None, conditions=None, expiration=3600):
    """Generate a presigned URL S3 POST request to upload a file

    :param bucket_name: string
    :param object_name: string
    :param fields: Dictionary of prefilled form fields
    :param conditions: List of conditions to include in the policy
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Dictionary with the following keys:
        url: URL to post to
        fields: Dictionary of form fields and values to submit with the POST
    :return: None if error.
    """

    try:
        response = s3_client.generate_presigned_post(bucket_name,
                                                     object_name,
                                                     Fields=fields,
                                                     Conditions=conditions,
                                                     ExpiresIn=3600)
        
    except Exception as e:
        print(e)
        return None

    # The response contains the presigned URL and required fields
    return response

def download_video_from_s3(bucket_name, s3_file_key, local_file_path):
    """
    Download a video file from S3 bucket
    
    Parameters:
    bucket_name (str): Name of the S3 bucket
    s3_file_key (str): The key (path) of the file in S3
    local_file_path (str): Local path where the file will be saved
    """
    try:        
        
        # Download the file
        logger.info(f"Downloading {s3_file_key} from bucket {bucket_name}...")
        s3_client.download_file(bucket_name, s3_file_key, local_file_path)
        logger.info(f"Successfully downloaded to {local_file_path}")
        
    except ClientError as e:
        print(f"Error downloading file: {e}")
        return False
    return True

# Step 1: Preprocess video and extract frames
def extract_frames(video_path, chunk_size=5, overlap=2, fps=1):
    cap = cv2.VideoCapture(video_path)
    frames_dict = {}
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(frame_rate / fps)
    
    for start_sec in range(0, int(total_frames / frame_rate) - chunk_size, chunk_size - overlap):
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
        chunk_frames = []
        for _ in range(int(chunk_size * fps)):
            ret, frame = cap.read()
            if ret:
                chunk_frames.append(frame)
        frames_dict[start_sec] = chunk_frames
    cap.release()
    return frames_dict

# Step 2: Encode frames and create chunk embeddings
def encode_chunks(frames_dict):
    chunk_embeddings = {}
    for start_sec, frames in frames_dict.items():
        inputs = processor(images=frames, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        chunk_embeddings[start_sec] = image_features.mean(dim=0).numpy()  # Average over frames
    return chunk_embeddings

# Step 3: Build FAISS index
def build_index(embeddings):
    dimension = 512  # CLIP embedding size
    index = faiss.IndexFlatL2(dimension)
    vectors = np.stack(list(embeddings.values()))
    index.add(vectors)
    return index, list(embeddings.keys())

# Step 4: Query with text
def search_clip(query, index, timestamps):
    inputs = processor(text=query, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs).numpy()
    distances, indices = index.search(text_features, k=1)
    best_chunk_start = timestamps[indices[0][0]]
    return best_chunk_start

# # Step 5: Extract clip
# def extract_clip(video_path, start_sec, output_path, duration=10):
#     ffmpeg.input(video_path, ss=start_sec - 2, t=duration).output(output_path).run()
