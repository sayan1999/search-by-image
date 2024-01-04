import streamlit as st
import torch
import os
import torchvision
import faiss
from PIL import Image
import traceback
from tqdm import tqdm
from PIL import ImageFile
from slugify import slugify
import opendatasets as od
import json
import argparse
from streamlit_cropper import st_cropper
from azure.storage.blob import BlobServiceClient
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms
import numpy as np
import faiss.contrib.torch_utils

BATCH_SIZE = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ImageFile.LOAD_TRUNCATED_IMAGES = True
FOLDER = "images/"
NUM_TREES = 100
FEATURES = 1000
FILETYPES = [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
LIBRARIES = [
    "https://www.kaggle.com/datasets/athota1/caltech101",
    "https://www.kaggle.com/datasets/gpiosenka/sports-classification",
    "https://www.kaggle.com/datasets/puneet6060/intel-image-classification",
    "https://www.kaggle.com/datasets/kkhandekar/image-dataset",
]


@st.cache_resource
def dl_embeddings():
    """dl pretrained embeddings in production environment instead of creating"""
    # Connect to your Blob Storage account
    if os.path.isfile(f"{slugify(FOLDER)}.index"):
        print("Embeddings files already exists, skip download")
        return
    connect_str = st.secrets["connectionstring"]
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Specify container and blob names
    container_name = "imagessearch"
    blob_name = f"{slugify(FOLDER)}.index"

    # Get a reference to the blob
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=blob_name
    )

    # Download the binary data
    download_file_path = f"{slugify(FOLDER)}.index"  # Path to save the downloaded file
    with open(download_file_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

    print(f"File downloaded to: {download_file_path}")


@st.cache_resource
def load_dataset():
    with open("kaggle.json", "w+") as f:
        json.dump(
            {
                "username": st.secrets["username"],
                "key": st.secrets["key"],
            },
            f,
        )
    for lib in LIBRARIES:
        od.download(
            lib,
            "images/",
        )


# Load a pre-trained image feature extractor model
@st.cache_resource
def load_model():
    """Loads a pre-trained image feature extractor model."""
    print("Loading pretrained model...")
    model = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub",
        "nvidia_efficientnet_b0",
        pretrained=True,
    )
    model.eval()  # Set model to evaluation mode
    return model


# Get all file paths within a folder and its subfolders
@st.cache_data
def get_all_file_paths(folder_path):
    """Returns a list of all file paths within a folder and its subfolders."""
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.lower().endswith(tuple(FILETYPES)):
                continue
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    print(f"Total {len(file_paths)} image files present")
    return sorted(file_paths)


# Load all the images from file paths
@st.cache_data
def load_images(file_paths):
    """Load all the images from file paths."""
    print("Loading images: ")
    images = list()
    for path in tqdm(file_paths):
        try:
            images.append(Image.open(path).resize([224, 224]))
        except BaseException as e:
            print("error loading ", path, e)
    return images


def load_image(file_path):
    """Load all the images from file paths."""
    try:
        image = Image.open(file_path).resize([224, 224])
        return image
    except BaseException as e:
        print("Error loading ", file_path, e)


# Function to preprocess images
def preprocess_image(image):
    """Preprocesses an image for feature extraction."""

    if image.mode == "RGB":  # Already has 3 channels
        pass  # No need to modify
    elif image.mode == "L":  # Grayscale image
        image = image.convert("RGB")  # Convert to 3-channel RGB
    else:  # Image has more than 3 channels
        image = image.convert(
            "RGB"
        )  # Convert to 3-channel RGB, discarding extra channels
    preprocess = torchvision.transforms.Compose(
        [
            # torchvision.transforms.Resize(224),  # Adjust for EfficientNet input size
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return preprocess(image)


class ImageLoader(Dataset):
    def __init__(self, image_files, transform, load_image):
        self.transform = transform
        self.load_image = load_image
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transform(self.load_image(self.image_files[index]))


# Extract features from a list of images
def extract_features(file_paths, model):
    """Extracts features from a list of images."""
    print("Extracting features:")
    loader = DataLoader(
        ImageLoader(file_paths, transform=preprocess_image, load_image=load_image),
        batch_size=BATCH_SIZE,
    )
    features = []
    model = model.to(DEVICE)
    with torch.no_grad():
        for batch_idx, images in enumerate(tqdm(loader)):
            images = images.to(DEVICE)
            features.append(model(images))
    return torch.cat(features)


# Build an Annoy index for efficient similarity search
def build_annoy_index(features):
    """Builds an Annoy index for efficient similarity search."""
    print("Building faiss index:")
    f = features[0].shape[0]  # Feature dimensionality
    index = faiss.IndexIDMap(faiss.IndexFlatIP(f))
    index.add_with_ids(
        features.cpu().detach().numpy(), np.array(range(len(features)))
    )  # Adjust num_trees for accuracy vs. speed trade-off
    print("built faiss index:")
    return index


# Perform reverse image search
def search_similar_images(query_image, num_results, f=FEATURES):
    """Finds similar images based on a query image feature."""
    index = faiss.read_index(f"{slugify(FOLDER)}.index")
    model = load_model().to(DEVICE)
    # Extract features and search
    proc_image = preprocess_image(query_image).unsqueeze(0).to(DEVICE)
    query_feature = model(proc_image)
    query_feature = query_feature.cpu().detach().numpy()
    distances, nearest_neighbors = index.search(
        query_feature,
        num_results,
    )
    return query_image, nearest_neighbors[0], distances[0]


@st.cache_data
def save_embedding(folder=FOLDER):
    if os.path.isfile(f"{slugify(FOLDER)}.index"):
        print("skipping recreating image embeddings")
        return
    print("Performing image embeddings")
    model = load_model()  # Load the model once
    file_paths = get_all_file_paths(folder_path=folder)
    # images = load_images(file_paths)
    features = extract_features(file_paths, model)
    index = build_annoy_index(features)
    faiss.write_index(index, f"{slugify(FOLDER)}.index")


def display_image(idx, dist):
    file_paths = get_all_file_paths(folder_path=FOLDER)
    image = Image.open(file_paths[idx])
    st.image(image.resize([256, 256]))
    st.markdown("SimScore: -" + str(round(dist, 2)))
    # st.markdown(file_paths[idx])


if __name__ == "__main__":
    # Main app logic
    st.set_page_config(layout="wide")
    st.title("Reverse Image Search App")

    try:
        load_dataset()
        # download dev embeddings if not developement environment
        ap = argparse.ArgumentParser()
        ap.add_argument("--dev", action="store_true")
        if not ap.parse_args().dev:
            dl_embeddings()
        save_embedding(FOLDER)

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image like a car, cat, dog, flower, fruits, bike, aeroplane, person",
            type=FILETYPES,
        )

        n_matches = st.slider(
            "Num of matches to be displayed", min_value=3, max_value=100, value=5
        )

        if uploaded_file is not None:
            query_image = Image.open(uploaded_file)
            cropped = st_cropper(query_image)
            query_image, nearest_neighbors, distances = search_similar_images(
                cropped.resize([256, 256]), n_matches
            )

            st.subheader("Similar Images:")
            cols = st.columns([1] * 5)
            for i, (idx, dist) in enumerate(
                zip(
                    *[
                        nearest_neighbors,
                        distances,
                    ]
                )
            ):
                with cols[i % 5]:
                    # Display results
                    display_image(idx, dist)
        else:
            st.write("Please upload an image to start searching.")

    except Exception as e:
        traceback.print_exc()
        print(e)
        st.error("An error occurred: {}".format(e))
