import streamlit as st
import torch
import os
import torchvision
from annoy import AnnoyIndex
from PIL import Image
import traceback
from tqdm import tqdm
from PIL import ImageFile
from slugify import slugify
import opendatasets as od
import json
import argparse


ImageFile.LOAD_TRUNCATED_IMAGES = True
FOLDER = "images/"
NUM_TREES = 100
FEATURES = 1000
FILETYPES = [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]

from azure.storage.blob import BlobServiceClient


@st.cache_resource
def dl_embeddings():
    """dl pretrained embeddings in production environment instead of creating"""
    # Connect to your Blob Storage account
    connect_str = st.secrets["connectionstring"]
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Specify container and blob names
    container_name = "imagessearch"
    blob_name = f"{slugify(FOLDER)}.tree"

    # Get a reference to the blob
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=blob_name
    )

    # Download the binary data
    download_file_path = f"{slugify(FOLDER)}.tree"  # Path to save the downloaded file
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
    od.download(
        "https://www.kaggle.com/datasets/kkhandekar/image-dataset",
        "images/",
    )


# Load a pre-trained image feature extractor model
@st.cache_resource
def load_model():
    """Loads a pre-trained image feature extractor model."""
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


# Extract features from a list of images
def extract_features(images, model):
    """Extracts features from a list of images."""
    print("Extracting features:")
    features = []
    for image in images:
        with torch.no_grad():
            feature = model(preprocess_image(image).unsqueeze(0)).squeeze(0)
            features.append(feature.numpy())
    return features


# Build an Annoy index for efficient similarity search
def build_annoy_index(features):
    """Builds an Annoy index for efficient similarity search."""
    print("Building annoy index:")
    f = features[0].shape[0]  # Feature dimensionality
    t = AnnoyIndex(f, "angular")  # Use angular distance for image features
    for i, feature in tqdm(enumerate(features)):
        t.add_item(i, feature)
    t.build(NUM_TREES)  # Adjust num_trees for accuracy vs. speed trade-off
    return t


# Perform reverse image search
def search_similar_images(uploaded_file, f=FEATURES, num_results=5):
    """Finds similar images based on a query image feature."""
    index = AnnoyIndex(f, "angular")
    index.load(f"{slugify(FOLDER)}.tree")
    query_image = Image.open(uploaded_file)
    model = load_model()
    # Extract features and search
    query_feature = (
        model(preprocess_image(query_image).unsqueeze(0)).squeeze(0).detach().numpy()
    )
    nearest_neighbors, distances = index.get_nns_by_vector(
        query_feature, num_results, include_distances=True
    )
    return query_image, nearest_neighbors, distances


@st.cache_data
def save_embedding(folder=FOLDER):
    if os.path.isfile(f"{slugify(FOLDER)}.tree"):
        return
    model = load_model()  # Load the model once
    file_paths = get_all_file_paths(folder_path=folder)
    images = load_images(file_paths)
    features = extract_features(images, model)
    index = build_annoy_index(features)
    index.save(f"{slugify(FOLDER)}.tree")


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
            query_image, nearest_neighbors, distances = search_similar_images(
                uploaded_file, num_results=n_matches
            )

            st.image(query_image.resize([256, 256]), caption="Query Image", width=200)
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
