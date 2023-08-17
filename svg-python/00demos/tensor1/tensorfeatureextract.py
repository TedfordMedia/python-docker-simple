import numpy as np
import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image

# Load the VGG16 model pre-trained on ImageNet data
model_vgg16 = VGG16(weights='imagenet', include_top=False)


def extract_features(img_path):
    # Load an image file that contains the desired visual, resizing to 224x224 pixels (required input size for VGG16)
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert the image to a numpy array and preprocess it for the model
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    # Get the features from the image
    features = model_vgg16.predict(preprocessed_img)

    # Flatten the features to create a feature vector
    feature_vector = features.flatten()

    return feature_vector


# Iterate through the tridentFull_pngs folder and process each image
folder_path = "tridentFull_pngs"
img_index = 0
feature_vectors = []

while True:
    img_path = os.path.join(folder_path, f"trident_{img_index}.png")
    if os.path.exists(img_path):
        print(f"Processing {img_path}")
        feature_vector = extract_features(img_path)
        feature_vectors.append(feature_vector)
        img_index += 1
    else:
        break

# Save the extracted feature vectors to a .npy file
output_path = os.path.join(folder_path, "extracted_features.npy")
np.save(output_path, feature_vectors)

print(f"Processed {img_index} images. Features saved to {output_path}.")
