import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import os


# Define the MLP model that takes the difference between two embeddings
class FaceVerificationMLP(nn.Module):
    def __init__(self, input_dim=512):
        super(FaceVerificationMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Output: [same_person_prob, different_person_prob]
        )

    def forward(self, x):
        return self.model(x)


# Optional Preprocessing: Resize, smooth, convert to tensor, and normalize the image if they require it in one of the tasks. Note: this is a sample transoformation, modify it accoringly to the task
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# Load MTCNN for face detection, note: you should adjust the size wrt data and model, it can be done directly here or in the transform defined above
mtcnn = MTCNN(image_size=160, margin=20)

# Load FaceNet for embedding extraction (we'll be using a pretrained model)
facenet = InceptionResnetV1(pretrained='vggface2').eval()


# Function to get face embedding from an image path
def get_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    # When needed add the preprocessing img = transform(img)
    face = mtcnn(img)  # returns a cropped, aligned face
    if face is None:
        raise ValueError(f"No face detected in {image_path}")
    face_embedding = facenet(face.unsqueeze(0))  # Add batch dimension
    return face_embedding.detach()


# Function to compare two images and get the absolute difference vector
def get_diff_vector(img1_path, img2_path):
    emb1 = get_embedding(img1_path)
    emb2 = get_embedding(img2_path)
    return torch.abs(emb1 - emb2)

# Prediction Example, for additional experiments you may want to return the decision in numeric form or add model certenity
def predict_same_person(img1_path, img2_path, model):
    model.eval()
    diff = get_diff_vector(img1_path, img2_path)
    output = model(diff)
    _, predicted = torch.max(output, 1)
    return 'Same person' if predicted.item() == 1 else 'Different people'


# Example augmentation function
def augment_image(image, augment_type="gaussian_noise"):
    if augment_type == "gaussian_noise":
        # Add Gaussian noise
        image_np = np.array(image).astype(np.float32)
        # Modify the parameter value to adjust noise level if needed
        noise = np.random.normal(0, 25, image_np.shape)
        noisy_image = image_np + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        augmented = Image.fromarray(noisy_image)
    elif augment_type == "blur":
        # Apply Gaussian blur with radius 3
        augmented = image.filter(ImageFilter.GaussianBlur(radius=3))
    elif augment_type == "increased_lighting":
        # Increase brightness by 50%
        enhancer = ImageEnhance.Brightness(image)
        augmented = enhancer.enhance(1.5)
    else:
        augmented = image
    return augmented
