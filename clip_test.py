import os

import torch
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

from clip_interrogator import load_list

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

image_path = "./examples/86b240783cd714e426cc9ff442ff6842.jpg"
directory = "./examples"

for root, dirs, files in os.walk(directory):
    for file in files:
        image_path = (os.path.join(root, file))

        image = Image.open(image_path)

        text = load_list('custom/camera_angles.txt')
        inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

        # Assuming `probs` is a tensor with the shape (1, num_angle_names)
        highest_score_index = torch.argmax(probs, dim=1).item()

        # Get the angle name with the highest score
        best_angle_name = text[highest_score_index]

        print(f"{image_path}: {best_angle_name}")