import torch
from PIL import Image

# clip_model='ViT-H-14/laion2b_s32b_b79k'
# clip_model='ViT-L-14/openai'
clip_model='openai/clip-vit-large-patch14'
blip_model='blip-base'
device = "cuda"

# Load the CLIP model
from clip_interrogator import Interrogator, Config, load_list

ci = Interrogator(Config(
            clip_model_name=clip_model,
            caption_model_name=blip_model,
            device=device
        ))

# List of angle names
angle_names = [
    "Eye-Level Camera",
    "Low Angle",
    "High Angle",
    "Bird's-Eye View Angle",
    "Dutch Angle",
    "Close-Up",
    "Long Angle",
    "Medium Shot Camera",
]
angle_names = load_list('custom/camera_angles4.txt')

# Tokenize the angle names
text_inputs = ci.tokenize(angle_names).to(device)

# Obtain text embeddings
with torch.no_grad():
    text_embeddings = ci.clip_model.encode_text(text_inputs).cpu().numpy()

# Calculate similarity scores between embeddings
similarity_matrix = text_embeddings @ text_embeddings.T

# Print the similarity scores
for i, angle_name in enumerate(angle_names):
    print(f"{angle_name}: {similarity_matrix[i][i]:.4f}")
