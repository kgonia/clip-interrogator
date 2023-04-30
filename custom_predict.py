import argparse
import os

from tqdm import tqdm

from clip_interrogator import Config, Interrogator, LabelTable, load_list
from PIL import Image

clip_model='ViT-L-14/openai'
# clip_model='ViT-H-14/laion2b_s32b_b79k'
# blip_model='blip-base'
blip_model='blip-large'
# blip_model='blip2-2.7b'
device = "cuda"


def get_all__images_file_paths(directory):
    file_paths = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

    for root, dirs, files in os.walk(directory):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                file_paths.append(os.path.join(root, file))

    return file_paths

def save_text_to_file(text, output_file_path):
    with open(output_file_path, 'w') as f:
        f.write(text)

ci = Interrogator(Config(
            clip_model_name=clip_model,
            caption_model_name=blip_model,
            caption_offload=False,
            device=device
        ))

camera_angles = LabelTable(load_list('custom/camera_angles.txt'), 'terms', ci)
focal_length = LabelTable(load_list('custom/focal_length.txt'), 'terms', ci)
lensens = LabelTable(load_list('custom/lensens.txt'), 'terms', ci)
light = LabelTable(load_list('custom/light.txt'), 'terms', ci)
rooms = LabelTable(load_list('custom/room.txt'), 'terms', ci)


def caption(path):
    image = Image.open(path).convert('RGB')
    image_features = ci.image_to_features(image)

    camera_r = camera_angles.rank(image_features, 1)[0]
    focal_length_r = focal_length.rank(image_features, 1)[0]
    lensens_r = lensens.rank(image_features, 1)[0]
    light_r = light.rank(image_features, 1)[0]
    room = rooms.rank(image_features, 1)[0]

    prompt = ci.interrogate_classic(image)
    prompt_full = f'{prompt}, {camera_r}, {focal_length_r}, {lensens_r}, {light_r}, {room}'
    return prompt_full


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default='examples')
    parser.add_argument('--out_image_dir', default='output_desc')

    args = parser.parse_args()

    file_paths = get_all__images_file_paths(args.image_dir)

    for path in tqdm(file_paths):
        caption_result = caption(path)
        file_name = os.path.splitext(os.path.basename(path))[0]
        text_path = os.path.join(args.out_image_dir, f'{file_name}.interrogate.caption')
        save_text_to_file(caption_result, text_path)
