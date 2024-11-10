import clip
import torch
from PIL import Image
import os

from modules.GetImg import OpenImg

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

data = r"C:\Users\41794\Desktop\Img-categorizations\assets\Data\Cat"
categories = {}

def getLoadedImages():
    for path, folder, files in os.walk(data):
        if len(folder)  <= 0 and len(files)  >= 1:
            folder_name = os.path.basename(path)
            if folder_name not in categories:
                categories[folder_name] = []
            for v in files:
                if v.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    categories[folder_name].append(os.path.join(path, v))
    category_features = {}
    for category, image_paths in categories.items():
        if not image_paths : continue
        images = [preprocess(Image.open(path).convert('RGB')).unsqueeze(0).to(device) for path in image_paths]
        images_tensor = torch.cat(images)

        with torch.no_grad():
            features = model.encode_image(images_tensor)
            features /= features.norm(dim=-1, keepdim=True)
        category_features[category] = features
    return category_features

compare = r"C:\Users\41794\Desktop\Img-categorizations\assets\compare"
def getComparePaths(imagesList:list=[]):
    opened_images = []
    for path, folder, files in os.walk(compare):
        for img in files:
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                imagesList.append(os.path.join(path, img))
    
    for path in imagesList:
        img = OpenImg(path)
        if img == None : return None
        opened_images.append({
                    "img": preprocess(img.convert('RGB')).unsqueeze(0).to(device),
                    # "img_path": os.path.join(path, img),
                    # "img_name":img
            })
    return opened_images


