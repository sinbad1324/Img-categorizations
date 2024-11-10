import clip  
import torch
from PIL import Image
from modules.LoadData import getLoadedImages , getComparePaths
from modules.GetImg import OpenImg

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


category_features = getLoadedImages() # charger les images

roblox_list= ["rbxassetid://7216979807" , "rbxassetid://13401773180" ,"rbxassetid://13348242294"]

for item in getComparePaths(roblox_list):
    # Encoder l'image à classifier
    with torch.no_grad():
        input_features = model.encode_image(item["img"])
        input_features /= input_features.norm(dim=-1, keepdim=True) 

    # Calculer la similarité moyenne pour chaque catégorie
    similarities = {}
    for category, features in category_features.items():
        similarity = (input_features @ features.T).mean().item()  # Moyenne des similarités
        similarities[category] = similarity

    predicted_category = max(similarities, key=similarities.get)
    print(f"La catégorie prédite est : {predicted_category} pour cette img")
