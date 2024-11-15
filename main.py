import clip  
import torch
from PIL import Image
from LoadData import categoris , getComparePaths
from modules.GetImg import OpenImg


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


category_features = categoris() 

roblox_list= ["rbxassetid://18617723669"]

for item in getComparePaths(roblox_list):
    # Encoder l'image à classifier
    with torch.no_grad():
        input_features = model.encode_image(item["img"])
        input_features /= input_features.norm(dim=-1, keepdim=True) 

    # Calculer la similarité moyenne pour chaque catégorie
    similarities = {}
    for category, features in category_features.items():
        similarity = (input_features @ features.T).mean().item()  # Moyenne des similarités
        if similarity > .3:
            similarities[category] = similarity
        if similarity > .85:
                break
    if len(similarities) > 0:
        predicted_category = max(similarities, key=similarities.get)
    else:
        predicted_category = "Other"
    
    print(similarities)
    print(f"La catégorie prédite est : {predicted_category} pour cette img {item["img_name"]}")


