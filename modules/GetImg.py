import requests
from io import BytesIO
from PIL import Image
import xml.etree.ElementTree as ET


def GetReelImg(assetsId):
    OriginalID= ""
    try:
        Res1 = requests.get(f"https://assetdelivery.roblox.com/v2/assetId/{assetsId}")
        Res1.raise_for_status()
        ResJson = Res1.json()
        if Res1 and ResJson and ResJson["locations"] and  ResJson["locations"][0] :
            OriginalID = assetsId
            url = ResJson["locations"][0]["location"]
            response = requests.get(url)
            response.raise_for_status()
            content_str:str = str(response.content)
            content_byte = response.content
            if "<roblox" and "<url>" in content_str:
                root = ET.fromstring(content_byte)
                url_element = root.find('.//Content/url') 
                if url_element is not None:
                    theUrl:str = url_element.text.strip() 
                    tableStr:list[str] = theUrl.split("/")[-1]
                    newAssetId = tableStr[4:]
                    OriginalID = newAssetId
                    print("old: " ,assetsId , "new: " ,newAssetId )
                    return GetReelImg(newAssetId)  
            image_data = BytesIO(content_byte)
            return  image_data , OriginalID
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête : {e}")
    except KeyError as e:
        print(f"Clé manquante dans le JSON : {e}")
    except ET.ParseError as e:
        print(f"Erreur lors de l'analyse XML : {e}")

    return None, OriginalID 


def OpenImg(path:str):
    if "C:" in path:
       return Image.open(path)
    elif "rbxassetid://" in path:
        newpath = path.split("//")[-1] 
        image_data , assete = GetReelImg(newpath)
        return Image.open(image_data) 
    elif path.isdigit():
        image_data , assete = GetReelImg(path)
        return Image.open(image_data) 
    elif "https://create.roblox.com" in path:
        newpath = path.split("/")[-2] 
        image_data , assete = GetReelImg(newpath)
        return Image.open(image_data) 






