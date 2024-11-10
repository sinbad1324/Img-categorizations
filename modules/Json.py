import json

def Write(dict:dict| list , file:str)->None:
    Path = f"Data/{file}.json"
    json_object = json.dumps(dict, indent=4)
    with open(Path, "w") as json_file:
        json_file.write(json_object)

def Read(file:str) -> dict:
    Path = f"Data/{file}.json"
    with open(Path, "r") as json_file:
        return json.load(json_file) or None
    

def AddJSON(index , value ,file:str)-> None:
    dict = Read(file)
    dict[index] = value
    Write(dict, file)