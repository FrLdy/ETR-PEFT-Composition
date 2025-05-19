from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
import pandas as pd
from tqdm import tqdm

def add_to_embedded_dataset(embeddings, db) :
    faiss.normalize_L2(embeddings)
    db.add(np.array(embeddings))

def save_embedded_dataset(db, file_path) : 
    faiss.write_index(db, file_path)

def load_embedded_dataset(file_path) :
    return faiss.read_index(file_path)

def get_embedding(model, text) :
    return model.encode([text])

def load_json_data(path):
    path = path+"/sources/"
    all_features = []  
    files_list = os.listdir(path)
    files_list = [file for file in files_list if 'test' in file] + [file for file in files_list if 'test' not in file]

    for filename in files_list:
        if filename.endswith('.json'): 
            filepath = os.path.join(path, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                for line in file :
                    try:
                        data = json.loads(line)
                        all_features.extend([data])  
                    except json.JSONDecodeError as e:
                        print(f"Erreur de décodage json dans le fichier : {filename}")
    return all_features

def load_order_data(path):
    with open(path, 'r', encoding='utf-8') as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            print(f"Erreur de décodage json dans le fichier : {filename}")

def save_data_in_order(data, path) :
    with open(path, 'w', encoding='utf-8') as json_file:
        json_file.write('[')
        for i, feature in enumerate(data):
            if i > 0:
                json_file.write(',')
            json.dump(feature, json_file, ensure_ascii=False)
        json_file.write(']')

def load_csv_data(path) :
    path = path+"/sources/"
    all_features = []  
    files_list = os.listdir(path)
    files_list = [file for file in files_list if 'test' in file] + [file for file in files_list if 'test' not in file]
    
    for filename in files_list :
        if filename.endswith('.csv'):
            df = pd.read_csv(path+filename, delimiter=',')
            df = df.iloc[:, :2] 
            for index, value in df.iterrows() :
                all_features.extend([{"value":value['original'], "simple":value['simple']}])  

    return all_features

def get_most_similar_id(db, idx, k) :
    embedding_idx = db.reconstruct(idx)
    embedding_idx = embedding_idx.reshape(1, -1)
    distances, indices = db.search(embedding_idx, k)
    indices = indices[0].tolist()
    indices.remove(idx)
    return indices

if __name__=="__main__" :
    dim = 768
    data_path = "data/original_data/"

    model = SentenceTransformer(
        "jinaai/jina-embeddings-v2-base-en",
        trust_remote_code=True,
    ).to('cuda')


    etr_dataset = load_json_data(data_path+"etr-fr")
    index = faiss.IndexFlatIP(dim)
    for feature in tqdm(etr_dataset) :
        embedding = get_embedding(model, feature["falc"])
        add_to_embedded_dataset(embedding, index)
    save_embedded_dataset(index, data_path+"data_embedded/etr-fr.index")
    save_data_in_order(etr_dataset, data_path+"data_embedded/etr-fr.json")

    etr_politic_dataset = load_json_data(data_path+"etr-fr-politic")
    index = faiss.IndexFlatIP(dim)
    for feature in tqdm(etr_politic_dataset) :
        embedding = get_embedding(model, feature["falc"])
        add_to_embedded_dataset(embedding, index)
    save_embedded_dataset(index, data_path+"data_embedded/etr-fr-politic.index")
    save_data_in_order(etr_politic_dataset, data_path+"data_embedded/etr-fr-politic.json")

    wikilarge_dataset = load_csv_data(data_path+"wikilarge-fr")
    index = faiss.IndexFlatIP(dim)
    for feature in tqdm(wikilarge_dataset) :
        embedding = get_embedding(model, feature["simple"])
        add_to_embedded_dataset(embedding, index)
    save_embedded_dataset(index, data_path+"data_embedded/wikilarge-fr.index")
    save_data_in_order(wikilarge_dataset, data_path+"data_embedded/wikilarge-fr.json")

