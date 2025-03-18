import json
import os
# from py_vncorenlp import VnCoreNLP
from collections import defaultdict
import re
import numpy as np
from extract_face import extract_faces, get_face_embedding
from extract_object import detect_objects, extract_object_embedding
from clip_get_sentences import retrieve_relevant_sentences
from tqdm import tqdm


def process_dataset(input_json_path, output_json_path):
    """
    Đọc dữ liệu từ input_json_path, xử lý và lưu vào output_json_path.
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = {}

    for hash_id, content in tqdm(data.items(), desc="Đang xử lý dataset"):
        new_entry = {}

        image_path = content.get("image_path", '')
        image_url = content.get("image_url",'')

        # Trích xuất face
        faces = extract_faces(image_path,image_url)
        faces_embbed = []
        if faces:
            # Embedding hết face trong faces
            for face in faces:
                face_emb = get_face_embedding(face)
                faces_embbed.append(face_emb)
        
            face_emb_path = os.path.join("/data/npl/ICEK/VACNIC/data/train/faces", f"{hash_id}.npy")
            np.save(face_emb_path, faces_embbed)
            new_entry["face_emb_dir"] = face_emb_path
        else: 
            new_entry["face_emb_dir"] = []


        objects, _ = detect_objects(image_path,image_url)
        objects_embbed = []
        if objects:
            # Embedding hết object trong objects
            for obj in objects:
                object_emb = extract_object_embedding(image_path,image_url, obj)
                objects_embbed.append(object_emb)

            object_emb_path = os.path.join(r"/data/npl/ICEK/VACNIC/data/train/objects", f"{hash_id}.npy")
            np.save(object_emb_path, objects_embbed)
            new_entry["obj_emb_dir"] = object_emb_path
        else: 
            new_entry["obj_emb_dir"] = []

        caption = content.get("caption", [])
        new_entry["caption"] = caption
        
        sents_byclip = content.get("sents_byclip", [])
        new_entry["sents_byclip"] = sents_byclip

        ner_cap = content.get('ner_cap')
        new_entry["ner_cap"] = ner_cap

        named_entites = content.get('named_entites')
        new_entry["named_entites"] = named_entites

        names_art = content.get('names_art')
        new_entry["names_art"] = names_art

        org_norp_art = content.get('org_norp_art')
        new_entry["org_norp_art"] = org_norp_art

        gpe_loc_art = content.get('gpe_loc_art')
        new_entry["gpe_loc_art"] = gpe_loc_art
        

        org_norp_cap = content.get('org_norp_cap',[])
        new_entry['org_norp_cap'] = org_norp_cap
        
        gpe_loc_cap = content.get('gpe_loc_cap',[])
        new_entry["gpe_loc_cap"] = gpe_loc_cap

        names = content.get('names')
        new_entry["names"] = names

        org_norp = content.get('org_norp')
        new_entry["org_norp"] = org_norp

        gpe_loc = content.get('gpe_loc')
        new_entry["gpe_loc"] = gpe_loc
        
        
        new_entry["image_path"] = image_path
        name_pos_cap = content.get('name_pos_cap')
        new_entry["name_pos_cap"] = name_pos_cap 
        new_entry['image_url'] = image_url
        new_data[hash_id] = new_entry

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)

    print(f"Dữ liệu đã được xử lý và lưu vào {output_json_path}")



if __name__ == "__main__":
    input_json = r"/data/npl/ICEK/VACNIC/data/final_train/re_content_13.json" 
    output_json = r"/data/npl/ICEK/VACNIC/data/final_train/final_content_13_3.json"  
    
    process_dataset(input_json, output_json)