import json 
import numpy as np
import hashlib
from tqdm import tqdm
from clip_get_sentences import retrieve_relevant_sentences

num = 11
with open(f'/data/npl/ICEK/VACNIC/data/train/content_{num}.json','r',encoding='utf-8') as f:
    original_data_dict = json.load(f)

total_articles = len(original_data_dict)

new_data_dict = {}
for article_id, article_data in tqdm(original_data_dict.items(), total=total_articles, desc='Processing Articles', unit='article'):
    context = article_data["context"]
    images = article_data.get("images", [])
    total_images = len(images)
    for image_data in tqdm(images, total=total_images, desc='Processing Images', unit='image', leave=False):
        image_path = image_data["path"]
        caption = image_data["caption"]
        image_url = image_data["url"]
        topk_sentences, _ = retrieve_relevant_sentences(image_path,image_url, context,0.23, 13)
        hash_id = hashlib.md5(image_path.encode('utf-8')).hexdigest()
        new_data_dict[hash_id] = {
            "caption": caption,
            'image_path': image_path,
            'image_url': image_url,
            "sents_byclip": topk_sentences,
            "face_emb_dir": [],
            "obj_emb_dir": [],
        }
    print(f'Xong 1 article, yeahhhhhh <3 <3 của file {num}')


    with open(f'/data/npl/ICEK/VACNIC/data/train/restructured_content_{num}.json', 'w', encoding='utf-8') as f:
        json.dump(new_data_dict, f, ensure_ascii=False, indent=4)


# 7,8,14, 19, 17, 16