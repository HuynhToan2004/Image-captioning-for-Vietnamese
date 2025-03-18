import json
#  sau khi down data từ Ngọc, chạy file này -> chạy file add_feature_temp
def merge_image_urls(base_file, image_url_file, output_file):

    with open(base_file, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    
    with open(image_url_file, 'r', encoding='utf-8') as f:
        image_url_data = json.load(f)
    
    for key, value in base_data.items():
        if key in image_url_data:
            base_data[key]['image_url'] = image_url_data[key]['image_url']
            base_data[key]['sents_byclip'] = base_data[key]['sents_byclip'].replace('//n//n', '\n\n')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(base_data, f, ensure_ascii=False, indent=4)

def merge_json(base_file1, base_file2, output_file):

    new_entry = {}

    with open(base_file1, 'r', encoding='utf-8') as f:
        base_file1 = json.load(f)
    
    with open(base_file2, 'r', encoding='utf-8') as f:
        base_file2 = json.load(f)

    for key, value in base_file1.items():
        new_entry[key] = value
    
    for key, value in base_file2.items():
        new_entry[key] = value

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_entry, f, ensure_ascii=False, indent=4)


base_file = '/data/npl/ICEK/VACNIC/data/final_train/final_content_13.json' 
image_url_file = '/data/npl/ICEK/VACNIC/data/train/restructured_content_13.json'  
output_file = '/data/npl/ICEK/VACNIC/data/final_train/fi_content_13.json'  

merge_image_urls(base_file, image_url_file, output_file)

print(f"Merged data has been saved to {output_file}")
