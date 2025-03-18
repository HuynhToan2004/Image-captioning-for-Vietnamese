from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import json
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
from torch.utils.data import Dataset, DataLoader
# import unidecode
import requests
import numpy as np
import copy
from io import BytesIO
import re
# import clip 
# import stanza 
MAX_PIXELS = 89478485
RESAMPLING = Image.LANCZOS  

# 1. Khởi tạo NER
# nlp = stanza.Pipeline(lang='vi', processors='tokenize,ner')

# 2. Khởi tạo Tokenizer và Mô hình BERT
bart_tokenizer = AutoTokenizer.from_pretrained("/data/npl/ICEK/VACNIC/src/data/assest/bartpho-syllable")
bart_model = AutoModelForSeq2SeqLM.from_pretrained("/data/npl/ICEK/VACNIC/src/data/assest/bartpho-syllable")

# Thêm các token đặc biệt vào BERT Tokenizer
additional_special_tokens = ['<PERSON>', '<ORGNORP>', '<GPELOC>', '<LOCATION>','<NONAME>']
bart_tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
bart_model.resize_token_embeddings(len(bart_tokenizer))
person_token_id = bart_tokenizer.convert_tokens_to_ids("<PERSON>")
# 3. Đưa mô hình BERT vào device và đặt ở chế độ đánh giá
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bart_model.to(device)
bart_model.eval()

# #### CLIPPP
from sentence_transformers import SentenceTransformer
from torchvision.transforms import ToPILImage

img_model = SentenceTransformer('/data/npl/ICEK/VACNIC/src/data/assest/clip-ViT-B-32')
text_model = SentenceTransformer('/data/npl/ICEK/VACNIC/src/data/assest/clip-ViT-B-32-multilingual-v1')
clip_tokenizer = text_model.tokenizer  
# import open_clip
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
# model = model.to(device)  # Chuyển mô hình lên đúng thiết bị

# clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")


# Tokenize văn bản
# text_tokens = clip_tokenizer(texts).to(device)  # Đảm bảo text_tokens cũng ở trên `device`

# def get_entities(ner_results):
#     """
#     Trích xuất các thực thể từ kết quả NER của pipeline Transformers.
    
#     Trả về:
#         - Danh sách các dictionary chứa 'text', 'label', và 'position'.
#     """
#     entities = []
#     for ent in ner_results:
#         entities.append({
#             'text': ent['word'],
#             'label': ent['entity_group'],
#             'position': [ent['start'], ent['end']]
#         })
#     return entities

def is_abbreviation(entity):
    """
    Kiểm tra xem một chuỗi có phải viết tắt theo dạng A.B. hay không.
    Pattern: A. B. (có thể thêm tên tiếp theo sau)
    """
    pattern = r'\b([A-Z]\.)+([A-Z][a-z]*)?\b'
    return re.match(pattern, entity) is not None

def get_entities(doc):
    """
    Trích xuất các thực thể từ kết quả NER của Stanza.

    Tham số:
        doc: đối tượng Document của Stanza sau khi chạy pipeline NER.
    
    Trả về:
        - Danh sách các dictionary chứa 'text', 'label', và 'position'.
    """
    entities = []
    for sentence in doc.sentences:
        for ent in sentence.ents:
            ent_text = ent.text.strip()
            if '.' in ent_text and not is_abbreviation(ent_text):
                continue
            entities.append({
                'text': ent_text,
                'label': ent.type,
                'position': [ent.start_char, ent.end_char]
            })
    return entities

def make_ner_dict_by_type(ent_list, ent_type_list):

    """
    Tạo một dictionary các thực thể có tên với các nhãn NER duy nhất theo từng loại.
    
    Trả về:
        - entities_type: Dictionary chứa thông tin về các thực thể với nhãn NER đã được gán.
        - start_pos_list: Danh sách các vị trí bắt đầu của từng thực thể.
        - person_count, org_count, gpe_count: Số lượng thực thể từng loại đã được gán nhãn.
    """
    person_count = 1  # Tổng số thực thể loại PERSON
    org_count = 1     # Tổng số thực thể loại ORGANIZATION/NORP
    gpe_count = 1     # Tổng số thực thể loại GPE/LOC
    
    unique_ner_dict = {}
    new_ner_type_list = []
    
    for i, ent in enumerate(ent_list):
        if ent in unique_ner_dict:
            new_ner_type_list.append(unique_ner_dict[ent])
        elif ent_type_list[i] == "PERSON":
            ner_type = "<PERSON>"
            unique_ner_dict[ent] = ner_type
            new_ner_type_list.append(ner_type)
            person_count += 1
        elif ent_type_list[i] in ["ORGANIZATION", "ORG", "NORP"]:
            ner_type = "<ORGNORP>"
            unique_ner_dict[ent] = ner_type
            new_ner_type_list.append(ner_type)
            org_count += 1
        elif ent_type_list[i] in ["GPE", "LOC"]:
            ner_type = "<GPELOC>"
            unique_ner_dict[ent] = ner_type
            new_ner_type_list.append(ner_type)
            gpe_count += 1
        else:
            # Xử lý các loại thực thể khác nếu có
            ner_type = f"<{ent_type_list[i]}>"
            unique_ner_dict[ent] = ner_type
            new_ner_type_list.append(ner_type)
    
    entities_type = {}  # Dictionary chứa thông tin các thực thể với nhãn NER đã được gán
    for i, ent in enumerate(ent_list):
        entities_type[i] = {
            'text': ent,
            'label': new_ner_type_list[i],
            'position': [] 
        }
    
    start_pos_list = []
    
    return entities_type, start_pos_list, person_count, org_count, gpe_count

def replace_entities_with_labels(article_full, entities_type, ent_list):
    """
    Thay thế các thực thể trong bài viết bằng các nhãn NER đặc biệt cho từng từ.
    
    Tham số:
        - article_full (str): Văn bản đầy đủ của bài viết.
        - entities_type (dict): Dictionary chứa thông tin về các thực thể với nhãn NER đã được gán.
        - ent_list (list of str): Danh sách các thực thể có tên.
    
    Trả về:
        - new_article (str): Văn bản đã được thay thế các thực thể bằng nhãn NER cho từng từ.
    """
    # Sắp xếp các thực thể theo độ dài giảm dần để tránh việc thay thế một thực thể nhỏ trong thực thể lớn
    sorted_entities = sorted(zip(ent_list, entities_type.values()), key=lambda x: len(x[0]), reverse=True)
    
    new_article = article_full
    for ent, ent_info in sorted_entities:
        label = ent_info['label']
        # Tách thực thể thành các từ
        ent_words = ent.split()
        # Thay thế mỗi từ bằng label
        replaced = ' '.join([label] * len(ent_words))
        # Thay thế trong bài viết
        new_article = new_article.replace(ent, replaced)
    return new_article
#  article_full_text_dir, article_all_ent_by_count_dir, article_all_ent_unique_by_count_dir,
def save_full_processed_articles_all_ent_by_count(data_dict, bart_tokenizer, ner_pipeline):
    """
    Xử lý các bài viết, trích xuất thực thể có tên, thay thế thực thể bằng nhãn NER và lưu kết quả.
    
    Tham số:
        - data_dict: Từ điển chứa các khóa (ID bài viết).
        - article_full_text_dir: Thư mục chứa các tệp văn bản đầy đủ của bài viết.
        - article_all_ent_by_count_dir: Thư mục để lưu trữ các tệp JSON chứa các ID sau khi xử lý.
        - article_all_ent_unique_by_count_dir: Thư mục để lưu trữ các tệp văn bản đã xử lý (đang bị comment trong mã).
        - bart_tokenizer: Tokenizer từ mô hình BERT.
        - ner_pipeline: Pipeline NER từ thư viện Transformers.
    """
    for key in tqdm(data_dict.keys(), desc="Processing articles"):
        # Đọc nội dung bài viết
        # article_path = os.path.join(article_full_text_dir, f"{key}.txt")
        # with open(article_path, 'r', encoding='utf-8') as f:
        #     article_full = f.read()
        article_full = data_dict[key].get('sents_byclip')
        # paragraphs = article_full.split("//n//n")
        article_full = article_full.replace('_',' ')
        article_full = article_full.replace("\n\n",'')
        paragraphs = article_full.split(',')
        
        all_entities = []
        for para in paragraphs:
            if para.strip() == "":
                continue
            ner_results = ner_pipeline(para)
            entities = get_entities(ner_results)
            all_entities.extend(entities)
        
        # Thực hiện NER
        ent_list = [ent["text"] for ent in all_entities]
        ent_type_list = [ent["label"] for ent in all_entities]
        
        if not ent_list:
            # Nếu không có thực thể nào, tiếp tục xử lý bài viết tiếp theo
            continue
        
        # Tạo dictionary NER theo loại
        entities_type, start_pos_list, _, _, _ = make_ner_dict_by_type(ent_list, ent_type_list)
        
        # Thay thế các thực thể trong văn bản bằng nhãn NER
        new_article = replace_entities_with_labels(article_full, entities_type, ent_list)
        
        # Tokenize với BERT tokenizer
        bart_inputs = bart_tokenizer(
            new_article,
            add_special_tokens=True,       
            return_tensors="pt",            
            padding=True,                  
            truncation=True,              
        )
        
        input_ids = bart_inputs["input_ids"]            # Tensor chứa các token IDs
        attention_mask = bart_inputs["attention_mask"]  # Tensor chứa attention mask
        
        # Chuyển tensor sang CPU và chuyển đổi thành danh sách để có thể lưu dưới dạng JSON
        input_ids = input_ids.cpu().tolist()
        attention_mask = attention_mask.cpu().tolist()
        
        # Lưu kết quả vào tệp JSON
        article_all_ent_by_count_out_dir = os.path.join('/data/npl/ICEK/VACNIC/data/train/article_all_ent_by_count_dir', f"{key}.json")
        if not os.path.isfile(article_all_ent_by_count_out_dir):
            article_ids_ner = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            with open(article_all_ent_by_count_out_dir, "w", encoding='utf-8') as f:
                json.dump(article_ids_ner, f, ensure_ascii=False, indent=4)
        # create_article_ner_mask_dict()
        # article_all_ent_unique_by_count_out_dir = os.path.join(article_all_ent_unique_by_count_dir, f"{key}.txt")
        # with open(article_all_ent_unique_by_count_out_dir, "w", encoding='utf-8') as f:
        #     f.write(new_article)

def find_first_sublist(seq, sublist, start=0):
    """
    Tìm kiếm một sublist trong một danh sách lớn hơn bắt đầu từ vị trí 'start'.
    
    Trả về:
        - Tuple (start_index, end_index) nếu tìm thấy.
        - None nếu không tìm thấy.
    """
    length = len(sublist)
    for index in range(start, len(seq) - length + 1):
        if seq[index:index+length] == sublist:
            return index, index+length
    return None

def replace_sublist(seq, sublist, replacement):
    """
    Thay thế tất cả các xuất hiện của sublist trong seq bằng replacement.
    
    Trả về:
        - Danh sách seq sau khi thay thế.
    """
    length = len(sublist)
    replacement_length = len(replacement)
    index = 0
    while index <= len(seq) - length:
        if seq[index:index+length] == sublist:
            seq[index:index+length] = replacement
            index += replacement_length
        else:
            index += 1
    return seq

def make_new_article_ids_all_ent(article_full, ent_list, entities_type, tokenizer):
    """
    Tạo một bản sao của input_ids từ bài viết gốc nhưng với các thực thể được thay thế bằng các nhãn NER đặc biệt.
    
    Trả về:
        - Dictionary chứa 'input_ids' đã được thay thế.
    """
    article_ids_ner = tokenizer(article_full)["input_ids"]
    counter = 0
    for ent in ent_list:
        # Tokenize thực thể gốc
        ent_ids_original = tokenizer(ent, add_special_tokens=False)["input_ids"]
        
        # Tìm vị trí xuất hiện đầu tiên của thực thể trong input_ids
        idx = find_first_sublist(article_ids_ner, ent_ids_original, start=0)
        if idx is not None:
            # Thay thế thực thể bằng nhãn NER đơn giản
            ner_label = entities_type[counter]["label"]
            ner_chain_ids = tokenizer(ner_label, add_special_tokens=False)["input_ids"]
            # Thay thế toàn bộ thực thể bằng một nhãn NER duy nhất
            article_ids_ner = replace_sublist(article_ids_ner, ent_ids_original, ner_chain_ids)
        else:
            ent_ids_original_start = tokenizer(ent, add_special_tokens=False)["input_ids"]
            ner_label = entities_type[counter]["label"]
            ner_chain_ids = tokenizer(ner_label, add_special_tokens=False)["input_ids"]
            article_ids_ner = replace_sublist(article_ids_ner, ent_ids_original_start, ner_chain_ids)
        
        counter += 1
    return {"input_ids": article_ids_ner}


def get_person_ids_position(article_ids_replaced,  person_token_id=person_token_id, article_max_length=512, is_tgt_input=False):
    """
    Tìm vị trí của các thực thể PERSON trong danh sách các token ID của bài viết.

    Tham số:
        - article_ids_replaced (list of int): Danh sách các token ID của bài viết đã được thay thế.
        - person_token_id (int): mã token của <PERSON>.
        - article_max_length (int): Độ dài tối đa của bài viết để xem xét.
        - is_tgt_input (bool): Cờ xác định cách tính vị trí (input hay target).

    Trả về:
        - position_list (list of list of int): Danh sách các vị trí của các thực thể PERSON.
    """

    position_list = []
    i = 0
    while i < len(article_ids_replaced) and i < article_max_length:
        position_i = []
        if article_ids_replaced[i] == person_token_id:
            if is_tgt_input:
                position_i.append(i + 1)
            else:
                position_i.append(i)
            # Tìm kết thúc của thực thể PERSON
            j = i + 1
            while j < len(article_ids_replaced) and j < article_max_length:
                if article_ids_replaced[j] == person_token_id:
                    j += 1
                else:
                    break
            if is_tgt_input:
                position_i.append(j)
            else:
                position_i.append(j - 1)
            position_list.append(position_i)
            i = j  # Di chuyển chỉ số đến sau thực thể hiện tại
        else:
            i += 1
    return position_list



def make_new_entity_ids(caption, ent_list, tokenizer, ent_separator="<ent>", max_length=80):
    """
    Hàm này tìm và mã hóa các thực thể trong `caption` thành dạng token ID. 
    Mục tiêu tương tự với hàm nguyên bản:
    - Token hóa caption
    - Với mỗi thực thể:
      + Tìm tokenization của thực thể trong caption_ids_ner
      + Nếu tìm thấy, thêm vào ent_ids_flatten và ent_ids_separate
      + Nếu không tìm thấy ở dạng ' ent', thử dạng 'ent' (không dấu cách)
    - Nếu không tìm thấy thực thể nào, thêm <NONAME>.
    - Thêm [BOS], [EOS], PAD đến max_length.
    - Trả về tensor ent_ids_flatten và danh sách ent_ids_separate.
    """

    # Lấy ID cho các token đặc biệt
    bos_id = tokenizer.convert_tokens_to_ids('[BOS]')
    eos_id = tokenizer.convert_tokens_to_ids('[EOS]')
    pad_id = tokenizer.pad_token_id  # Mặc định tokenizer BERT có token [PAD]
    
    caption_ids_ner = tokenizer(caption,  add_special_tokens=False)["input_ids"]
    sep_token = tokenizer(ent_separator,  add_special_tokens=False)["input_ids"]
    # NONAME token, bỏ [CLS], [SEP] nếu có, thường là do add_special_tokens=False thì không thêm [CLS],[SEP]
    noname_token = tokenizer("<NONAME>",  add_special_tokens=False)["input_ids"]

    ent_ids_flatten = []
    ent_ids_separate = []

    for ent in ent_list:
        ent_ids_original = tokenizer(" " + ent,  add_special_tokens=False)["input_ids"]
        ent_ids_original_space = tokenizer(f" {ent}",  add_special_tokens=False)["input_ids"]
        ent_ids_original_nospace = tokenizer(ent,  add_special_tokens=False)["input_ids"]

        # Thử tìm thực thể trong caption với ent_ids_original_space trước:
        idx = find_first_sublist(caption_ids_ner, ent_ids_original_space, start=0)

        if idx is not None:
            # Tìm thấy thực thể trong caption
            found_ent_ids = ent_ids_original_space
        else:
            # Không tìm thấy dạng có dấu cách, thử dạng không dấu cách
            idx = find_first_sublist(caption_ids_ner, ent_ids_original_nospace, start=0)
            if idx is not None:
                found_ent_ids = ent_ids_original_nospace
            else:
                # Không tìm thấy thực thể trong caption, ta sử dụng dạng không space làm fallback
                found_ent_ids = ent_ids_original_nospace

        # Thêm thực thể và sep token vào ent_ids_flatten
        ent_ids_flatten.extend(found_ent_ids)
        ent_ids_flatten.extend(sep_token)
        ent_ids_separate.append([bos_id] + found_ent_ids + [eos_id])

        if len(ent_ids_flatten) > max_length - 2:
            ent_ids_flatten = ent_ids_flatten[:max_length - 2]
            break

    if len(ent_ids_flatten) == 0:
        ent_ids_flatten.extend(noname_token)

    # Thêm [BOS], [EOS] vào ent_ids_flatten
    ent_ids_flatten = [bos_id] + ent_ids_flatten + [eos_id]

    # Pad đến max_length
    if len(ent_ids_flatten) < max_length:
        ent_ids_flatten += [pad_id] * (max_length - len(ent_ids_flatten))
    else:
        ent_ids_flatten = ent_ids_flatten[:max_length]

    ent_ids_separate.append([bos_id] + noname_token + [eos_id])

    # Pad cho ent_ids_separate
    ent_ids_separate = pad_list(ent_ids_separate, pad_id)

    return torch.LongTensor([ent_ids_flatten]), ent_ids_separate



def pad_list(list_of_name_ids, pad_token):
    max_len = max([len(seq) for seq in list_of_name_ids])
    padded_list = []
    for seq in list_of_name_ids:
        if len(seq) == max_len:
            padded_list.append(seq)
        else:
            padded_num = max_len - len(seq)
            seq.extend([pad_token] * padded_num)
            padded_list.append(seq)
    return padded_list


def concat_ner(ner_list, entity_token_start, entity_token_end):
    concat_ner_list = []
    if entity_token_start == "no" or entity_token_end=="no":
        for ner in ner_list:
            concat_ner_list.extend(ner)
    elif entity_token_start == "|":
        ner_nums = len(ner_list)
        for i, ner in enumerate(ner_list):
            if i < ner_nums-1:
                concat_ner_list.extend([ner + " " + entity_token_start])
            else:
                concat_ner_list.extend([ner])
    elif entity_token_start == "</s>":
        ner_nums = len(ner_list)
        for i, ner in enumerate(ner_list):
            if i < ner_nums-1:
                concat_ner_list.extend([ner + " " + entity_token_start + entity_token_end])
            else:
                concat_ner_list.extend([ner])
    else:
        for ner in ner_list:
            # print(entity_token_start + " " + ner + " " + entity_token_end)
            concat_ner_list.extend([entity_token_start + " " + ner + " " + entity_token_end])
    # print(concat_ner_list)
    return " ".join(concat_ner_list)

# Đối chiếu giữa ner và nerlist xem có trùng nhau j hay ko
def compare_ner(ner, ner_list):
    counter = 0
    for compare_ner in ner_list:
        if ner in compare_ner:
            counter += 1
        else:
            continue
    if counter > 0:
        return True
    else:
        return False


def load_image(path_or_url):
    try:
        # Lưu lại giá trị giới hạn ban đầu
        original_max_pixels = Image.MAX_IMAGE_PIXELS
        # Tạm thời vô hiệu hóa giới hạn pixel
        Image.MAX_IMAGE_PIXELS = None

        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/18.19041"
            }
            response = requests.get(path_or_url, headers=headers)
            response.raise_for_status()
            with Image.open(BytesIO(response.content)) as img:
                image = img.convert('RGB') if img.mode != 'RGB' else img.copy()
        else:
            with Image.open(path_or_url) as img:
                image = img.convert('RGB') if img.mode != 'RGB' else img.copy()

        # Kiểm tra số lượng pixel và giảm kích thước nếu cần
        num_pixels = image.width * image.height
        if num_pixels > MAX_PIXELS:
            print(f"Giảm kích thước ảnh từ {image.width}x{image.height} pixels.")
            scaling_factor = (MAX_PIXELS / num_pixels) ** 0.5
            new_width = max(1, int(image.width * scaling_factor))
            new_height = max(1, int(image.height * scaling_factor))
            image = image.resize((new_width, new_height), RESAMPLING)
            print(f"Kích thước ảnh sau khi giảm: {new_width}x{new_height} pixels.")

        # Khôi phục lại giá trị giới hạn ban đầu
        Image.MAX_IMAGE_PIXELS = original_max_pixels

        return image
    except Exception as e:
        print(f"Lỗi khi tải ảnh từ '{path_or_url}': {e}")
        return None

def pad_sequence_from_list(seq_list_list, special_token_id, bos_token_id, pad_token_id, eos_token_id, max_len):
    """
    Pad các chuỗi ID thực thể đến độ dài tối đa và số lượng thực thể trong batch.
    
    Tham số:
        seq_list_list (list of list of list of int): Danh sách các batch chứa danh sách các chuỗi ID.
        special_token_id (int): ID của token đặc biệt (ví dụ: <NONAME>).
        bos_token_id (int): ID của token bắt đầu (BOS).
        pad_token_id (int): ID của token pad.
        eos_token_id (int): ID của token kết thúc (EOS).
        max_len (int): Độ dài tối đa của mỗi chuỗi ID.
    
    Trả về:
        torch.Tensor: Tensor đã được pad với kích thước (batch_size, max_num_seq, max_len).
    """
    max_num_seq = max(len(seq_list) for seq_list in seq_list_list) if seq_list_list else 0
    padded_list_all = []
    
    for seq_list in seq_list_list:
        padded_list = []
        for seq in seq_list:
            pad_num = max_len - len(seq)
            if pad_num > 0:
                seq = seq + [pad_token_id] * pad_num
            else:
                seq = seq[:max_len]
            padded_list.append(seq)
        
        # Pad thêm các chuỗi đặc biệt nếu số lượng chuỗi trong seq_list < max_num_seq
        while len(padded_list) < max_num_seq:
            pad_batch_wise = [bos_token_id, special_token_id, eos_token_id] + [pad_token_id] * (max_len - 3)
            padded_list.append(pad_batch_wise)
        
        padded_list_all.append(torch.tensor(padded_list, dtype=torch.long))
    
    return torch.stack(padded_list_all)  # Kích thước (batch_size, max_num_seq, max_len)
def pad_sequence_from_list(seq_list_list, special_token_id, bos_token_id, pad_token_id, eos_token_id, max_len):
    """
    Pad các chuỗi ID thực thể đến độ dài tối đa và số lượng thực thể trong batch.
    
    Tham số:
        seq_list_list (list of list of list of int): Danh sách các batch chứa danh sách các chuỗi ID.
        special_token_id (int): ID của token đặc biệt (ví dụ: <NONAME>).
        bos_token_id (int): ID của token bắt đầu (BOS).
        pad_token_id (int): ID của token pad.
        eos_token_id (int): ID của token kết thúc (EOS).
        max_len (int): Độ dài tối đa của mỗi chuỗi ID.
    
    Trả về:
        torch.Tensor: Tensor đã được pad với kích thước (batch_size, max_num_seq, max_len).
    """
    max_num_seq = max(len(seq_list) for seq_list in seq_list_list) if seq_list_list else 0
    padded_list_all = []
    
    for seq_list in seq_list_list:
        padded_list = []
        for seq in seq_list:
            pad_num = max_len - len(seq)
            if pad_num > 0:
                seq = seq + [pad_token_id] * pad_num
            else:
                seq = seq[:max_len]
            padded_list.append(seq)
        
        # Pad thêm các chuỗi đặc biệt nếu số lượng chuỗi trong seq_list < max_num_seq
        while len(padded_list) < max_num_seq:
            pad_batch_wise = [bos_token_id, special_token_id, eos_token_id] + [pad_token_id] * (max_len - 3)
            padded_list.append(pad_batch_wise)
        
        padded_list_all.append(torch.tensor(padded_list, dtype=torch.long))
    
    return torch.stack(padded_list_all)  # Kích thước (batch_size, max_num_seq, max_len)

def get_max_len(items):
    """
    Trả về độ dài lớn nhất dựa trên kiểu của phần tử trong danh sách:
      - Nếu items là list of tensors (có thuộc tính 'size'), trả về max(tensor.size(1))
      - Nếu items là list of lists, trả về max(len(lst))
      - Nếu danh sách rỗng, trả về 0.
    """
    if not items:
        return 0

    first_item = items[0]
    # Kiểm tra nếu phần tử có thuộc tính 'size' (ví dụ như torch.Tensor)
    if hasattr(first_item, 'size') and callable(first_item.size):
        return max(tensor.size(1) for tensor in items)
    # Nếu phần tử là list, coi như danh sách các list
    elif isinstance(first_item, list):
        return max(len(lst) for lst in items)
    else:
        raise ValueError("Input phải là list of tensors hoặc list of lists.")

def pad_sequence(sequences, pad_value, max_len=None):
    """
    Pad các chuỗi đầu vào đến độ dài tối đa.
    
    Tham số:
        sequences (list of torch.Tensor): Danh sách các tensor cần được pad.
        pad_value (int hoặc float): Giá trị được sử dụng để pad.
        max_len (int, optional): Độ dài tối đa để pad. Nếu không được cung cấp, sử dụng độ dài lớn nhất trong sequences.
    
    Trả về:
        torch.Tensor: Tensor đã được pad với kích thước (batch_size, max_len).
    """
    if not sequences:
        return torch.empty((0,))
    
    if isinstance(sequences[0], list):
        if max_len is None:
            max_len = max(len(seq) for seq in sequences if isinstance(seq, list))
        padded_seqs = []
        for seq in sequences:

            if not isinstance(seq, list):
                print("Cảnh báo: seq không phải là list:", seq)
                continue
            if len(seq) < max_len:
                seq = seq + [pad_value] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            padded_seqs.append(torch.tensor(seq))
        return torch.stack(padded_seqs)
    else:
        # Nếu là tensor, xử lý như cũ
        if max_len is None:
            max_len = max(seq.size(1) for seq in sequences)
        padded_seqs = []
        for seq in sequences:
            if seq.size(1) < max_len:
                padding = (0, max_len - seq.size(1))
                padded_seq = torch.nn.functional.pad(seq, padding, value=pad_value)
            else:
                padded_seq = seq[:, :max_len]
            padded_seqs.append(padded_seq)
        return torch.cat(padded_seqs, dim=0)



# def pad_tensor_feat(feat_np_list, pad_feat_tensor):
#     """
#     Pad các tensor đặc trưng đến độ dài tối đa bằng cách thêm tensor pad.
    
#     Tham số:
#         feat_np_list (list of np.ndarray): Danh sách các đặc trưng dạng numpy array.
#         pad_feat_tensor (torch.Tensor): Tensor pad để thêm vào.
    
#     Trả về:
#         torch.Tensor: Tensor đã được pad với kích thước (batch_size, max_len, feature_dim).
#     """
def pad_tensor_feat(feat_np_list, pad_feat_tensor):
    """
    Pad các tensor đặc trưng đến độ dài tối đa bằng cách thêm tensor pad.
    
    Tham số:
        feat_np_list (list of np.ndarray): Danh sách các đặc trưng dạng numpy array.
        pad_feat_tensor (torch.Tensor): Tensor pad để thêm vào.
    
    Trả về:
        torch.Tensor: Tensor đã được pad với kích thước (batch_size, max_len, feature_dim).
    """
    tensor_list = [torch.from_numpy(feat) for feat in feat_np_list]
    lengths = [tensor.size(0) for tensor in tensor_list]
    max_len = max(lengths) if lengths else 0
    
    padded_tensors = []
    for tensor in tensor_list:
        # Nếu tensor rỗng, chuyển về dạng có shape (0, feature_dim)
        if tensor.numel() == 0:
            tensor = tensor.view(0, pad_feat_tensor.size(1))
        
        if tensor.size(0) < max_len:
            pad_length = max_len - tensor.size(0)
            pad_tensor_expanded = pad_feat_tensor.repeat(pad_length, 1)
            padded_tensor = torch.cat([tensor, pad_tensor_expanded], dim=0)
        else:
            padded_tensor = tensor[:max_len]
        padded_tensors.append(padded_tensor)
    
    if padded_tensors:
        return torch.stack(padded_tensors, dim=0)
    else:
        return torch.empty((0, pad_feat_tensor.size(1)))


import os
import json
import re
from collections import defaultdict
from transformers import AutoTokenizer
import stanza

def is_abbreviation(entity):
    """
    Kiểm tra xem một chuỗi có phải viết tắt theo dạng A.B. hay không.
    Pattern: A. B. (có thể thêm tên tiếp theo sau)
    """
    pattern = r'\b([A-Z]\.)+([A-Z][a-z]*)?\b'
    return re.match(pattern, entity) is not None

def extract_entities(text, nlp):
    """
    Sử dụng Stanza để trích xuất các thực thể từ văn bản.
    Đồng thời áp dụng kiểm tra chữ viết tắt để chỉ thêm vào kết quả 
    nếu là abbreviation hợp lệ hoặc là thực thể bình thường không chứa '.'.
    
    Trả về một dictionary với các loại thực thể (PERSON, ORGANIZATION, v.v.),
    mỗi loại là một danh sách các thực thể duy nhất.
    """
    entities = defaultdict(set)  
    if not text.strip():
        return {}

    try:
        doc = nlp(text)
    except Exception as e:
        print(f"Lỗi khi xử lý text: {e}")
        return {}

    # Ánh xạ nhãn từ Stanza sang nhãn chuẩn nếu cần
    label_mapping = {
        "PER": "PERSON",
        "ORG": "ORGANIZATION",
        "LOC": "LOCATION",
        "GPE": "GPE",
        "NORP": "NORP",
        "MISC": "MISC"
    }

    # Duyệt qua từng câu và từng thực thể trong câu
    for sentence in doc.sentences:
        for ent in sentence.ents:
            mapped_type = label_mapping.get(ent.type, ent.type)
            ent_text = ent.text.strip()

            if mapped_type and ent_text:
                # Nếu thực thể có chứa dấu chấm, chỉ thêm nếu đúng định dạng abbreviation
                if '.' in ent_text:
                    if is_abbreviation(ent_text):
                        entities[mapped_type].add(ent_text)
                else:
                    entities[mapped_type].add(ent_text)

    # Chuyển set về list để xuất kết quả
    return {key: list(value) for key, value in entities.items()}

def create_article_ner_mask_dict(article_text, hash_id, article_ner_mask_dir, person_token_id, max_len,tokenizer,nlp):
    """
    Tạo đối tượng article_ner_mask_dict với 2 thành phần:
      - input_ids: danh sách token id của bài báo (sử dụng tokenizer của Bartpho)
      - ner_mask: mảng có độ dài bằng số token, đánh dấu vị trí của thực thể PERSON hợp lệ (theo xử lý của extract_entities)
    """

    inputs = tokenizer(article_text, truncation=True, max_length=max_len, return_offsets_mapping=True)
    input_ids = inputs["input_ids"][0].tolist()
    offset_mapping = inputs["offset_mapping"][0]
    # 3. Sử dụng extract_entities để lấy danh sách các thực thể PERSON hợp lệ
    valid_entities = extract_entities(article_text, nlp)
    valid_person_texts = set(valid_entities.get("PERSON", []))
    
    # 4. Lấy thông tin vị trí của các thực thể PERSON từ kết quả NER (chỉ chọn các thực thể hợp lệ)
    doc = nlp(article_text)
    person_entities = [
        ent for sentence in doc.sentences 
        for ent in sentence.ents 
        if ent.type == "PER" and ent.text.strip() in valid_person_texts
    ]
    ner_mask = [0] * len(offset_mapping)
    for ent in person_entities:
        ent_start = ent.start_char
        ent_end = ent.end_char
        # Duyệt qua từng token, nếu khoảng của token giao với khoảng của thực thể thì gán person_token_id
        for i, (start, end) in enumerate(offset_mapping):
            if start < ent_end and end > ent_start:
                ner_mask[i] = person_token_id
                
    # 6. Tạo dictionary và lưu ra file JSON
    article_ner_mask_dict = {
        "input_ids": input_ids,
        "ner_mask": ner_mask
    }
    os.makedirs(article_ner_mask_dir, exist_ok=True)
    file_path = os.path.join(article_ner_mask_dir, f"{hash_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(article_ner_mask_dict, f, ensure_ascii=False, indent=4)
        
    # return article_ner_mask_dict


def collate_fn_vndataset_entity_type(batch):
    """
    Hàm collate để xử lý một batch dữ liệu từ Dataset.

    Tham số:
        batch (list of dict): Danh sách các mẫu dữ liệu trong batch.

    Trả về:
        dict: Dictionary chứa các trường dữ liệu đã được pad và batch hóa.
    """
    # Khởi tạo các danh sách để chứa các trường dữ liệu
    article_list = []
    article_id_list = []
    article_ner_mask_id_list = []

    caption_list = []
    caption_id_list = []
    caption_id_clip_list = []
    names_art_list = []
    names_art_ids_list = []
    org_norp_gpe_loc_art_list = []
    org_norp_gpe_loc_art_ids_list = []
    names_list = []
    names_ids_list = []
    org_norp_gpe_loc_list = []
    org_norp_gpe_loc_ids_list = []

    names_ids_flatten_list = []
    org_norp_gpe_loc_ids_flatten_list = []

    all_gt_ner_list = []
    all_gt_ner_ids_list = []
    face_emb_list = []
    obj_emb_list = []
    img_tensor_list = []

    person_id_positions_list = []
    person_id_positions_cap_list = []

    for sample in batch:
        if sample is None:
            continue  # Bỏ qua các mẫu None (nếu có)
        
        (
            article, article_ids, article_ner_mask_ids, caption, caption_ids, caption_ids_clip, 
            names_art, org_norp_gpe_loc_art, names_art_ids, org_norp_gpe_loc_art_ids, 
            names, org_norp_gpe_loc, names_ids, org_norp_gpe_loc_ids, 
            all_gt_ner_ids, all_gt_ner, face_emb, obj_emb, img_tensor, 
            person_id_positions, person_id_positions_cap
        ) = (
            sample["article"], sample["article_ids"], sample["article_ner_mask_ids"], 
            sample["caption"], sample["caption_ids"], sample["caption_ids_clip"], 
            sample["names_art"], sample["org_norp_gpe_loc_art"], sample["names_art_ids"], 
            sample["org_norp_gpe_loc_art_ids"], sample["names"], sample["org_norp_gpe_loc"], 
            sample["names_ids"], sample["org_norp_gpe_loc_ids"], 
            sample["all_gt_ner_ids"], sample["all_gt_ner"], sample["face_emb"], 
            sample["obj_emb"], sample["img_tensor"], 
            sample["person_id_positions"], sample["person_id_positions_cap"]
        )

        names_ids_flatten, org_norp_gpe_loc_ids_flatten = sample["names_ids_flatten"], sample["org_norp_gpe_loc_ids_flatten"]

        names_ids_flatten_list.append(names_ids_flatten)
        org_norp_gpe_loc_ids_flatten_list.append(org_norp_gpe_loc_ids_flatten)

        article_list.append(article)
        article_id_list.append(article_ids)
        article_ner_mask_id_list.append(article_ner_mask_ids)
        # print('article_ner_mask_ids:', article_ner_mask_ids)
        caption_list.append(caption)
        caption_id_list.append(caption_ids)
        if caption_ids_clip is not None:
            caption_id_clip_list.append(caption_ids_clip)

        names_art_list.append(names_art)
        names_art_ids_list.append(names_art_ids)
        org_norp_gpe_loc_art_list.append(org_norp_gpe_loc_art)
        org_norp_gpe_loc_art_ids_list.append(org_norp_gpe_loc_art_ids)

        names_list.append(names)
        names_ids_list.append(names_ids)
        org_norp_gpe_loc_list.append(org_norp_gpe_loc)
        org_norp_gpe_loc_ids_list.append(org_norp_gpe_loc_ids)

        all_gt_ner_list.append(all_gt_ner)
        all_gt_ner_ids_list.append(all_gt_ner_ids)
        
        face_emb_list.append(face_emb)
        obj_emb_list.append(obj_emb)
        
        img_tensor_list.append(img_tensor)
        
        person_id_positions_list.append(person_id_positions)
        person_id_positions_cap_list.append(person_id_positions_cap)
    
    # Xử lý pad cho các trường dữ liệu
    # max_len_input = get_max_len([article_id_list, article_ner_mask_id_list])
    # article_ids_batch = pad_sequence(article_id_list, pad_value=1, max_len=max_len_input)
    max_len_input = get_max_len(article_id_list)
    article_ids_batch = pad_sequence(article_id_list, pad_value=1, max_len=max_len_input)
    # article_ids_batch = torch.cat([article_ids.squeeze(0) for article_ids in article_id_list], dim=0)
    article_ner_mask_ids_batch = pad_sequence(article_ner_mask_id_list, pad_value=1, max_len=max_len_input)
    caption_ids_batch = pad_sequence(caption_id_list, pad_value=1)
    if len(caption_id_clip_list) > 0:
        caption_ids_clip_batch = pad_sequence(caption_id_clip_list, pad_value=0)
    else:
        caption_ids_clip_batch = torch.empty((1,1))
    
    max_len_art_ids = get_max_len([names_art_ids_list, org_norp_gpe_loc_art_ids_list])

    max_len_name_ids = get_max_len([lst for lst in names_ids_list])
    max_len_org_norp_gpe_loc_ids = get_max_len([lst for lst in org_norp_gpe_loc_ids_list])

    names_art_ids_batch = pad_sequence(names_art_ids_list, pad_value=1, max_len=max_len_art_ids)
    org_norp_gpe_loc_art_ids_batch = pad_sequence(org_norp_gpe_loc_art_ids_list, pad_value=1, max_len=max_len_art_ids)

    names_ids_batch = pad_sequence_from_list(
        names_ids_list, 
        special_token_id=bart_tokenizer.convert_tokens_to_ids('<NONAME>'), 
        bos_token_id=0, 
        pad_token_id=1, 
        eos_token_id=2,  
        max_len=max_len_name_ids
    )

    org_norp_gpe_loc_ids_batch = pad_sequence_from_list(
        org_norp_gpe_loc_ids_list, 
        special_token_id=bart_tokenizer.convert_tokens_to_ids('<NONAME>'), 
        bos_token_id=0, 
        pad_token_id=1, 
        eos_token_id=2, 
        max_len=max_len_org_norp_gpe_loc_ids
    )

    all_gt_ner_ids_batch = pad_sequence(all_gt_ner_ids_list, pad_value=1)

    max_len_ids_flatten = get_max_len([names_ids_flatten_list, org_norp_gpe_loc_ids_flatten_list])
    names_ids_flatten_batch = pad_sequence(names_ids_flatten_list, pad_value=1, max_len=max_len_ids_flatten)
    org_norp_gpe_loc_ids_flatten_batch = pad_sequence(org_norp_gpe_loc_ids_flatten_list, pad_value=1, max_len=max_len_ids_flatten)
    
    img_batch = torch.stack(img_tensor_list, dim=0).squeeze(1)  # Kích thước (batch_size, channels, height, width)

    face_pad = torch.ones((1, 512))
    obj_pad = torch.ones((1, 1000))
    face_batch = pad_tensor_feat(face_emb_list, pad_feat_tensor=face_pad)
    obj_batch = pad_tensor_feat(obj_emb_list, pad_feat_tensor=obj_pad)
    # print(f"Article ID list: {[x.shape for x in article_id_list]}, Max len input: {max_len_input}")
    # print(f"Article IDs batch shape: {article_ids_batch.shape}")    

    return {
        "article_ids": article_ids_batch,
        "article_ner_mask_ids": article_ner_mask_ids_batch,
        "caption_ids": caption_ids_batch,
        "caption_ids_clip": caption_ids_clip_batch if caption_ids_clip_batch.numel() > 0 else None,
        "names_art_ids": names_art_ids_batch,
        "org_norp_gpe_loc_art_ids": org_norp_gpe_loc_art_ids_batch,
        "names_ids": names_ids_batch,
        "org_norp_gpe_loc_ids": org_norp_gpe_loc_ids_batch,
        "all_gt_ner_ids": all_gt_ner_ids_batch,
        "names_ids_flatten": names_ids_flatten_batch,
        "org_norp_gpe_loc_ids_flatten": org_norp_gpe_loc_ids_flatten_batch,
        # Các trường khác giữ nguyên
        "article": article_list,
        "caption": caption_list,
        "names_art": names_art_list,
        "org_norp_gpe_loc_art": org_norp_gpe_loc_art_list,
        "names": names_list,
        "org_norp_gpe_loc": org_norp_gpe_loc_list,
        "all_gt_ner": all_gt_ner_list,
        "face_emb": face_batch.float(),
        "obj_emb": obj_batch.float(),
        "img_tensor": img_batch,
        "person_id_positions": person_id_positions_list,
        "person_id_positions_cap": person_id_positions_cap_list,
    }


class VNDataset(Dataset):
    def __init__(self, data_dict, data_base_dir, tokenizer, use_clip_tokenizer=False, entity_token_start="no", entity_token_end="no", transform=None, max_article_len=512, max_ner_type_len=80, max_ner_type_len_gt=20, retrieved_sent=True, person_token_id=person_token_id):
        super().__init__()
        self.data_dict = copy.deepcopy(data_dict) #
        self.face_dir = os.path.join(data_base_dir, "faces") #
        self.obj_dir = os.path.join(data_base_dir, "objects") #
        # self.article_dir = os.path.join(data_base_dir, "articles_full")
        self.article_ner_mask_dir = os.path.join(data_base_dir, "article_all_ent_by_count_dir") #
        self.img_dir = os.path.join('/data/npl/ICEK/Wikipedia/images', "images") #
        self.tokenizer = tokenizer
        self.use_clip_tokenizer = use_clip_tokenizer
        self.max_len = max_article_len
        self.transform = transform
        self.entity_token_start = entity_token_start
        self.entity_token_end = entity_token_end
        self.hash_ids = [*data_dict.keys()]
        self.max_ner_type_len = max_ner_type_len
        self.max_ner_type_len_gt = max_ner_type_len_gt

        self.retrieved_sent=retrieved_sent

        self.person_token_id = person_token_id
    
    def __getitem__(self, index):
        hash_id = self.hash_ids[index]
        # lấy ảnh

        img_url = self.data_dict[hash_id].get('image_url','')
        image_path = self.data_dict[hash_id].get('image_path','')

        img = load_image(image_path)
        if img is None:
            print(f'Lỗi r bạn ei, tải ảnh từ {img_url}')
            img = load_image(img_url)

        # img = Image.open(os.path.join(self.img_dir, f"{hash_id}.jpg")).convert('RGB')

        # lấy face embedding và names nếu face_embedding không rỗng, nếu rỗng trả về rỗng hết
        if self.data_dict[hash_id]["face_emb_dir"] != []:
            face_emb = np.load(os.path.join(self.face_dir, f"{hash_id}.npy"))
            # lấy tên
            names = self.data_dict[hash_id]["names"]
        else:
            face_emb = np.array([[]])
            names = []
        # lấy obj_emb_dir nếu không rỗng
        if self.data_dict[hash_id]["obj_emb_dir"] != []:
            obj_emb = np.load(os.path.join(self.obj_dir, f"{hash_id}.npy"))
        else:
            obj_emb = np.array([[]])

        # lấy sents_byclip
        if self.retrieved_sent:
            article = self.data_dict[hash_id]["sents_byclip"]
        # print('article: ',article[:50])
        caption = self.data_dict[hash_id]["caption"]
        names = self.data_dict[hash_id]["names"]
        org_norp = self.data_dict[hash_id]["org_norp"]
        gpe_loc = self.data_dict[hash_id]["gpe_loc"]
        names_art = self.data_dict[hash_id]["names_art"]
        org_norp_art = self.data_dict[hash_id]["org_norp_art"]
        gpe_loc_art = self.data_dict[hash_id]["gpe_loc_art"]
        # article = preprocess_article(article)

        # Khử sự trùng lặp của tên trong article
        new_names_art_list = []
        for i in range(len(names_art)):
            if compare_ner(names_art[i], names_art[:i] + names_art[i+1:]):
                continue
            else:
                new_names_art_list.append(names_art[i])


        # Khử sự trùng lặp của org_norp trong article
        new_org_norp_art_list = []
        for i in range(len(org_norp_art)):
            if compare_ner(org_norp_art[i], org_norp_art[:i] + org_norp_art[i+1:]):
                continue
            else:
                new_org_norp_art_list.append(org_norp_art[i])

        # Khử sự trùng lặp của gpe_loc trong article
        new_gpe_loc_art_list = []
        for i in range(len(gpe_loc_art)):
            if compare_ner(gpe_loc_art[i], gpe_loc_art[:i] + gpe_loc_art[i+1:]):
                continue
            else:
                new_gpe_loc_art_list.append(gpe_loc_art[i])
        
        # danh sách chứa các org_norp_gpe_loc mới
        new_org_norp_gpe_loc_art_list = [*new_org_norp_art_list, *new_gpe_loc_art_list]
        org_norp_gpe_loc = [*org_norp, *gpe_loc]

        # lấy tất cả các ner 
        all_gt_ner = names + org_norp + gpe_loc
        # print(all_gt_ner)

        # lúc này có 1 danh sách các ner r thì tiến hành kết giữa các ner theo token được nhập vào
        concat_gt_ner = concat_ner(all_gt_ner, self.entity_token_start, self.entity_token_end)
        # print(concat_gt_ner)

        # Tokenize các concat_gt_ner, article
        gt_ner_ids = self.tokenizer(concat_gt_ner,  return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_ner_type_len_gt)["input_ids"]
        article_ids = self.tokenizer(article,  return_tensors="pt", truncation=True, padding=True,max_length=self.max_len)["input_ids"]
        # print('article_ids: ',article_ids)
        # print('gt_ner_ids: ',gt_ner_ids)
        # article_ner_mask_ids = torch.randn((1,512))


        with open(os.path.join(self.article_ner_mask_dir, f"{hash_id}.json")) as f:
            article_ner_mask_dict = json.load(f)
        
        article_ner_mask_ids = article_ner_mask_dict["input_ids"][0]
        # lấy vị trí của person trong article và caption
        person_id_positions = get_person_ids_position(article_ner_mask_dict["input_ids"][0], person_token_id=self.person_token_id, article_max_length=self.max_len)
        person_id_positions_cap = self.data_dict[hash_id]["name_pos_cap"]
        
        caption_ids = self.tokenizer(caption,  return_tensors="pt", truncation=True,  max_length=100)["input_ids"]
        # print("caption_ids: ",caption_ids)
        if self.use_clip_tokenizer:
            encoded_input = clip_tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=77,      
            return_tensors='pt' 
            )
            caption_ids_clip = encoded_input['input_ids']
        else:
            caption_ids_clip = None
        
        # print("caption_ids_clip: ",caption_ids_clip)
        names_art_ids, _ = make_new_entity_ids(article, new_names_art_list, self.tokenizer, ent_separator=self.entity_token_start, max_length=self.max_ner_type_len)
        names_ids_flatten, names_ids = make_new_entity_ids(caption, names, self.tokenizer, ent_separator=self.entity_token_start, max_length=self.max_ner_type_len_gt)

        org_norp_gpe_loc_art_ids, _ = make_new_entity_ids(article, new_org_norp_gpe_loc_art_list, self.tokenizer, ent_separator=self.entity_token_start, max_length=self.max_ner_type_len)
        org_norp_gpe_loc_ids_flatten, org_norp_gpe_loc_ids = make_new_entity_ids(caption, org_norp_gpe_loc, self.tokenizer, ent_separator=self.entity_token_start, max_length=self.max_ner_type_len_gt)

        img_tensor = self.transform(img).unsqueeze(0)

        return {"article": article, "article_ids":article_ids, "article_ner_mask_ids":article_ner_mask_ids, "caption": caption, "caption_ids": caption_ids, "caption_ids_clip": caption_ids_clip, "names_art": new_names_art_list, "org_norp_gpe_loc_art": new_org_norp_gpe_loc_art_list, "names_art_ids": names_art_ids, "org_norp_gpe_loc_art_ids": org_norp_gpe_loc_art_ids, "names": names, "org_norp_gpe_loc": org_norp_gpe_loc, "names_ids": names_ids, "org_norp_gpe_loc_ids": org_norp_gpe_loc_ids, "all_gt_ner":all_gt_ner, "all_gt_ner_ids":gt_ner_ids, "face_emb":face_emb, "obj_emb":obj_emb, "img_tensor":img_tensor, "names_ids_flatten":names_ids_flatten, "org_norp_gpe_loc_ids_flatten":org_norp_gpe_loc_ids_flatten, "person_id_positions":person_id_positions, "person_id_positions_cap":person_id_positions_cap}
    def __len__(self):
        return len(self.data_dict)


if __name__ == "__main__":
    
    # from torchvision import transforms
    # with open('/data/npl/ICEK/VACNIC/data/train/test.json', 'r', encoding='utf-8') as f:
    #     data_dict = json.load(f)

    # # Định nghĩa các biến cấu hình
    # data_base_dir = '/data/npl/ICEK/VACNIC/data/train'
    # use_clip_tokenizer = True  # Bật để sử dụng CLIP
    # entity_token_start = "<ent_start>"
    # entity_token_end = "<ent_end>"
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])
    # max_article_len = 512
    # max_ner_type_len = 80
    # max_ner_type_len_gt = 20
    # retrieved_sent = True
    # person_token_id = person_token_id  
    # # Khởi tạo Dataset với CLIP
    # dataset = VNDataset(
    #     data_dict=data_dict,
    #     data_base_dir=data_base_dir,
    #     tokenizer=bart_tokenizer,
    #     # clip_model=clip_model,
    #     # clip_preprocess=clip_preprocess,
    #     use_clip_tokenizer=use_clip_tokenizer,
    #     entity_token_start=entity_token_start,
    #     entity_token_end=entity_token_end,
    #     transform=transform,
    #     max_article_len=max_article_len,
    #     max_ner_type_len=max_ner_type_len,
    #     max_ner_type_len_gt=max_ner_type_len_gt,
    #     retrieved_sent=retrieved_sent,
    #     person_token_id=person_token_id
    # )

    # # Khởi tạo DataLoader
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn_vndataset_entity_type)

    # # Kiểm tra một batch
    # for batch in dataloader:
    #     # Xử lý batch
    #     # print('OK!')
    #     print('article_ids: ' ,batch['article_ids'].shape)
    #     print('captions_ids: ' , batch['caption_ids'].shape)
    #     print('caption_ids_clip: ', batch['caption_ids_clip'].shape if batch['caption_ids_clip'] is not None else "No CLIP features")
    #     print("article_ner_mask_ids: ",batch["article_ner_mask_ids"].shape)
    #     print("img_tensor: ",batch["img_tensor"].shape )
    #     print("names_ids_flatten: ",batch["names_ids_flatten"].shape)
    #     print("names_art_ids: ",batch["names_art_ids"].shape)
    #     print('names_ids: ',batch["names_ids"].shape)
    #     print('face_emb: ',batch["face_emb"].shape)
        

        
    #     break

    # # Ví dụ văn bản
    # line1 = "Cuộc tấn_công đường không do Mỹ tiến_hành mùa Xuân và Hè năm 1972 , đến mùa thu trên chiến_trường chỉ có 71 máy_bay_tiêm_kích của Việt_Nam do đồng chí Nguyễn Tất Thành ( 40 chiếc MiG-17 và MiG-19 , 31 chiếc MiG-21 ) chống lại 360 máy_bay_tiêm_kích chiến_thuật của Không_quân Mỹ và 96 máy_bay_tiêm_kích của Hải_quân."
    # line1 = line1.replace('_',' ')
    # line1 = line1.replace("//n//n",'')
    # paragraphs = line1.split(',')


    # all_entities = []
    # for para in paragraphs:
    #     if para.strip() == "":
    #         continue
    #     ner_results = ner_pipeline(para)
    #     entities = get_entities(ner_results)
    #     all_entities.extend(entities)

    # # Thực hiện NER
    # ent_list = [ent["text"] for ent in all_entities]
    # ent_type_list = [ent["label"] for ent in all_entities]

    # print("ent_list:", ent_list)
    # print("ent_type_list:", ent_type_list)
    
    # # Tạo dictionary NER theo loại
    # entities_type, start_pos_list, _, _, _ = make_ner_dict_by_type(ent_list, ent_type_list)
    # print("entities_type:", entities_type)
    
    # # Thay thế các thực thể trong văn bản bằng nhãn NER
    # new_article = replace_entities_with_labels(line1, entities_type, ent_list)
    # print("new_article:", new_article)
    
    # # Tokenize với BERT tokenizer
    # bert_inputs = bart_tokenizer(
    #     new_article,
    #     add_special_tokens=True,      # Thêm các token đặc biệt như [CLS], [SEP]
    #     return_tensors="pt",          # Trả về dạng tensor PyTorch
    #     padding=True,                # Không padding vì chỉ có một câu
    #     truncation=True,              # Cắt bớt nếu quá dài (có thể điều chỉnh max_length nếu cần)
    # )
    
    # input_ids = bert_inputs["input_ids"].to(device).tolist()[0]           # Tensor chứa các token IDs
    # attention_mask = bert_inputs["attention_mask"].to(device)  # Tensor chứa attention mask
    
    # print("input_ids:", input_ids)
    # # print("attention_mask:", attention_mask)
    # person_position = get_person_ids_position(article_ids_replaced=input_ids,person_token_id=62000,article_max_length= 512)
    # print("person_position: ",person_position)


    with open('/data/npl/ICEK/VACNIC/data/train/test3.json','r',encoding='utf-8') as f:
        data = json.load(f)

    save_full_processed_articles_all_ent_by_count(data, bart_tokenizer, nlp)



    