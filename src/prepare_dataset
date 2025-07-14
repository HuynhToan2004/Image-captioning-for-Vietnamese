import os
from re import I
import numpy as np
from PIL import Image
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import json
from torchvision import transforms
import copy
import stanza
import re
MAX_PIXELS = 89478485
RESAMPLING = Image.LANCZOS  

from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True   # tránh lỗi ảnh hỏng

def debug_loader(loader, n_batches=3):
    for i,(batch) in enumerate(loader):
        print(f"\nBatch {i}")
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k:16s}", v.shape,
                      "finite=", torch.isfinite(v).all().item(),
                      "min=", v.min().item(), "max=", v.max().item())
        if i+1 == n_batches: break


def safe_open_resize(path, max_px=16_000_000):              # ≈4K × 4K  
    """
    - path    : đường dẫn file ảnh
    - max_px  : giới hạn W×H tối đa. Vượt ngưỡng → resize.
    Trả về:   đối tượng PIL.Image  (RGB)
              hoặc None nếu không đọc được.
    """
    try:
        img = Image.open(path).convert("RGB")
        if img.size[0] * img.size[1] > max_px:
            ratio = (max_px / (img.size[0] * img.size[1])) ** 0.5
            new_w = int(img.size[0] * ratio)
            new_h = int(img.size[1] * ratio)
            img   = img.resize((new_w, new_h), Image.BICUBIC)
        return img
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError):
        return None              # caller tự quyết định bỏ qua hay thay thế
# 1. Khởi tạo NER
# nlp = stanza.Pipeline(lang='vi', processors='tokenize,ner')


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


def get_max_len_list(seq_list_of_list):
    # input list of seq_list, output max len
    max_len_list = []
    for seq_list in seq_list_of_list:
        # print(seq_list)
        max_len_list.extend([len(seq) for seq in seq_list])
    return max(max_len_list)


def pad_sequence_from_list(seq_list_list, special_token_id, bos_token_id, pad_token_id, eos_token_id, max_len):
    # special_token_id: <NONAME>
    max_num_seq = max([len(seq_list) for seq_list in seq_list_list])
    padded_list_all = []
    for seq_list in seq_list_list:
        padded_list = []
        for seq in seq_list:
            # pad in each sample
            pad_num = max_len - len(seq)
            seq = seq + [pad_token_id] * pad_num
            # print(seq, pad_num)
            if max_num_seq == 1:
                padded_list.append([seq])
            else:
                padded_list.append(seq)
        if len(seq_list) < max_num_seq:
            # pad in each batch
            pad_batch_wise = [bos_token_id] + [special_token_id] + [eos_token_id] + [pad_token_id] * (max_len-3)
            for i in range(max_num_seq - len(seq_list)):
                padded_list.append(pad_batch_wise)
        padded_list_all.append(torch.tensor(padded_list, dtype=torch.long))
    return torch.stack(padded_list_all)


def get_max_len(seq_tensor_list_of_list):
    # input list of seq_tensor_list, output max len
    max_len_list = []
    for seq_tensor_list in seq_tensor_list_of_list:
        max_len_list.append(max([seq.size(1) for seq in seq_tensor_list]))
    return max(max_len_list)


# def pad_sequence(seq_tensor_list, pad_token_id, max_len=None):
#     if max_len is None:
#         max_len = max([seq.squeeze(0).size(1) for seq in seq_tensor_list])

#     pad_token = torch.tensor([pad_token_id])
#     padded = []
#     for seq in seq_tensor_list:
#         seq = seq.squeeze(0) if seq.dim() == 3 and seq.size(0) == 1 else seq  # 👈
#         pad_num = max_len - seq.size(1)
#         if pad_num > 0:
#             pad = torch.full((1, pad_num), pad_token_id, dtype=torch.long)
#             padded.append(torch.cat((seq, pad), dim=1))
#         else:
#             padded.append(seq)
#     return torch.stack(padded)

def pad_sequence(seq_tensor_list, pad_token_id, max_len=None):
    """
    Chuẩn hoá mọi tensor về (1, max_len) rồi stack → (B, 1, max_len).
    Hỗ trợ input (L,), (1,L), (1,1,L) v.v.
    """
    # 1. đưa tất cả tensor về 1-chiều
    flat = [seq.view(-1) for seq in seq_tensor_list]

    # 2. xác định max_len đúng
    true_max = max(t.size(0) for t in flat)
    if max_len is None or true_max > max_len:
        max_len = true_max

    pad_tok = torch.tensor(pad_token_id, dtype=torch.long)

    padded = []
    for t in flat:
        if t.size(0) < max_len:                  # cần pad
            pad = pad_tok.repeat(max_len - t.size(0))
            t   = torch.cat((t, pad), dim=0)
        padded.append(t.unsqueeze(0))            # -> (1,max_len)

    return torch.stack(padded)                   # (B,1,max_len)


def pad_sequence_lm(seq_tensor_list, sos_token_id, eos_token_id, pad_token_id):
    max_len = max([seq.size(1) for seq in seq_tensor_list])

    pad_token = torch.tensor([pad_token_id])
    padded_list = []
    sos_token = torch.tensor([sos_token_id], dtype=torch.long).unsqueeze(0)
    eos_token = torch.tensor([eos_token_id], dtype=torch.long).unsqueeze(0)
    for seq in seq_tensor_list:
        # print(seq.size())
        pad_num = max_len + 2 - seq.size(1)
        if pad_num > 0:
            to_be_padded = torch.tensor([pad_token]*pad_num, dtype=torch.long).unsqueeze(0)
            # padded_seq = torch.cat((seq, to_be_padded), dim=0)
            padded_list.append(torch.cat((sos_token, seq, eos_token, to_be_padded), dim=1))
        else:
            padded_list.append(torch.cat((sos_token, seq, eos_token), dim=1))
    return torch.stack(padded_list)



def pad_article(seq_tensor_list, pad_token_id):
    """2D padding to pad list of sentences"""
    max_len = max([seq.size(1) for seq in seq_tensor_list])

    pad_token = torch.tensor([pad_token_id])
    padded_list = []
    for seq in seq_tensor_list:
        # print(seq.size())
        pad_num = max_len - seq.size(1)
        if pad_num > 0:
            to_be_padded = seq.size(0)*[torch.tensor([pad_token]*pad_num, dtype=torch.long)]
            to_be_padded = torch.stack(to_be_padded, dim=0)
            # padded_seq = torch.cat((seq, to_be_padded), dim=0)
            # print(to_be_padded.size(), seq.size())
            padded_list.append(torch.cat((seq, to_be_padded), dim=1))
        else:
            padded_list.append(seq)
    
    max_len_article = max([seq.size(0) for seq in seq_tensor_list])
    padded_list_out = []
    for seq in padded_list:
        # print(seq.size())
        pad_num = max_len_article - seq.size(0)
        if pad_num > 0:
            to_be_padded = pad_num * [torch.tensor([pad_token]*max_len, dtype=torch.long)]
            to_be_padded = torch.stack(to_be_padded, dim=0)
            # padded_seq = torch.cat((seq, to_be_padded), dim=0)
            padded_list_out.append(torch.cat((seq, to_be_padded), dim=0))
        else:
            padded_list_out.append(seq)
    return torch.stack(padded_list_out)


# def pad_tensor_feat(feat_np_list, pad_feat_tensor):
#     # tensor_list = [torch.from_numpy(feat) for feat in feat_np_list]
#     len_list = []
#     for feat in feat_np_list:
#         if feat.shape[1] == 0:
#             len_list.append(0)
#         else:
#             len_list.append(feat.shape[0])
#     max_len = max(len_list)
#     # print(max_len)
#     padded_list = []
#     for i, feat in enumerate(feat_np_list):
#         pad_num = max_len - len_list[i]
#         if pad_num > 0:
#             to_be_padded = pad_num* [pad_feat_tensor]
#             to_be_padded = torch.stack(to_be_padded, dim=0)
#             to_be_padded = to_be_padded.squeeze(1)
#             # print(to_be_padded.size())
#             if feat.shape[1] != 0:
#                 # padded_list.append(torch.stack((torch.from_numpy(feat), to_be_padded), dim=0).squeeze(1))
#                 padded_list.append(torch.cat((torch.from_numpy(feat), to_be_padded), dim=0).squeeze(1))
#             else:
#                 padded_list.append(to_be_padded)
#         elif max_len == 0:
#             to_be_padded = 1* [pad_feat_tensor]
#             to_be_padded = torch.stack(to_be_padded, dim=0)
#             to_be_padded = to_be_padded.squeeze(1)
#             padded_list.append(to_be_padded)
#         else:
#             # print(torch.from_numpy(feat).size())
#             padded_list.append(torch.from_numpy(feat))
#     return  torch.stack(padded_list)


def make_new_entity_ids(caption, ent_list, tokenizer, ent_separator="<ent>", max_length=80):
    caption_ids_ner = tokenizer(caption, add_special_tokens=False)["input_ids"]
    # print(caption_ids_ner)

    sep_token = tokenizer(ent_separator, add_special_tokens=False)["input_ids"]
    # print(sep_token)

    noname_token = tokenizer("<NONAME>")["input_ids"][1:-1]

    ent_ids_flatten = []
    ent_ids_separate = []

    for ent in ent_list:
        # in case entities were in the middle of the sentence
        ent_ids_original = tokenizer(f" {ent}")["input_ids"][1:-1]
        # print(ent_ids_original)
        # if ent_ids_original in article_ids_ner:
        idx = find_first_sublist(caption_ids_ner, ent_ids_original, start=0)
        if idx is not None:
            # print(ent_ids_original)
            ent_ids_flatten.extend(ent_ids_original)
            ent_ids_flatten.extend(sep_token)
            ent_ids_separate.append([tokenizer.bos_token_id] + ent_ids_original + [tokenizer.eos_token_id])
            if len(ent_ids_flatten) > max_length-2:
                ent_ids_flatten = ent_ids_flatten[:max_length-2]
                break
            else:
                continue
        else:
            # print(ent_ids_original)
            # in case entities were at the start of the sentence
            ent_ids_original_start = tokenizer(f"{ent}")["input_ids"][1:-1]
            # print(f"start:{ent_ids_original_start}")
            ent_ids_flatten.extend(ent_ids_original_start)
            ent_ids_flatten.extend(sep_token)
            ent_ids_separate.append([tokenizer.bos_token_id] + ent_ids_original_start + [tokenizer.eos_token_id])
            if len(ent_ids_flatten) > max_length-2:
                ent_ids_flatten = ent_ids_flatten[:max_length-2]
                break
            else:
                continue
    if len(ent_ids_flatten) ==0:
        ent_ids_flatten.extend(noname_token)

    # torch.LongTensor([[tokenizer.bos_token_id] + ent_ids_flatten + [tokenizer.eos_token_id]])
    ent_ids_flatten = [tokenizer.bos_token_id] + ent_ids_flatten + [tokenizer.eos_token_id]
    if len(ent_ids_flatten) < max_length:
        ent_ids_flatten = ent_ids_flatten + [tokenizer.pad_token_id] * (max_length -  len(ent_ids_flatten))
    
    ent_ids_separate.append([tokenizer.bos_token_id] + noname_token + [tokenizer.eos_token_id])
    ent_ids_separate = pad_list(ent_ids_separate, tokenizer.pad_token_id)
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


def get_person_ids_position(article_ids_replaced, person_token_id=40030, article_max_length=512, is_tgt_input=False):
    position_list = []
    # for i in range(len(article_ids_replaced)):
    i = 0
    while i < len(article_ids_replaced):
        position_i = []
        if article_ids_replaced[i] == person_token_id and i < article_max_length:
            if is_tgt_input:
                position_i.append(i+1)
            else:
                position_i.append(i)
            for j in range(i, len(article_ids_replaced)):
                if article_ids_replaced[j] == person_token_id:
                    continue
                else:
                    if is_tgt_input:
                        position_i.append(j)
                    else:
                        position_i.append(j-1)
                    i=j-1
                    # print(i)
                    break
            position_list.append(position_i)
            # print("i:",i)
        i += 1
    return position_list



# goodnews_vi_dataset.py
import os, json, copy
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from open_clip import tokenize as clip_tokenize   
import torchvision.transforms as T 
from torchvision.transforms import ToPILImage
# -------------------------------------------------


# def pad_tensor_feat(feat_np_list, pad_feat_tensor, expected_dim):
#     len_list = []

#     # Kiểm tra các tensor có hợp lệ không
#     for feat in feat_np_list:
#         if len(feat.shape) != 2 or feat.shape[1] != expected_dim or np.isnan(feat).any() or np.isinf(feat).any():
#             len_list.append(1)  # Tensor không hợp lệ, dùng độ dài 1
#         else:
#             len_list.append(feat.shape[0])

#     # Tính độ dài tối đa của các tensor hợp lệ
#     max_len = max(len_list) if max(len_list) > 0 else 1

#     padded_list = []
#     for i, feat in enumerate(feat_np_list):
#         # Kiểm tra lại tensor có hợp lệ không
#         if len(feat.shape) != 2 or feat.shape[1] != expected_dim or np.isnan(feat).any() or np.isinf(feat).any():
#             # Nếu không có mặt hoặc đối tượng, sử dụng tensor padding mặc định
#             padded = pad_feat_tensor.repeat(max_len, 1)
#         else:
#             # Nếu tensor hợp lệ, thực hiện padding
#             pad_num = max_len - feat.shape[0]
#             if pad_num > 0:
#                 to_be_padded = pad_feat_tensor.repeat(pad_num, 1)
#                 padded = torch.cat((torch.from_numpy(feat).float(), to_be_padded), dim=0)
#             else:
#                 padded = torch.from_numpy(feat).float()
#         padded_list.append(padded)

#     result = torch.stack(padded_list)

#     # Kiểm tra NaN/Inf trong kết quả đầu ra
#     if torch.isnan(result).any() or torch.isinf(result).any():
#         print(f"Cảnh báo: Tensor đầu ra chứa NaN hoặc Inf, trả về tensor mặc định")
#         return pad_feat_tensor.repeat(len(feat_np_list), max_len, 1)
    
#     return result



def pad_tensor_feat(feat_list, pad_feat_tensor):
    """
    Pad các tensor đặc trưng đến độ dài tối đa bằng cách thêm tensor pad.
    
    Tham số:
        feat_list (list of Tensor): Danh sách các tensor.
        pad_feat_tensor (torch.Tensor): Tensor pad để thêm vào.
    
    Trả về:
        torch.Tensor: Tensor đã được pad với kích thước (batch_size, max_len, feature_dim).
    """
    tensor_list = []
    
    for feat in feat_list:
        if isinstance(feat, np.ndarray):
            tensor_list.append(torch.from_numpy(feat))
        elif isinstance(feat, torch.Tensor):
            tensor_list.append(feat)
        else:
            raise TypeError(f"Expected np.ndarray or torch.Tensor, but got {type(feat)}")

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


import torch
from torchvision.transforms import ToPILImage
import torchvision.transforms as T

def collate_fn_goodnews_entity_type(batch, noname_id: int, tokenizer):
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
    face_pad = torch.ones((1, 512))
    obj_pad = torch.ones((1, 1000))
    person_id_positions_list = []
    person_id_positions_cap_list = []

    resize_transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    to_pil_image = ToPILImage()

    for i in range(len(batch)):
        article, article_ids, article_ner_mask_ids, caption, caption_ids, caption_ids_clip, \
        names_art, org_norp_gpe_loc_art, names_art_ids, org_norp_gpe_loc_art_ids, \
        names, org_norp_gpe_loc, names_ids, org_norp_gpe_loc_ids, all_gt_ner_ids, \
        all_gt_ner, face_emb, obj_emb, img_tensor, person_id_positions, person_id_positions_cap = \
            batch[i]["article"], batch[i]["article_ids"], batch[i]["article_ner_mask_ids"], \
            batch[i]["caption"], batch[i]["caption_ids"], batch[i]["caption_ids_clip"], \
            batch[i]["names_art"], batch[i]["org_norp_gpe_loc_art"], batch[i]["names_art_ids"], \
            batch[i]["org_norp_gpe_loc_art_ids"], batch[i]["names"], batch[i]["org_norp_gpe_loc"], \
            batch[i]["names_ids"], batch[i]["org_norp_gpe_loc_ids"], batch[i]["all_gt_ner_ids"], \
            batch[i]["all_gt_ner"], batch[i]["face_emb"], batch[i]["obj_emb"], batch[i]["img_tensor"], \
            batch[i]["person_id_positions"], batch[i]["person_id_positions_cap"]
        
        names_ids_flatten, org_norp_gpe_loc_ids_flatten = \
            batch[i]["names_ids_flatten"], batch[i]["org_norp_gpe_loc_ids_flatten"]

        # Xử lý kiểu dữ liệu của các trường ID
        def sanitize_ids(ids):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            if isinstance(ids, list) and all(isinstance(x, list) for x in ids):
                ids = [item for sublist in ids for item in sublist]  # Làm phẳng danh sách lồng nhau
            return [int(id) if id < tokenizer.vocab_size else noname_id for id in ids]

        names_ids = sanitize_ids(names_ids)
        names_art_ids = sanitize_ids(names_art_ids)
        names_ids_flatten = sanitize_ids(names_ids_flatten)
        org_norp_gpe_loc_ids = sanitize_ids(org_norp_gpe_loc_ids)
        org_norp_gpe_loc_ids_flatten = sanitize_ids(org_norp_gpe_loc_ids_flatten)
        all_gt_ner_ids = sanitize_ids(all_gt_ner_ids)
        org_norp_gpe_loc_art_ids = sanitize_ids(org_norp_gpe_loc_art_ids)

        names_ids_flatten_list.append(torch.tensor(names_ids_flatten, dtype=torch.long))
        org_norp_gpe_loc_ids_flatten_list.append(torch.tensor(org_norp_gpe_loc_ids_flatten, dtype=torch.long))
        names_ids_list.append(torch.tensor(names_ids, dtype=torch.long))
        names_art_ids_list.append(torch.tensor(names_art_ids, dtype=torch.long))
        org_norp_gpe_loc_ids_list.append(torch.tensor(org_norp_gpe_loc_ids, dtype=torch.long))
        org_norp_gpe_loc_art_ids_list.append(torch.tensor(org_norp_gpe_loc_art_ids, dtype=torch.long))
        all_gt_ner_ids_list.append(torch.tensor(all_gt_ner_ids, dtype=torch.long))

        article_list.append(article)
        article_id_list.append(article_ids)
        article_ner_mask_id_list.append(article_ner_mask_ids)
        caption_list.append(caption)
        caption_id_list.append(caption_ids)
        if caption_ids_clip is not None:
            caption_id_clip_list.append(caption_ids_clip)

        names_art_list.append(names_art)
        org_norp_gpe_loc_art_list.append(org_norp_gpe_loc_art)
        names_list.append(names)
        org_norp_gpe_loc_list.append(org_norp_gpe_loc)
        all_gt_ner_list.append(all_gt_ner)

        if face_emb is None or len(face_emb) == 0:
            face_emb = face_pad
        else:
            face_emb = torch.from_numpy(face_emb).float()
            face_emb = torch.nan_to_num(face_emb, nan=0.0, posinf=1e4, neginf=-1e4)
            if torch.isnan(face_emb).any() or torch.isinf(face_emb).any():
                print(f"Cảnh báo: face_emb chứa NaN/Inf tại mẫu {i}. Thay thế bằng tensor mặc định.")
                face_emb = face_pad

        if obj_emb is None or len(obj_emb) == 0:
            obj_emb = obj_pad
        else:
            obj_emb = torch.from_numpy(obj_emb).float()
            obj_emb = torch.nan_to_num(obj_emb, nan=0.0, posinf=1e4, neginf=-1e4)
            if torch.isnan(obj_emb).any() or torch.isinf(obj_emb).any():
                print(f"Cảnh báo: obj_emb chứa NaN/Inf tại mẫu {i}. Thay thế bằng tensor mặc định.")
                obj_emb = obj_pad

        face_emb_list.append(face_emb)
        obj_emb_list.append(obj_emb)

        if isinstance(img_tensor, torch.Tensor):
            img_pil = to_pil_image(img_tensor.squeeze(0))
        else:
            img_pil = img_tensor
        img_tensor_resized = resize_transform(img_pil)
        img_tensor_list.append(img_tensor_resized)

        person_id_positions_list.append(person_id_positions)
        person_id_positions_cap_list.append(person_id_positions_cap)
    
    max_len_input = get_max_len([article_id_list, article_ner_mask_id_list])
    article_ids_batch = pad_sequence(article_id_list, 1, max_len=max_len_input)
    article_ner_mask_ids_batch = pad_sequence(article_ner_mask_id_list, 1, max_len=max_len_input)
    caption_ids_batch = pad_sequence(caption_id_list, 1)
    if len(caption_id_clip_list) > 0:
        caption_ids_clip_batch = pad_sequence(caption_id_clip_list, 0)
    else:
        caption_ids_clip_batch = torch.empty((1,1))
    
    max_len_art_ids = get_max_len([names_art_ids_list, org_norp_gpe_loc_art_ids_list])
    names_art_ids_batch = pad_sequence(names_art_ids_list, 1, max_len=max_len_art_ids)
    org_norp_gpe_loc_art_ids_batch = pad_sequence(org_norp_gpe_loc_art_ids_list, 1, max_len=max_len_art_ids)

    max_len_name_ids = get_max_len_list(names_ids_list)
    names_ids_batch = pad_sequence_from_list(names_ids_list, special_token_id=noname_id, 
                                            bos_token_id=0, pad_token_id=1, eos_token_id=2, 
                                            max_len=max_len_name_ids)

    max_len_org_norp_gpe_loc_ids = get_max_len_list(org_norp_gpe_loc_ids_list)
    org_norp_gpe_loc_ids_batch = pad_sequence_from_list(org_norp_gpe_loc_ids_list, 
                                                       special_token_id=noname_id, 
                                                       bos_token_id=0, pad_token_id=1, 
                                                       eos_token_id=2, 
                                                       max_len=max_len_org_norp_gpe_loc_ids)

    all_gt_ner_ids_batch = pad_sequence(all_gt_ner_ids_list, 1)

    max_len_ids_flatten = get_max_len([names_ids_flatten_list, org_norp_gpe_loc_ids_flatten_list])
    names_ids_flatten_batch = pad_sequence(names_ids_flatten_list, 1, max_len=max_len_ids_flatten)
    org_norp_gpe_loc_ids_flatten_batch = pad_sequence(org_norp_gpe_loc_ids_flatten_list, 1, max_len=max_len_ids_flatten)
    
    img_batch = torch.stack(img_tensor_list, dim=0).squeeze(1)
    face_batch = pad_tensor_feat(face_emb_list, face_pad)
    obj_batch = pad_tensor_feat(obj_emb_list, obj_pad)

    # Kiểm tra NaN/Inf cho tất cả batch
    assert torch.isfinite(article_ids_batch).all(), "article_ids_batch chứa NaN/Inf"
    assert torch.isfinite(article_ner_mask_ids_batch).all(), "article_ner_mask_ids_batch chứa NaN/Inf"
    assert torch.isfinite(caption_ids_batch).all(), "caption_ids_batch chứa NaN/Inf"
    assert torch.isfinite(names_art_ids_batch).all(), "names_art_ids_batch chứa NaN/Inf"
    assert torch.isfinite(org_norp_gpe_loc_art_ids_batch).all(), "org_norp_gpe_loc_art_ids_batch chứa NaN/Inf"
    assert torch.isfinite(names_ids_batch).all(), "names_ids_batch chứa NaN/Inf"
    assert torch.isfinite(org_norp_gpe_loc_ids_batch).all(), "org_norp_gpe_loc_ids_batch chứa NaN/Inf"
    assert torch.isfinite(all_gt_ner_ids_batch).all(), "all_gt_ner_ids_batch chứa NaN/Inf"
    assert torch.isfinite(names_ids_flatten_batch).all(), "names_ids_flatten_batch chứa NaN/Inf"
    assert torch.isfinite(org_norp_gpe_loc_ids_flatten_batch).all(), "org_norp_gpe_loc_ids_flatten_batch chứa NaN/Inf"
    assert torch.isfinite(img_batch).all(), "img_batch chứa NaN/Inf"
    assert torch.isfinite(face_batch).all(), "face_batch chứa NaN/Inf"
    assert torch.isfinite(obj_batch).all(), "obj_batch chứa NaN/Inf"

    # Kiểm tra ID hợp lệ
    assert names_art_ids_batch.max() < tokenizer.vocab_size, f"names_art_ids_batch chứa ID vượt quá vocab_size: {names_art_ids_batch.max()} >= {tokenizer.vocab_size}"
    assert names_ids_flatten_batch.max() < tokenizer.vocab_size, f"names_ids_flatten_batch chứa ID vượt quá vocab_size: {names_ids_flatten_batch.max()} >= {tokenizer.vocab_size}"
    assert org_norp_gpe_loc_ids_flatten_batch.max() < tokenizer.vocab_size, f"org_norp_gpe_loc_ids_flatten_batch chứa ID vượt quá vocab_size: {org_norp_gpe_loc_ids_flatten_batch.max()} >= {tokenizer.vocab_size}"
    assert all_gt_ner_ids_batch.max() < tokenizer.vocab_size, f"all_gt_ner_ids_batch chứa ID vượt quá vocab_size: {all_gt_ner_ids_batch.max()} >= {tokenizer.vocab_size}"

    return {
        "article": article_list,
        "article_ids": article_ids_batch.squeeze(1),
        "article_ner_mask_ids": article_ner_mask_ids_batch.squeeze(1),
        "caption": caption_list,
        "caption_ids": caption_ids_batch.squeeze(1),
        "caption_ids_clip": caption_ids_clip_batch.squeeze(1),
        "names_art": names_art_list,
        "names_art_ids": names_art_ids_batch.squeeze(1),
        "org_norp_gpe_loc_art": org_norp_gpe_loc_art_list,
        "org_norp_gpe_loc_art_ids": org_norp_gpe_loc_art_ids_batch.squeeze(1),
        "names": names_list,
        "names_ids": names_ids_batch.squeeze(1),
        "org_norp_gpe_loc": org_norp_gpe_loc_list,
        "org_norp_gpe_loc_ids": org_norp_gpe_loc_ids_batch.squeeze(1),
        "all_gt_ner_ids": all_gt_ner_ids_batch.squeeze(1),
        "all_gt_ner": all_gt_ner_list,
        "face_emb": face_batch.float(),
        "obj_emb": obj_batch.float(),
        "img_tensor": img_batch,
        "person_id_positions": person_id_positions_list,
        "person_id_positions_cap": person_id_positions_cap_list,
        "names_ids_flatten": names_ids_flatten_batch.squeeze(1),
        "org_norp_gpe_loc_ids_flatten": org_norp_gpe_loc_ids_flatten_batch.squeeze(1)
    }


class GoodNewsDictDatasetEntityTypeFixLenEntPos(Dataset):
    """
    Phiên bản tiếng Việt: đọc thẳng sents_byclip trong JSON, face/obj embedding,
    ảnh & mask NER đã sinh ở bước pre-process.
    """
    def __init__(
        self,
        data_dict: dict,
        base_dir: str,
        tokenizer,
        transform,
        type: str         ,
        max_article_len: int = 512,
        max_ner_type_len: int = 80,
        max_ner_type_len_gt: int = 20,
        use_clip_tokenizer: bool = False,
        retrieved_sent: bool = True,
        
    ):
        super().__init__()
        self.meta = copy.deepcopy(data_dict)
        self.tok  = tokenizer
        self.tr   = transform
        self.clip = use_clip_tokenizer
        self.max_article = max_article_len
        self.max_ent_len = max_ner_type_len
        self.max_ent_gt  = max_ner_type_len_gt

        # ---- đường dẫn cố định theo folder bạn đưa ----
        self.dir_face   = os.path.join(base_dir, "faces")
        self.dir_obj    = os.path.join(base_dir, "objects")
        self.dir_mask   = os.path.join(base_dir, "article_all_ent_by_count_dir", type)
        self.dir_img    = "/data/npl/ICEK/Wikipedia/images"

        self.keys  = list(self.meta.keys())
        # token id động
        self.person_id  = self.tok.convert_tokens_to_ids("<PERSON>")

    # --------------------------------------------------

    def _load_npy(self, path, expected_dim):
        # Trả về tensor ones nếu thiếu file hoặc sai kích thước
        if not path or not os.path.isfile(path):
            return np.ones((1, expected_dim), dtype=np.float32)

        emb = np.load(path)
        # Nếu shape không khớp, thay bằng tensor mặc định
        if emb.ndim != 2 or emb.shape[1] != expected_dim:
            print(f"[WARN] {path} có shape {emb.shape}, mong đợi (*, {expected_dim}). "
                "Dùng padding mặc định.")
            return np.ones((1, expected_dim), dtype=np.float32)
        return emb


    def __getitem__(self, idx):
        max_retry = 10                                   # tránh vòng lặp vô hạn
        for _ in range(max_retry):
            hid  = self.keys[idx]
            item      = self.meta[hid]
            try:
        
     
            # ---------- ẢNH ----------
            
                img_path  = os.path.join(self.dir_img, f"{str(hid).zfill(10)}.jpg")
                img_pil  = safe_open_resize(img_path)
                if img_pil is None:                         
                  
                    img_pil = Image.new("RGB", (224, 224), color=(127, 127, 127))

                img_tensor = self.tr(img_pil).unsqueeze(0)  
                # img_tensor= self.tr(Image.open(img_path).convert("RGB")).unsqueeze(0)

                # ---------- Face & Obj ----------
                face_emb = self._load_npy(item.get("face_emb_dir"), expected_dim=512)
                obj_emb  = self._load_npy(item.get("obj_emb_dir"),  expected_dim=1000)

                # ---------- Văn bản ----------
                article  = item["sents_byclip"].replace("\n\n", " ")
                caption  = item["caption"]

                names          = item["names"]
                org_norp       = item["org_norp"]
                gpe_loc        = item["gpe_loc"]
                names_art      = item["names_art"]
                org_norp_art   = item["org_norp_art"]
                gpe_loc_art    = item["gpe_loc_art"]

                # khử trùng lặp
                def uniq(seq):                                                         
                    out=[]
                    for i, w in enumerate(seq):
                        if any(w in others for others in seq[:i]+seq[i+1:]): continue
                        out.append(w)
                    return out
                names_art_u  = uniq(names_art)
                org_art_u    = uniq(org_norp_art)
                loc_art_u    = uniq(gpe_loc_art)

                org_loc_art  = org_art_u + loc_art_u
                org_loc_cap  = org_norp + gpe_loc
                all_gt_ner   = names + org_loc_cap

                # ---------- Token hoá ----------
                concat_gt    = " ".join(all_gt_ner) if all_gt_ner else "<NONAME>"
                gt_ner_ids   = self.tok(concat_gt, max_length=self.max_ent_gt,
                                        truncation=True, padding="max_length",
                                        return_tensors="pt")["input_ids"]

                art_ids      = self.tok(article, max_length=self.max_article,
                                        truncation=True, padding=True,
                                        return_tensors="pt")["input_ids"]

            
                with open(os.path.join(self.dir_mask, f"{hid}.json")) as f:
                    mask_ids = torch.LongTensor(json.load(f)["input_ids"])   # 1-D tensor (L,)

                # ---- CHỈNH Ở ĐÂY -------------------------------------------------
                L = mask_ids.size(0)
                if L > self.max_article:                 # quá dài  →  cắt bớt
                    mask_ids = mask_ids[: self.max_article]

                elif L < self.max_article:               # thiếu     →  pad thêm <pad>(=1)
                    pad_len  = self.max_article - L
                    pad_part = torch.ones(pad_len, dtype=torch.long)  # 1 = <pad>
                    mask_ids = torch.cat([mask_ids, pad_part], dim=0)
                # -----------------------------------------------------------------

                art_mask_ids = mask_ids.unsqueeze(0)     # (1, max_article)

                # ----- vị trí PERSON trong mask -----
                person_pos = get_person_ids_position(
                                art_mask_ids.squeeze(0).tolist(),
                                person_token_id=self.person_id,
                                article_max_length=self.max_article)

                # caption ids
                cap_ids   = self.tok(caption, max_length=100,
                                    truncation=True, return_tensors="pt")["input_ids"]

                cap_ids_clip = clip_tokenize([caption], context_length=77) \
                            if self.clip else None

                # entity id helper
                # from utils_entity import make_new_entity_ids
                names_art_ids, _ = make_new_entity_ids(article, names_art_u,
                                                    self.tok, max_length=self.max_ent_len)
                org_loc_art_ids, _ = make_new_entity_ids(article, org_loc_art,
                                                        self.tok, max_length=self.max_ent_len)

                names_flat, names_ids = make_new_entity_ids(caption, names,
                                                            self.tok, max_length=self.max_ent_gt)
                orgloc_flat, orgloc_ids = make_new_entity_ids(caption, org_loc_cap,
                                                            self.tok, max_length=self.max_ent_gt)

                # -------------- return --------------
                return {
                    "article":                article,
                    "article_ids":            art_ids,
                    "article_ner_mask_ids":   art_mask_ids.unsqueeze(0),   # (1, L)
                    "caption":                caption,
                    "caption_ids":            cap_ids,
                    "caption_ids_clip":       cap_ids_clip,
                    "names_art":              names_art_u,
                    "org_norp_gpe_loc_art":   org_loc_art,
                    "names_art_ids":          names_art_ids,
                    "org_norp_gpe_loc_art_ids": org_loc_art_ids,
                    "names":                  names,
                    "org_norp_gpe_loc":       org_loc_cap,
                    "names_ids":              names_ids,
                    "org_norp_gpe_loc_ids":   orgloc_ids,
                    "all_gt_ner":             all_gt_ner,
                    "all_gt_ner_ids":         gt_ner_ids,
                    "face_emb":               face_emb,
                    "obj_emb":                obj_emb,
                    "img_tensor":             img_tensor,
                    "names_ids_flatten":      names_flat,
                    "org_norp_gpe_loc_ids_flatten": orgloc_flat,
                    "person_id_positions":    person_pos,
                    "person_id_positions_cap": item["name_pos_cap"],
                }
            except FileNotFoundError as e:
                import random
                import warnings
                warnings.warn(f"Thiếu file {e.filename}, skip sample {hid}")
                # chọn một idx ngẫu nhiên khác để thử lại
                idx = random.randint(0, len(self) - 1)
        raise RuntimeError(f"Không tìm được sample hợp lệ sau {max_retry} lần thử")
    
    def __len__(self):
        return len(self.meta)




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

def preprocess_article(article):
    out = []
    for sent in article:
        if sent == "\n":
            continue
        else:
            out.append(sent.replace("\n", ""))
    return out



def make_ner_dict_by_type(processed_doc, ent_list, ent_type_list):
    # make dict for unique ners with format as: {"Bush": PERSON_1}
    person_count = 1 # total count of PERSON type entities
    org_count = 1 # total count of ORG type entities
    gpe_count = 1 # total count of GPE type entities

    unique_ner_dict = {}
    new_ner_type_list = []

    for i, ent in enumerate(ent_list):
        if ent in unique_ner_dict.keys():
            new_ner_type_list.append(unique_ner_dict[ent])
        
        elif ent_type_list[i] == "PERSON" or ent_type_list[i] == "PER":
            ner_type = "<PERSON>_" + f"{person_count}"
            unique_ner_dict[ent] = ner_type
            new_ner_type_list.append(ner_type)
            person_count += 1
        elif ent_type_list[i] in ["ORGANIZATION", "ORG", "NORP"]:
            ner_type = "<ORG>_" + f"{org_count}"
            unique_ner_dict[ent] = ner_type
            new_ner_type_list.append(ner_type)
            org_count += 1
        elif ent_type_list[i] in ["GPE", "LOC"]:
            ner_type = "<LOC>_" + f"{gpe_count}"
            unique_ner_dict[ent] = ner_type
            new_ner_type_list.append(ner_type)
            gpe_count += 1
        
    entities_type = {} # dict with ner labels replaced by "PERSON_i", "ORG_j", "GPE_k"

    entities = get_entities(processed_doc)

    for i, ent in enumerate(entities):
        entities_type[i] = ent
        entities_type[i]["label"] = new_ner_type_list[i]
    # print(entities_type)

    start_pos_list = [sample["position"][0] for sample in entities_type.values()] # list of start positions for each entity
    # print(start_pos_list)
        
    return entities_type, start_pos_list, person_count, org_count, gpe_count


def make_new_articles_all_ent(processed_doc, start_pos_list, ent_length_list, entities_type_dict):
    # make new articles by replace PERSON/ORG/GPE type entities with their respective entity type
    counter = 0
    article_list = []
    article_list_unique_ner = []
    doc_len = len(processed_doc)
    ent_count = 0
    for i in range(doc_len):
        if i in start_pos_list and i+1 < doc_len:
            if processed_doc[i+1].is_punct or processed_doc[i+1].text == "'s":
                # if the entity is before punctuation or "'s"
                # we add it n-1 times concat with " ", 1 time with it self
                # n is the length of the tokenized entity from our tokenizer
                article_list_unique_ner.extend((ent_length_list[ent_count]-1) * [entities_type_dict[counter]["label"]+" "])
                article_list_unique_ner.append(entities_type_dict[counter]["label"])
                article_list.extend((ent_length_list[ent_count]-1) * [entities_type_dict[counter]["label"].split("_")[0]+ " "])
                article_list.append(entities_type_dict[counter]["label"].split("_")[0])
            else:
                article_list_unique_ner.extend(ent_length_list[ent_count] * [entities_type_dict[counter]["label"]+" "])
                article_list.extend(ent_length_list[ent_count] * [entities_type_dict[counter]["label"].split("_")[0]+" "])
            counter += 1 
            ent_count += 1
            start_pos_list = start_pos_list[1:]
        else:
            article_list_unique_ner.append(processed_doc[i].text_with_ws)
            article_list.append(processed_doc[i].text_with_ws)
    new_article_unique_ner = "".join(article_list_unique_ner)
    new_article = "".join(article_list)

    # print(new_article)
    # print(processed_doc)

    return new_article, new_article_unique_ner




def make_new_article_ids_all_ent(article_full, ent_list, entities_type, tokenizer):
    article_ids_ner = tokenizer(article_full)["input_ids"]
    # article_ids_ner_count = tokenizer(article_full)["input_ids"]
    counter = 0
    for ent in ent_list:
        # in case entities were in the middle of the sentence
        # print(ent)
        ent_ids_original = tokenizer(f" {ent}")["input_ids"][1:-1]
        # print(ent_ids_original)
        # if ent_ids_original in article_ids_ner:
        idx = find_first_sublist(article_ids_ner, ent_ids_original, start=0)
        if idx is not None:
            # print(ent_ids_original)
            ner_chain = " ".join([entities_type[counter]["label"].split("_")[0]] * len(ent_ids_original))
            # print(tokenizer(ner_chain)["input_ids"][1:-1])
            article_ids_ner = replace_sublist(article_ids_ner, ent_ids_original, tokenizer(ner_chain)["input_ids"][1:-1])
        else:
            # print(ent_ids_original)
            # in case entities were at the start of the sentence
            ent_ids_original_start = tokenizer(f"{ent}")["input_ids"][1:-1]
            ner_chain = " ".join([entities_type[counter]["label"].split("_")[0]] * len(ent_ids_original_start))
            article_ids_ner = replace_sublist(article_ids_ner, ent_ids_original_start, tokenizer(ner_chain)["input_ids"][1:-1])
        # print(article_ids_ner)
        # # in case entities were in the middle of the sentence
        # ent_ids_original = tokenizer(f" {ent}")["input_ids"][1:-1]
        # ner_chain_by_count = " ".join([entities_type[counter]["label"]] * len(ent_ids_original))
        # article_ids_ner_count = replace_sublist(article_ids_ner_count, ent_ids_original, tokenizer(ner_chain_by_count)["input_ids"][1:-1])
        # # in case entities were at the start of the sentence
        # ent_ids_original_start = tokenizer(f"{ent}")["input_ids"][1:-1]
        # article_ids_ner_count = replace_sublist(article_ids_ner_count, ent_ids_original_start, tokenizer(ner_chain_by_count)["input_ids"][1:-1])
        
        counter += 1
    # print(len(article_ids_ner), len(tokenizer(article_full)["input_ids"]))
    # return article_ids_ner, article_ids_ner_count
    return {"input_ids":article_ids_ner}


def replace_sublist(seq, sublist, replacement):
    length = len(replacement)
    index = 0
    for start, end in iter(lambda: find_first_sublist(seq, sublist, index), None):
        seq[start:end] = replacement
        index = start + length
    return seq

def find_first_sublist(seq, sublist, start=0):
    length = len(sublist)
    for index in range(start, len(seq)):
        if seq[index:index+length] == sublist:
            return index, index+length


def get_caption_with_ent_type(nlp, caption, tokenizer):
    processed_doc = nlp(caption)
    entities = get_entities(processed_doc)
        
    ent_list = [ entities[i]["text"] for i in range(len(entities)) ]
    ent_type_list = [ entities[i]["label"] for i in range(len(entities)) ]
        
    entities_type, start_pos_list, _, _, _ = make_ner_dict_by_type(processed_doc, ent_list, ent_type_list)

    new_caption, caption_ids_ner = make_new_caption_ids_all_ent(caption, ent_list, entities_type, tokenizer)
    return new_caption, caption_ids_ner


def make_new_caption_ids_all_ent(caption, ent_list, entities_type, tokenizer):
    caption_ids_ner = tokenizer(caption)["input_ids"]
    # caption_ids_ner_count = tokenizer(article_full)["input_ids"]
    counter = 0
    for ent in ent_list:
        # in case entities were in the middle of the sentence
        ent_ids_original = tokenizer(f" {ent}")["input_ids"][1:-1]
        # if ent_ids_original in caption_ids_ner:
        idx = find_first_sublist(caption_ids_ner, ent_ids_original, start=0)
        if idx is not None:
            ner_chain = " ".join([entities_type[counter]["label"].split("_")[0]] * len(ent_ids_original))
            caption_ids_ner = replace_sublist(caption_ids_ner, ent_ids_original, tokenizer(ner_chain)["input_ids"][1:-1])
        else:
            # in case entities were at the start of the sentence
            ent_ids_original_start = tokenizer(f"{ent}")["input_ids"][1:-1]
            ner_chain = " ".join([entities_type[counter]["label"].split("_")[0]] * len(ent_ids_original_start))
            caption_ids_ner = replace_sublist(caption_ids_ner, ent_ids_original_start, tokenizer(ner_chain)["input_ids"][1:-1])
        counter += 1
    return tokenizer.decode(caption_ids_ner), caption_ids_ner


def add_name_pos_list_to_dict(data_dict, nlp, tokenizer):
    new_dict = {}
    for key, value in tqdm(data_dict.items()):
        new_dict[key] = {}
        new_dict[key] = value
        _, caption_ids_ner = get_caption_with_ent_type(nlp, value["caption"], tokenizer)
        position_list = get_person_ids_position(caption_ids_ner, person_token_id=40030, article_max_length=20, is_tgt_input=True)

        new_dict[key]["name_pos_cap"] = position_list
    return new_dict


if __name__ == "__main__":
    nlp = stanza.Pipeline(lang='vi', processors='tokenize,ner')
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained("/data/npl/ICEK/License-Plate-Detection-Pipeline-with-Experiment-Tracking/assets/bartpho-syllable")
    model     = AutoModelForSeq2SeqLM.from_pretrained("/data/npl/ICEK/License-Plate-Detection-Pipeline-with-Experiment-Tracking/assets/bartpho-syllable")
    SPECIAL_TOKENS = ['<PERSON>', '<ORG>', '<LOC>', '<NONAME>']
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    PERSON_ID = tokenizer.convert_tokens_to_ids('<PERSON>')
    print('PERSON_ID: ',PERSON_ID)
    model.resize_token_embeddings(len(tokenizer))
    import json 
    with open('/data/npl/ICEK/DATASET/content/vacnic/final/test_vacnic_final.json','r',encoding='utf-8') as f:
        data_dict = json.load(f)

    train_clip_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    from functools import partial
    noname_id = tokenizer.convert_tokens_to_ids("<NONAME>")
    print('noname_id: ',noname_id)
    

    sample_ds = GoodNewsDictDatasetEntityTypeFixLenEntPos(
        data_dict,
        base_dir="/data/npl/ICEK/DATASET/content/vacnic",
        tokenizer=tokenizer,
        transform=train_clip_transform,
        use_clip_tokenizer=True
    )
    dataloader = DataLoader(
        sample_ds,
        batch_size=4,
        shuffle=True,
        collate_fn=partial(collate_fn_goodnews_entity_type, noname_id=noname_id)
    )
    print(sample_ds[0].keys())           # đủ 23 khóa
    b = next(iter(dataloader))
    print(b["article_ids"].shape)        # (B, L)
    print(len(b["names_art"]))           # batch list gốc của caption names


