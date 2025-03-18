# import open_clip
# from PIL import Image
# from multilingual_clip import pt_multilingual_clip
# import transformers
# import torch.nn.functional as F
# from PIL import Image
# import requests

##### chạy file này xong sẽ có dữ liệu => gửi dữ liệu về local để chạy NER 
from sentence_transformers import SentenceTransformer, util
from PIL import Image, __version__ as PILLOW_VERSION
import glob
import torch
import pickle
import zipfile
from IPython.display import display
from IPython.display import Image as IPImage
import os
import json 
from tqdm.autonotebook import tqdm
import requests
from io import BytesIO
from packaging import version
MAX_PIXELS = 178956970  # Giới hạn mặc định của PIL

# Kiểm tra phiên bản Pillow để xác định chế độ resampling
if version.parse(PILLOW_VERSION) >= version.parse("10.0.0"):
    RESAMPLING = Image.Resampling.LANCZOS
else:
    RESAMPLING = Image.ANTIALIAS

# Kiểm tra và thiết lập thiết bị
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Khởi tạo mô hình với thiết bị cụ thể
text_model = SentenceTransformer('clip-ViT-B-32-multilingual-v1', device=device)
image_model = SentenceTransformer('clip-ViT-B-32', device=device)

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
            response = requests.get(path_or_url,headers=headers)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(path_or_url)

        # Kiểm tra và chuyển đổi ảnh sang RGB nếu cần
        if image.mode != 'RGB':
            print(f"Chuyển đổi ảnh từ {image.mode} sang RGB.")
            image = image.convert('RGB')

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
    
def calculate_cosine_similarity(image_embeddings, text_embeddings):
    cosine_scores = util.cos_sim(image_embeddings, text_embeddings)
    return cosine_scores

def retrieve_relevant_sentences(image_path, image_url, context, threshold=0.25, max_sentences=25):
    """
    Tìm các câu trong context liên quan đến hình ảnh tại image_path dựa trên độ tương đồng cosine.

    Parameters:
    - image_path (str): Đường dẫn đến tệp hình ảnh.
    - image_url (str): URL của hình ảnh (nếu không tải được từ đường dẫn).
    - context (list of str): Danh sách các câu văn để so sánh.
    - threshold (float): Ngưỡng độ tương đồng để chọn câu. Mặc định là 0.25.
    - max_sentences (int): Số lượng câu tối đa để trả về. Mặc định là 25.

    Returns:
    - str: Các câu liên quan được nối bằng hai ký tự xuống dòng.
    - list of float: Danh sách các điểm số tương đồng của các câu liên quan.
    """
    topk_sentences = []
    topk_scores = []

    try:
        # Tải và mã hóa hình ảnh
        image = load_image(image_path)
        if image is None:
            print(f"Không thể tải ảnh từ '{image_path}'.")
            print(f"Tiến hành tải từ {image_url} ")
            image = load_image(image_url)
            if image is None:
                return "", []

        # Mã hóa hình ảnh
        image_embeddings = image_model.encode([image], convert_to_tensor=True, device=device, show_progress_bar=False)
        # Nếu chỉ có một hình ảnh, trích xuất embedding đầu tiên
        if len(image_embeddings) == 1:
            image_embeddings = image_embeddings[0].unsqueeze(0)
    except Exception as e:
        print(f"Lỗi khi xử lý hình ảnh: {e}")
        return "", []

    try:
        if not isinstance(context, list):
            raise ValueError("Tham số 'context' phải là danh sách các chuỗi văn bản.")
        if not all(isinstance(sentence, str) for sentence in context):
            raise ValueError("Tất cả các phần tử trong 'context' phải là chuỗi văn bản.")

        text_embeddings = text_model.encode(context, convert_to_tensor=True, device=device, show_progress_bar=False)
    except Exception as e:
        print(f"Lỗi khi mã hóa văn bản: {e}")
        return "", []

    try:
        # Tính độ tương đồng cosine giữa hình ảnh và văn bản
        cosine_scores = calculate_cosine_similarity(image_embeddings, text_embeddings)

        # Thu thập các câu và điểm số tương đồng vượt ngưỡng
        sentences_scores = []
        for sentence_idx, score in enumerate(cosine_scores[0]):
            if score > threshold:
                sentences_scores.append((context[sentence_idx], score.item()))

        # Sắp xếp các câu theo điểm số tương đồng giảm dần
        sorted_sentences_scores = sorted(sentences_scores, key=lambda x: x[1], reverse=True)

        # Lấy tối đa max_sentences câu
        topk_sentences_scores = sorted_sentences_scores[:max_sentences]

        # Tách riêng câu và điểm số
        topk_sentences = [sentence for sentence, score in topk_sentences_scores]
        topk_scores = [score for sentence, score in topk_sentences_scores]

        return "\n\n".join(topk_sentences), topk_scores
    except Exception as e:
        print(f"Lỗi khi tính toán độ tương đồng: {e}")
        return "", []

            
# with open('/data/npl/ICEK/VACNIC/data/train/content_7.json','r',encoding='utf-8') as f:
#    data = json.load(f)

# image_path = "/data/npl/ICEK/Wikipedia/images/0000027592.jpg"

# for id in json_data:
#   context = json_data[id]['context']
#   for img in json_data[id]['images']:
#     topk_sentences, _ = related_sentences(img['path'], context, 0.22) # you can replace 'url' by 'path'
#     print(len(_))
#     print(_)

# context = get_article(data, image_path)
# a, b = related_sentences(image_path,context, threshold=0.25)
# print('length article: ', len(context))
# print(len(b))
# print(b)
# print(a)



# ====================================================VERSION 2=================================================
# from io import BytesIO
# import json 
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image
# from transformers import AutoTokenizer, AutoModel
# from io import BytesIO
# device = "cuda" if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {device}")
# model_name = "multilingual-CLIP/mclip-vit-base-patch32"  
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# model.to(device)
# model.eval()


# def retrieve_relevant_sentences(image_path, article_sentences: list, top_k=7, device=device):
#     # Xử lý hình ảnh
#     image = Image.open(image_path).convert('RGB')
#     preprocess = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
#                              std=[0.26862954, 0.26130258, 0.27577711]),
#     ])
#     image_tensor = preprocess(image).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         # Mã hóa hình ảnh
#         image_features = model.encode_image(image_tensor)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
    
#     # Cắt ngắn câu (nếu cần)
#     def truncate_sentences(sentences, tokenizer, max_length=512):
#         truncated = []
#         for sentence in sentences:
#             encoding = tokenizer.encode_plus(
#                 sentence,
#                 max_length=max_length,
#                 truncation=True,
#                 return_tensors='pt'
#             )
#             truncated_sentence = tokenizer.decode(encoding['input_ids'][0], skip_special_tokens=True)
#             truncated.append(truncated_sentence)
#         return truncated
    
#     truncated_sentences = truncate_sentences(article_sentences, tokenizer, max_length=512)
    
#     with torch.no_grad():
#         # Tokenize các câu và chuyển sang device
#         text_tokens = tokenizer(
#             truncated_sentences,
#             padding=True,
#             truncation=True,
#             max_length=77,
#             return_tensors='pt'
#         ).to(device)
        
#         # Mã hóa văn bản
#         text_features = model.encode_text(text_tokens)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
    
#     # Tính toán độ tương đồng cosine giữa hình ảnh và các câu văn bản
#     similarities = F.cosine_similarity(image_features, text_features)
    
#     # Lọc và chọn top_k câu có độ tương đồng cao nhất
#     topk_similarities, topk_indices = torch.topk(similarities, top_k)
    
#     topk_sentences = [truncated_sentences[idx] for idx in topk_indices.cpu().numpy()]
#     topk_scores = topk_similarities.cpu().numpy().tolist()
    
#     return "\n\n".join(topk_sentences), topk_scores




# =================================================VERSION 1===================================================

# def retrieve_relevant_sentences(image_path, article_sentences, top_k=7, device='cuda'):
#     # Khởi tạo mô hình ảnh
#     image_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
#     image_model.to(device)

#     # Xử lý hình ảnh
#     image = Image.open(image_path).convert('RGB')  # Đảm bảo hình ảnh ở định dạng RGB
#     image = preprocess(image).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         image_features = image_model.encode_image(image)
#         image_features /= image_features.norm(dim=-1, keepdim=True)  

#     # Khởi tạo mô hình văn bản và tokenizer
#     model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus'
#     text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
#     text_model.to(device)
#     text_model.eval()  # Đặt mô hình ở chế độ đánh giá
#     tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
#     # Định nghĩa hàm để cắt ngắn câu
#     def truncate_sentences(sentences, tokenizer, max_length=512):
#         truncated = []
#         for sentence in sentences:
#             # Token hóa câu với truncation
#             encoding = tokenizer.encode_plus(
#                 sentence,
#                 max_length=max_length,
#                 truncation=True,
#                 return_tensors='pt'
#             )
#             # Chuyển các tensor sang device
#             encoding = {key: value.to(device) for key, value in encoding.items()}
#             # Giải mã lại câu đã cắt ngắn
#             truncated_sentence = tokenizer.decode(encoding['input_ids'][0], skip_special_tokens=True)
#             truncated.append(truncated_sentence)
#         return truncated
    
#     # Cắt ngắn các câu trong bài viết
#     truncated_sentences = truncate_sentences(article_sentences, tokenizer, max_length=512)
    
#     with torch.no_grad():
#         # Chuyển các tensor đầu vào sang device trong hàm forward của mô hình văn bản
#         # Nếu mô hình không tự động chuyển, bạn cần điều chỉnh nó
#         # Giả sử bạn có thể sửa hàm forward của text_model như sau:
#         def forward_with_device(model, sentences, tokenizer):
#             txt_tok = tokenizer(
#                 sentences,
#                 return_tensors="pt",
#                 padding=True,
#                 truncation=True
#             ).to(device)  # Đảm bảo các tensor trên device
#             embs = model.transformer(**txt_tok)[0]
#             return embs

#         text_features = forward_with_device(text_model, truncated_sentences, tokenizer)
#         text_features /= text_features.norm(dim=-1, keepdim=True)  

#     # Tính toán độ tương đồng cosine giữa hình ảnh và các câu văn bản
#     similarities = F.cosine_similarity(image_features, text_features)

#     # Lọc và chọn top_k câu có độ tương đồng cao nhất
#     topk_similarities, topk_indices = torch.topk(similarities, top_k)
    
#     topk_sentences = [truncated_sentences[idx] for idx in topk_indices.cpu().numpy()]
#     topk_scores = topk_similarities.cpu().numpy().tolist()

#     return "\n\n".join(topk_sentences), topk_scores



# def open_image_from_url(image_url):
#     response = requests.get(image_url)
#     img_data = BytesIO(response.content)
#     image = Image.open(img_data)
#     return image

# def get_article(data,img_path):
#     # for id in data:
#     #     for img in range(len(data[id]['images'])):
#     #         if data[id]['images'][img]['path'] == img_path:
#     #             return data[id]['context']
            
#     for id, content in data.items():
#         for img in content.get('images',[]):
#             if img.get('path','') == img_path:
#                 return content.get('context',[])
            
# import json 

# with open('/data/npl/ICEK/VACNIC/data/test/content_demo.json','r',encoding='utf-8') as f:
#     data = json.load(f)

# for id, content in data.items():

#     context = content.get('context',[])
#     images = content.get("images", [])
#     total_images = len(images)
#     for image_data in tqdm(images, total=total_images, desc='Processing Images', unit='image', leave=False):
#         image_path = image_data["path"]
#         caption = image_data["caption"]
#         topk_sentences, _ = retrieve_relevant_sentences(image_path, context)
#         print(len(_))
#         print(_)