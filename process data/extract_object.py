import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import json
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import json
import os
MAX_PIXELS = 89478485
RESAMPLING = Image.LANCZOS  


# Khởi tạo mô hình và biến đổi
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Tải mô hình ResNet50
resnet_model = models.resnet50(pretrained=True)
resnet_model.to(device)
resnet_model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Tải mô hình Faster R-CNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn
obj_detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
obj_detection_model.to(device)
obj_detection_model.eval()

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


# Hàm phát hiện đối tượng
def detect_objects(image_path, image_url):
    img = load_image(image_path)
    if img is None:
        print(f"Không thể tải ảnh từ {image_path}.")
        print(f"Tiến hành tải ảnh từ {image_url}")
        img = load_image(image_url)
        if img is None:
            print(f"Ảnh mạng cũng lỗi rồi bạn eiiiii")
            return [],[]
        
    # img = Image.open(image_path).convert('RGB')
    img_t = transforms.ToTensor()(img).to(device)
    with torch.no_grad():
        predictions = obj_detection_model([img_t])
    score_threshold = 0.5
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']
    indices = scores > score_threshold
    boxes = boxes[indices].cpu().int().tolist()
    labels = labels[indices].cpu().tolist()
    return boxes, labels

# import clip
# # Tải mô hình CLIP
# model, preprocess = clip.load("ViT-B/32")

# def extract_object_embedding_clip(image_path, bbox):
#     img = Image.open(image_path).convert('RGB')
#     obj_img = img.crop(bbox)
#     img_preprocessed = preprocess(obj_img).unsqueeze(0)

#     with torch.no_grad():
#         embedding = model.encode_image(img_preprocessed)

#     embedding = embedding / embedding.norm(dim=-1, keepdim=True)
#     embedding_np = embedding.cpu().numpy()
#     return embedding_np

# Hàm trích xuất embedding đối tượng
def extract_object_embedding(image_path,image_url, bbox):
    img = load_image(image_path)
    if img is None:
        print(f"Không thể tải ảnh từ {image_path}.")
        print(f"Tiến hành tải ảnh từ {image_url}")
        img = load_image(image_url)
        if img is None:
            print(f"Ảnh mạng cũng lỗi rồi bạn eiiiii")
            return []
    # img = Image.open(image_path).convert('RGB')
    obj_img = img.crop(bbox)
    img_t = preprocess(obj_img).to(device)
    batch_t = torch.unsqueeze(img_t, 0)
    with torch.no_grad():
        embedding = resnet_model(batch_t)
    embedding_np = embedding.cpu().numpy().squeeze()
    return embedding_np

# # Hàm lưu embedding đối tượng
# def save_obj_embeddings(hash_id, obj_embeddings, obj_labels):
#     obj_emb_dir_list = []
#     for embedding, label in zip(obj_embeddings, obj_labels):
#         clean_lbl = label.replace(' ', '_')
#         filename = f"objects/{hash_id}_{clean_lbl}.npy"
#         np.save(filename, embedding)
#         obj_emb_dir_list.append(filename)
#     return obj_emb_dir_list

# # Hàm cập nhật JSON
# def update_json_data(json_path, hash_id, obj_emb_dir_list):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#     data[hash_id]['obj_emb_dir'] = obj_emb_dir_list
#     with open(json_path, 'w') as f:
#         json.dump(data, f)

# # Danh sách tên nhãn từ COCO
# COCO_INSTANCE_CATEGORY_NAMES = [
#     '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     # ... thêm các nhãn khác ...
# ]

# # Hàm xử lý hình ảnh
# def process_image(hash_id, image_dir, json_path):
#     image_path = os.path.join(image_dir, f"{hash_id}.jpg")
#     boxes, labels = detect_objects(image_path)
#     if not boxes:
#         print(f"No objects found in image {hash_id}.jpg")
#         return
#     obj_embeddings = []
#     obj_labels = []
#     for box, label_idx in zip(boxes, labels):
#         embedding = extract_object_embedding(image_path, box)
#         obj_embeddings.append(embedding)
#         label_name = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
#         obj_labels.append(label_name)
#     obj_emb_dir_list = save_obj_embeddings(hash_id, obj_embeddings, obj_labels)
#     update_json_data(json_path, hash_id, obj_emb_dir_list)
#     print(f"Processed image {hash_id}.jpg and updated JSON.")

# # Ví dụ sử dụng
# hash_id = '123abc'
# image_dir = 'images_processed'
# json_path = 'data.json'
# os.makedirs('objects', exist_ok=True)
# process_image(hash_id, image_dir, json_path)
# a,b = detect_objects(r"/data/npl/ICEK/Wikipedia/images/0000000006.jpg",'')
# for i in a:
#     print(extract_object_embedding(r"/data/npl/ICEK/Wikipedia/images/0000000006.jpg","",i))