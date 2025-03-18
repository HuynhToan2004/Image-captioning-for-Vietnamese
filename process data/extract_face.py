from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms
import json
import os

MAX_PIXELS = 89478485
RESAMPLING = Image.LANCZOS  

detector = MTCNN()
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
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

def extract_faces(image_path, image_url):
    image = load_image(image_path)
    if image is None:
        print(f"Không thể tải ảnh từ {image_path}.")
        print(f"Tiến hành tải ảnh từ {image_url}")
        image = load_image(image_url)
        if image is None:
            print(f"Ảnh mạng cũng lỗi rồi bạn eiiiii")
            return []

    pixels = np.asarray(image)
    results = detector.detect_faces(pixels)
    faces = []
    if not results:
        print("Không phát hiện khuôn mặt nào trong ảnh.")
        return faces
    for i, result in enumerate(results):
        x, y, width, height = result['box']
        x, y = abs(x), abs(y)
        face = pixels[y:y+height, x:x+width]
        face_image = Image.fromarray(face)
        faces.append(face_image)
    return faces

def get_face_embedding(face_image):
    # Chuyển đổi ảnh khuôn mặt thành tensor
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        # Chuẩn hóa theo chuẩn của FaceNet
        transforms.Normalize([0.5], [0.5])
    ])
    face_tensor = transform(face_image).unsqueeze(0)
    # Tạo embedding
    with torch.no_grad():
        embedding = facenet_model(face_tensor)
    return embedding.squeeze(0).numpy()

def label_faces(faces, names_list):
    face_names = []
    for i, face in enumerate(faces):
        # Hiển thị khuôn mặt để nhận diện
        face.show()
        print(f"Danh sách tên có sẵn: {names_list}")
        name = input(f"Nhập tên cho khuôn mặt {i+1}: ")
        while name not in names_list:
            print("Tên không có trong danh sách. Vui lòng nhập tên hợp lệ.")
            name = input(f"Nhập tên cho khuôn mặt {i+1}: ")
        face_names.append(name)
    return face_names

def save_face_embeddings(hash_id, face_embeddings, face_names):
    face_emb_dir_list = []
    for embedding, name in zip(face_embeddings, face_names):
        filename = f"faces/{hash_id}_{name.replace(' ', '_')}.npy"
        np.save(filename, embedding)
        face_emb_dir_list.append(filename)
    return face_emb_dir_list

def update_json_data(json_path, hash_id, face_emb_dir_list):
    with open(json_path, 'r') as f:
        data = json.load(f)
    data[hash_id]['face_emb_dir'] = face_emb_dir_list
    with open(json_path, 'w') as f:
        json.dump(data, f)

def process_image(hash_id, image_dir, json_path):
    image_path = os.path.join(image_dir, f"{hash_id}.jpg")
    faces = extract_faces(image_path)
    if not faces:
        print(f"Không tìm thấy khuôn mặt trong ảnh {hash_id}.jpg")
        return

    # Load danh sách tên từ JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    names_list = data[hash_id]['names']

    # Gán tên cho mỗi khuôn mặt
    face_names = label_faces(faces, names_list)

    # Tạo embedding cho mỗi khuôn mặt
    face_embeddings = [get_face_embedding(face) for face in faces]

    # Lưu trữ embedding và cập nhật JSON
    face_emb_dir_list = save_face_embeddings(hash_id, face_embeddings, face_names)
    update_json_data(json_path, hash_id, face_emb_dir_list)
    print(f"Đã xử lý ảnh {hash_id}.jpg và cập nhật JSON.")

# Ví dụ sử dụng
# hash_id = '123abc'
# image_dir = 'images_processed'
# json_path = 'data.json'
# process_image(hash_id, image_dir, json_path)


# face = extract_faces(r"/data/npl/ICEK/Wikipedia/images/0000000007.jpg",'')
# print(len(face))
# import numpy as np
# np.save(os.path.join("/data/npl/ICEK/VACNIC/data/train/faces", "hehe.npy"),get_face_embedding(face[0]))