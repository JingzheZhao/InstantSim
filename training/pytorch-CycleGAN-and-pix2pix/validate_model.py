import os
import json
import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from app.models.networks import SolarGenerator

MODEL_PATH = r"checkpoints/solar_project/latest_G.pth"
TEST_DATA_DIR = r"datasets/direct_sun_hours/test"
JSON_FILE = r"datasets/direct_sun_hours/test_data_log.jsonl"
TEST_WEB_DIR = r"checkpoints/solar_project/test_web"
TEST_IMG_DIR = os.path.join(TEST_WEB_DIR, "images")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(TEST_IMG_DIR, exist_ok=True)

def build_vec36_from_json_vectors(vecs):
    raw27 = np.zeros(27, dtype=np.float32)
    mask9 = np.zeros(9, dtype=np.float32)

    num_to_fill = min(len(vecs), 9)
    if num_to_fill > 0:
        raw = np.array(vecs[:num_to_fill], dtype=np.float32).reshape(-1)
        raw27[: num_to_fill * 3] = raw[: num_to_fill * 3]
        mask9[:num_to_fill] = 1.0

    return torch.tensor(np.concatenate([raw27, mask9]), dtype=torch.float32)

vector_map = {}
with open(JSON_FILE, "r") as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            img_id = str(entry["id"])
            vecs = entry.get("vectors", [])
            vector_map[img_id] = build_vec36_from_json_vectors(vecs)

model = SolarGenerator(input_nc=3, output_nc=3, vec_dim=36).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def write_test_html(web_dir, results):
    html_path = os.path.join(web_dir, "index.html")
    with open(html_path, "w") as f:
        f.write("""
        <html>
        <head><title>Solar Model Test Report</title><style>
            body { font-family: sans-serif; background: #f0f0f0; padding: 20px; }
            table { width: 100%; border-collapse: collapse; background: white; }
            th, td { padding: 15px; border: 1px solid #ddd; text-align: center; }
            img { max-width: 100%; height: auto; border: 1px solid #eee; }
            h1 { color: #333; }
        </style></head>
        <body>
            <h1>AI Solar Test Results (Conditional U-Net)</h1>
            <table>
                <tr><th>ID</th><th>Input | AI Prediction | Real Ladybug</th></tr>
        """)
        for img_id in results:
            f.write(f"<tr><td>{img_id}</td><td><img src='images/result_{img_id}.png'></td></tr>")
        f.write("</table></body></html>")

# ✅ 和训练一致：A 用 NEAREST，B 本来只是展示/对比，Resize 用 BILINEAR 更合理
transform_A = transforms.Compose([
    transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_B = transforms.Compose([
    transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_files = sorted([f for f in os.listdir(TEST_DATA_DIR) if f.endswith(".png")])
processed_ids = []

print("🚀 开始生成 Web 测试报告...")
with torch.no_grad():
    for img_name in test_files:
        img_id = img_name.split(".")[0]
        full_img = cv2.imread(os.path.join(TEST_DATA_DIR, img_name))
        full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
        w_half = full_img.shape[1] // 2

        real_A = full_img[:, :w_half, :]
        real_B = full_img[:, w_half:, :]

        input_tensor = transform_A(Image.fromarray(real_A)).unsqueeze(0).to(DEVICE)
        vec_tensor = vector_map.get(img_id, torch.zeros(36)).unsqueeze(0).to(DEVICE)

        fake_B_tensor = model(input_tensor, vec_tensor)

        def tensor_to_img(t):
            img = ((t[0].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).astype(np.uint8)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 仅用于展示：输入A、真值B都 resize 到 256
        img_A = cv2.resize(cv2.cvtColor(real_A, cv2.COLOR_RGB2BGR), (256, 256), interpolation=cv2.INTER_NEAREST)
        img_Fake = tensor_to_img(fake_B_tensor)
        img_Real = cv2.resize(cv2.cvtColor(real_B, cv2.COLOR_RGB2BGR), (256, 256), interpolation=cv2.INTER_LINEAR)

        comparison = np.hstack([img_A, img_Fake, img_Real])
        cv2.imwrite(os.path.join(TEST_IMG_DIR, f"result_{img_id}.png"), comparison)
        processed_ids.append(img_id)

write_test_html(TEST_WEB_DIR, processed_ids)
print(f"🎉 测试报告已生成！请打开查看: {os.path.abspath(os.path.join(TEST_WEB_DIR, 'index.html'))}")
