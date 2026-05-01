import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode  # Required to specify Resize interpolation mode
import torch.backends.cudnn as cudnn

# ================= Configuration =================
DATASET_ROOT = r"datasets/direct_sun_hours"
JSON_FILE = "train_data_log.jsonl"
CHECKPOINTS_BASE = r"checkpoints/solar_project"

BATCH_SIZE = 16
LR = 0.0001
EPOCHS = 200
SAVE_EPOCH_FREQ = 5
L1_LAMBDA = 50.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Condition vector: 27 (9x3 sun vectors) + 9 (hour masks)
VEC_DIM = 36
EMB_DIM = 128
D_COND_CH = 16

# ================= 1. Sun Condition Encoder (9x3 + 9mask -> embedding) =================
class SunConditionEncoder(nn.Module):
    """
    Input vec: (B, 36) = [9x3 vectors, 9 masks]
    Output emb: (B, EMB_DIM)

    Key design: preserves the order of 9 hourly sun vectors (no mean pooling)
    Strategy: per-vector MLP -> (B,9,emb) -> mask -> flatten -> concat(mask, daylen) -> fuse
    """
    def __init__(self, emb_dim=128):
        super().__init__()
        self.emb_dim = emb_dim

        # per-vector shared MLP
        self.per_vec = nn.Sequential(
            nn.Linear(3, emb_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(0.2, True),
        )

        # Fuse input dim: 9*emb_dim + 9(mask) + 1(daylen)
        fuse_in = (9 * emb_dim) + 9 + 1
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, emb_dim),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, vec):
        # vec: (B,36)
        v = vec[:, :27].view(-1, 9, 3)          # (B,9,3)
        m = vec[:, 27:].view(-1, 9, 1)          # (B,9,1)
        m = (m > 0.5).float()

        # per-vector embedding: (B,9,emb_dim)
        feat = self.per_vec(v)

        # Preserve order: apply mask without averaging
        feat = feat * m                         # (B,9,emb_dim)
        feat_flat = feat.reshape(feat.size(0), -1)      # (B, 9*emb_dim)

        # mask flat (B,9)
        mask_flat = m.view(m.size(0), 9)

        # daylen (B,1) = ratio of valid (daytime) hours
        denom = mask_flat.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B,1)
        daylen = denom / 9.0

        # fuse
        emb = self.fuse(torch.cat([feat_flat, mask_flat, daylen], dim=1))
        return emb

# ================= 2. FiLM / Conditional InstanceNorm =================
class FiLMIN(nn.Module):
    """
    InstanceNorm + FiLM(γ,β) from embedding
    y = γ * IN(x) + β
    """
    def __init__(self, num_features, emb_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False, eps=1e-5)
        self.to_gamma_beta = nn.Linear(emb_dim, num_features * 2)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x, emb):
        b, c, _, _ = x.shape
        x = self.norm(x)
        gb = self.to_gamma_beta(emb)
        gamma, beta = gb[:, :c], gb[:, c:]
        gamma = gamma.view(b, c, 1, 1) + 1.0
        beta = beta.view(b, c, 1, 1)
        return gamma * x + beta

# ================= 3. Generator (U-Net + multi-scale FiLM) =================
class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, emb_dim, use_norm=True):
        super().__init__()
        self.use_norm = use_norm
        self.act = nn.LeakyReLU(0.2, True)
        self.conv = nn.Conv2d(in_c, out_c, 4, 2, 1)
        self.film = FiLMIN(out_c, emb_dim) if use_norm else None

    def forward(self, x, emb):
        x = self.act(x)
        x = self.conv(x)
        if self.use_norm:
            x = self.film(x, emb)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, emb_dim, dropout=False):
        super().__init__()
        self.act = nn.ReLU(True)
        self.deconv = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1)
        self.film = FiLMIN(out_c, emb_dim)
        self.dropout = nn.Dropout(0.5) if dropout else None

    def forward(self, x, emb):
        x = self.act(x)
        x = self.deconv(x)
        x = self.film(x, emb)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class SolarGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, vec_dim=VEC_DIM, emb_dim=EMB_DIM):
        super().__init__()
        self.cond_enc = SunConditionEncoder(emb_dim=emb_dim)

        # encoder
        self.enc1 = nn.Conv2d(input_nc, 64, 4, 2, 1)
        self.enc2 = DownBlock(64, 128, emb_dim, use_norm=True)
        self.enc3 = DownBlock(128, 256, emb_dim, use_norm=True)
        self.enc4 = DownBlock(256, 512, emb_dim, use_norm=True)
        self.enc5 = DownBlock(512, 512, emb_dim, use_norm=True)
        self.enc6 = DownBlock(512, 512, emb_dim, use_norm=True)
        self.enc7 = DownBlock(512, 512, emb_dim, use_norm=True)
        self.enc8_act = nn.LeakyReLU(0.2, True)
        self.enc8 = nn.Conv2d(512, 512, 4, 2, 1)

        self.bottleneck_fc = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.LeakyReLU(0.2, True)
        )

        # decoder
        self.dec1 = UpBlock(1024, 512, emb_dim, dropout=True)
        self.dec2 = UpBlock(1024, 512, emb_dim, dropout=True)
        self.dec3 = UpBlock(1024, 512, emb_dim, dropout=True)
        self.dec4 = UpBlock(1024, 512, emb_dim, dropout=False)
        self.dec5 = UpBlock(1024, 256, emb_dim, dropout=False)
        self.dec6 = UpBlock(512, 128, emb_dim, dropout=False)
        self.dec7 = UpBlock(256, 64, emb_dim, dropout=False)
        self.last = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(128, output_nc, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x, vec):
        emb = self.cond_enc(vec)

        e1 = self.enc1(x)
        e2 = self.enc2(e1, emb)
        e3 = self.enc3(e2, emb)
        e4 = self.enc4(e3, emb)
        e5 = self.enc5(e4, emb)
        e6 = self.enc6(e5, emb)
        e7 = self.enc7(e6, emb)
        e8 = self.enc8(self.enc8_act(e7))

        b, c, h, w = e8.size()
        v = self.bottleneck_fc(emb).view(b, 512, 1, 1).expand(b, 512, h, w)

        d1 = self.dec1(torch.cat([e8, v], 1), emb)
        d1 = torch.cat([d1, e7], 1)
        d2 = self.dec2(d1, emb); d2 = torch.cat([d2, e6], 1)
        d3 = self.dec3(d2, emb); d3 = torch.cat([d3, e5], 1)
        d4 = self.dec4(d3, emb); d4 = torch.cat([d4, e4], 1)
        d5 = self.dec5(d4, emb); d5 = torch.cat([d5, e3], 1)
        d6 = self.dec6(d5, emb); d6 = torch.cat([d6, e2], 1)
        d7 = self.dec7(d6, emb); d7 = torch.cat([d7, e1], 1)

        return self.last(d7)

# ================= 4. Discriminator (Conditional PatchGAN) =================
class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_nc=6, vec_dim=VEC_DIM, emb_dim=EMB_DIM, cond_ch=D_COND_CH):
        super().__init__()
        self.cond_enc = SunConditionEncoder(emb_dim=emb_dim)
        self.cond_to_map = nn.Sequential(
            nn.Linear(emb_dim, cond_ch),
            nn.LeakyReLU(0.2, True)
        )

        in_c = input_nc + cond_ch
        self.model = nn.Sequential(
            nn.Conv2d(in_c, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128, affine=False, eps=1e-5),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256, affine=False, eps=1e-5),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, 4, 1, 1),
            nn.InstanceNorm2d(512, affine=False, eps=1e-5),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x, y, vec):
        emb = self.cond_enc(vec)
        c = self.cond_to_map(emb).view(emb.size(0), -1, 1, 1)
        c = c.expand(-1, c.size(1), x.size(2), x.size(3))
        inp = torch.cat([x, y, c], dim=1)
        return self.model(inp)

# ================= 5. Dataset (256px mode + mask) =================
class SolarDataset(Dataset):
    def __init__(self, root_dir, json_file):
        self.train_dir = os.path.join(root_dir, "train")
        self.vector_map = {}
        json_path = os.path.join(root_dir, json_file)

        with open(json_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                vecs = entry.get('vectors', [])

                raw_vecs = np.zeros((9, 3), dtype=np.float32)
                mask = np.zeros((9,), dtype=np.float32)

                num_to_fill = min(len(vecs), 9)
                if num_to_fill > 0:
                    raw_vecs[:num_to_fill] = np.array(vecs[:num_to_fill], dtype=np.float32)
                    mask[:num_to_fill] = 1.0

                vec36 = np.concatenate([raw_vecs.flatten(), mask], axis=0)
                self.vector_map[str(entry['id'])] = torch.tensor(vec36, dtype=torch.float32)

        self.image_files = [f for f in os.listdir(self.train_dir) if f.endswith('.png')]

        # A and B use separate resize strategies:
        # A (height map) → NEAREST to preserve hard boundaries
        # B (radiation map) → BILINEAR for smooth gradients
        self.transform_A = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform_B = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_id = img_name.split('.')[0]

        img = cv2.imread(os.path.join(self.train_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        w_half = img.shape[1] // 2
        real_A, real_B = img[:, :w_half, :], img[:, w_half:, :]

        vec = self.vector_map.get(img_id, torch.zeros(VEC_DIM, dtype=torch.float32))

        return {
            "A": self.transform_A(real_A),
            "B": self.transform_B(real_B),
            "vec": vec,
            "id": img_id
        }

# ================= 6. Visual preview =================
def save_visuals(epoch, img_id, real_A, fake_B, real_B, img_dir):
    imgs = [("1_input", real_A), ("2_fake", fake_B), ("3_real", real_B)]
    for name, tensor in imgs:
        img = ((tensor[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(img_dir, f"epoch_{epoch:03d}_{img_id}_{name}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def update_html(web_dir, img_dir):
    html_path = os.path.join(web_dir, "index.html")
    with open(html_path, "w") as f:
        f.write("<html><body><h1>Solar 256px Training (Conditional FiLM + cD)</h1>"
                "<table border='1'><tr><th>ID</th><th>Input</th><th>AI Prediction</th><th>Truth</th></tr>")
        all_files = sorted([i for i in os.listdir(img_dir) if i.endswith(".png")], reverse=True)
        ids = []
        for file in all_files:
            gid = "_".join(file.split("_")[:3])
            if gid not in ids:
                ids.append(gid)
                f.write(f"<tr><td>{gid}</td>"
                        f"<td><img src='images/{gid}_1_input.png' width='256'></td>"
                        f"<td><img src='images/{gid}_2_fake.png' width='256'></td>"
                        f"<td><img src='images/{gid}_3_real.png' width='256'></td></tr>")
        f.write("</table></body></html>")

# ================= 7. Training loop =================
def train():
    os.makedirs(CHECKPOINTS_BASE, exist_ok=True)
    WEB_DIR = os.path.join(CHECKPOINTS_BASE, "web")
    IMG_DIR = os.path.join(WEB_DIR, "images")
    os.makedirs(IMG_DIR, exist_ok=True)

    dataset = SolarDataset(DATASET_ROOT, JSON_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    netG = SolarGenerator(vec_dim=VEC_DIM, emb_dim=EMB_DIM).to(DEVICE)
    netD = ConditionalDiscriminator(vec_dim=VEC_DIM, emb_dim=EMB_DIM, cond_ch=D_COND_CH).to(DEVICE)

    if torch.cuda.is_available():
        cudnn.benchmark = True

    latest_g_path = os.path.join(CHECKPOINTS_BASE, "latest_G.pth")
    latest_d_path = os.path.join(CHECKPOINTS_BASE, "latest_D.pth")

    if os.path.exists(latest_g_path):
        try:
            netG.load_state_dict(torch.load(latest_g_path, map_location=DEVICE))
            print("[OK] Loaded latest_G.pth")
        except Exception as e:
            print(f"[Warning] Failed to load latest_G.pth (architecture may have changed; consider training from scratch): {e}")

    if os.path.exists(latest_d_path):
        try:
            netD.load_state_dict(torch.load(latest_d_path, map_location=DEVICE))
            print("[OK] Loaded latest_D.pth")
        except Exception as e:
            print(f"[Warning] Failed to load latest_D.pth (architecture may have changed; consider training from scratch): {e}")

    optG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))

    criterionGAN = nn.MSELoss().to(DEVICE)
    criterionL1 = nn.L1Loss().to(DEVICE)

    fixed_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    fixed_batch = next(iter(fixed_loader))
    fixed_A = fixed_batch["A"].to(DEVICE)
    fixed_B = fixed_batch["B"].to(DEVICE)
    fixed_vec = fixed_batch["vec"].to(DEVICE)
    fixed_id = fixed_batch["id"][0]

    print(f"[Training] 256px run started (FiLM + Conditional D) | Batch: {BATCH_SIZE} | LR: {LR} | L1: {L1_LAMBDA} | vec_dim: {VEC_DIM}")

    for epoch in range(EPOCHS):
        for i, batch in enumerate(dataloader):
            real_A = batch['A'].to(DEVICE, non_blocking=True)
            real_B = batch['B'].to(DEVICE, non_blocking=True)
            vec = batch['vec'].to(DEVICE, non_blocking=True)

            # 1) Update D
            optD.zero_grad(set_to_none=True)

            with torch.no_grad():
                fake_B = netG(real_A, vec)

            pred_real = netD(real_A, real_B, vec)
            pred_fake = netD(real_A, fake_B.detach(), vec)

            loss_D_real = criterionGAN(pred_real, torch.ones_like(pred_real))
            loss_D_fake = criterionGAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5

            loss_D.backward()
            optD.step()

            # 2) Update G
            optG.zero_grad(set_to_none=True)

            fake_B = netG(real_A, vec)
            pred_fake_for_g = netD(real_A, fake_B, vec)

            loss_G_GAN = criterionGAN(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            loss_G_L1 = criterionL1(fake_B, real_B) * L1_LAMBDA
            loss_G = loss_G_GAN + loss_G_L1

            loss_G.backward()
            optG.step()

            if i % 20 == 0:
                print(f"E[{epoch:03d}] S[{i:04d}] LD:{loss_D.item():.4f} LG:{loss_G.item():.4f} (GAN:{loss_G_GAN.item():.4f} L1:{loss_G_L1.item():.4f})")

        # Save checkpoint and update visual preview
        if epoch % SAVE_EPOCH_FREQ == 0:
            netG.eval()
            with torch.no_grad():
                fixed_fake = netG(fixed_A, fixed_vec)
            save_visuals(epoch, fixed_id, fixed_A, fixed_fake, fixed_B, IMG_DIR)
            update_html(WEB_DIR, IMG_DIR)
            netG.train()

            torch.save(netG.state_dict(), latest_g_path)
            torch.save(netD.state_dict(), latest_d_path)

            save_g = os.path.join(CHECKPOINTS_BASE, f"epoch_{epoch:03d}_G.pth")
            save_d = os.path.join(CHECKPOINTS_BASE, f"epoch_{epoch:03d}_D.pth")
            torch.save(netG.state_dict(), save_g)
            torch.save(netD.state_dict(), save_d)

            print(f"[OK] Checkpoint saved: {save_g} | {save_d}")

if __name__ == "__main__":
    train()
