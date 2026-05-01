import torch
import torch.nn as nn


# ================= 1. Sun Condition Encoder =================
class SunConditionEncoder(nn.Module):
    """
    Matches the checkpoint architecture:
    - Preserves the order of 9 hourly sun vectors
    - Input vec: (B, 36) = [9x3 vectors, 9 masks]
    - Output emb: (B, emb_dim)
    - Fuse input dim = 9*emb_dim + 9(mask) + 1(daylen)
    """
    def __init__(self, emb_dim=128):
        super().__init__()
        self.emb_dim = emb_dim

        self.per_vec = nn.Sequential(
            nn.Linear(3, emb_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(0.2, True),
        )

        fuse_in = 9 * emb_dim + 9 + 1
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, emb_dim),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, vec):
        # vec: (B,36)
        v = vec[:, :27].view(-1, 9, 3)          # (B,9,3)
        m = vec[:, 27:].view(-1, 9)             # (B,9)
        m01 = (m > 0.5).float()                 # (B,9)

        # (B,9,emb)
        feat = self.per_vec(v)

        # Preserve order: flatten to (B, 9*emb)
        feat_flat = feat.reshape(feat.size(0), -1)

        # daylen: (B,1)
        daylen = (m01.sum(dim=1, keepdim=True).clamp(min=1.0) / 9.0)

        # Concatenate: 9*emb + 9(mask) + 1(daylen)
        x = torch.cat([feat_flat, m01, daylen], dim=1)
        return self.fuse(x)



# ================= 2. FiLM Layer =================
class FiLMIN(nn.Module):
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


# ================= 3. U-Net Sub-modules =================
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


# ================= 4. Generator =================
class SolarGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, vec_dim=36, emb_dim=128):
        super().__init__()
        self.cond_enc = SunConditionEncoder(emb_dim=emb_dim)

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
        d2 = self.dec2(d1, emb);
        d2 = torch.cat([d2, e6], 1)
        d3 = self.dec3(d2, emb);
        d3 = torch.cat([d3, e5], 1)
        d4 = self.dec4(d3, emb);
        d4 = torch.cat([d4, e4], 1)
        d5 = self.dec5(d4, emb);
        d5 = torch.cat([d5, e3], 1)
        d6 = self.dec6(d5, emb);
        d6 = torch.cat([d6, e2], 1)
        d7 = self.dec7(d6, emb);
        d7 = torch.cat([d7, e1], 1)

        return self.last(d7)