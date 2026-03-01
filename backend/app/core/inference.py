import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import os
import sys
import datetime
from typing import List, Optional, Tuple, Dict, Any

# 导入网络定义
try:
    from app.models.networks import SolarGenerator  # 改用SolarGenerator
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
    from app.models.networks import SolarGenerator


class SolarPredictor:
    """
    负责：
    1) 接收高度图 + 太阳向量 (36维 = 27 vectors + 9 mask)
    2) pix2pix 推理输出 analysis 图（模型输入 256x256）
    3) 基于输入灰度语义生成 mask（背景255=无效）
    4) 颜色重映射：从训练时的 9 色 -> 前端新配色
    5) 使用前端传入的 legend palette（9色）在 Lab 空间做最近邻分类统计
    """

    # ✅ 统一尺寸：模型训练/推理输入
    MODEL_SIZE = 256

    # ✅ 返回给前端的尺寸：Rhino 输入是 512，这里默认也返回 512，避免前端自己放大导致采样变糊
    FRONTEND_SIZE = 512

    # ✅ 你如果希望前端直接拿 256，就把这个改成 False
    UPSCALE_TO_FRONTEND_SIZE = True

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[InstantSim] AI 引擎启动... 设备: {self.device}", flush=True)

        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "../models/city_sun_model.pth")

        self.model = self._load_model(model_path)

        # ✅ 关键：推理端预处理要和训练一致
        # - A 是高度图语义（离散/边界强），用 NEAREST 保边界
        # - Normalize 到 [-1,1]（和训练一致）
        self.transform = transforms.Compose([
            transforms.Resize((self.MODEL_SIZE, self.MODEL_SIZE), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # 太阳向量缓存（从前端接收）
        self.sun_vectors = None

        # --- 1. 训练时的原始配色 (严格匹配你的 9 小时训练图) ---
        # 对应标注: 0, 1, 2, 3, 4, 6, 7, 8, 9
        self.original_palette_hex = [
            "#0000ff", "#2e00d1", "#8500b3", "#bc0052", "#ff0000",
            "#ff4600", "#ff9100", "#ffc600", "#ffff00"
        ]
        self.original_palette_bgr = np.stack([self._hex_to_bgr_u8(c) for c in self.original_palette_hex], axis=0)
        self.original_palette_lab = cv2.cvtColor(
            self.original_palette_bgr.reshape(1, 9, 3),
            cv2.COLOR_BGR2LAB
        ).reshape(9, 3).astype(np.float32)

        # palette 由前端传入：9 色，对应 0..8 索引（新配色）
        self.palette_bgr_u8: Optional[np.ndarray] = None  # shape (9,3) uint8 BGR
        self.palette_lab_f32: Optional[np.ndarray] = None  # shape (9,3) float32 Lab

        # Debug 目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.debug_dir = os.path.join(current_dir, "../debug_outputs")
        os.makedirs(self.debug_dir, exist_ok=True)

    def _load_model(self, path: str):
        if not os.path.exists(path):
            print(f"模型文件未找到: {path}", flush=True)
            return None

        net = SolarGenerator(
            input_nc=3,
            output_nc=3,
            vec_dim=36
        )

        try:
            state_dict = torch.load(path, map_location=self.device)
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            net.load_state_dict(state_dict)
            net.to(self.device)
            net.eval()
            print("模型加载成功！", flush=True)
            return net
        except Exception as e:
            print(f"模型加载失败: {e}", flush=True)
            return None

    @staticmethod
    def _hex_to_bgr_u8(hex_color: str) -> np.ndarray:
        s = hex_color.strip()
        if s.startswith("#"):
            s = s[1:]
        if len(s) != 6:
            raise ValueError(f"Invalid hex color: {hex_color}")
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return np.array([b, g, r], dtype=np.uint8)

    def set_sun_vectors(self, vectors: List[float]) -> None:
        """
        设置太阳向量（从前端接收 - Ladybug 坐标系）
        vectors: 36 = 27 vectors + 9 mask
        """
        if len(vectors) != 36:
            raise ValueError(f"Sun vectors must have 36 elements (27 + 9 mask). Got: {len(vectors)}")

        vec36 = np.array(vectors, dtype=np.float32)
        vec36[27:] = (vec36[27:] > 0.5).astype(np.float32)

        self.sun_vectors = torch.tensor(vec36, dtype=torch.float32).to(self.device)
        print(f"[后端] Sun vectors updated. len=36", flush=True)

    def set_palette_from_hex_list(self, hex_colors: List[str]) -> None:
        if len(hex_colors) != 9:
            raise ValueError(f"Palette must have 9 colors (0..8). Got: {len(hex_colors)}")

        bgr = np.stack([self._hex_to_bgr_u8(c) for c in hex_colors], axis=0)  # (9,3)
        self.palette_bgr_u8 = bgr
        lab = cv2.cvtColor(bgr.reshape(1, 9, 3), cv2.COLOR_BGR2LAB).reshape(9, 3).astype(np.float32)
        self.palette_lab_f32 = lab
        print("[后端] Legend palette 已更新（来自前端）", flush=True)

    def _build_valid_mask_from_input(self, input_bgr: np.ndarray) -> np.ndarray:
        """
        基于你的输入语义生成有效区域 mask：
        - 背景=255（白） -> 无效
        - 地面=240 + 建筑=0..195 -> 有效
        """
        gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
        valid = (gray != 255)

        # 轻微腐蚀 1px：降低边缘抗锯齿/混色对统计的影响
        kernel = np.ones((3, 3), np.uint8)
        valid = cv2.erode(valid.astype(np.uint8), kernel, iterations=1).astype(bool)
        return valid

    def _remap_colors(self, output_bgr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        if self.palette_bgr_u8 is None:
            print("新配色未设置，跳过颜色重映射", flush=True)
            return output_bgr

        output_lab = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        remapped_bgr = output_bgr.copy()

        valid_pixels = output_lab[valid_mask]
        if valid_pixels.size == 0:
            return remapped_bgr

        diff = valid_pixels[:, None, :] - self.original_palette_lab[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        nearest_indices = np.argmin(dists, axis=1)

        new_colors = self.palette_bgr_u8[nearest_indices]
        remapped_bgr[valid_mask] = new_colors

        # 增强饱和度：只增强有效区域
        remapped_hsv = cv2.cvtColor(remapped_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        valid_hsv = remapped_hsv[valid_mask]
        valid_hsv[:, 1] = np.clip(valid_hsv[:, 1] * 1.4, 0, 255)
        remapped_hsv[valid_mask] = valid_hsv
        remapped_bgr = cv2.cvtColor(remapped_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        print(f"颜色重映射完成（饱和度增强），处理了 {len(valid_pixels)} 个像素", flush=True)
        return remapped_bgr

    def predict(self, image_bytes: bytes) -> Tuple[bytes, bytes, Dict[str, Any]]:
        print("\n[后端] 收到预测请求！正在处理...", flush=True)

        if self.model is None:
            return image_bytes, b"", {}

        if self.palette_lab_f32 is None:
            raise RuntimeError("Legend palette not set. Frontend must send LEGEND_PALETTE first.")

        if self.sun_vectors is None:
            raise RuntimeError("Sun vectors not set.")

        # 1) decode input
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_input_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_input_bgr is None:
            return image_bytes, b"", {}

        # ✅ 2) resize input to MODEL_SIZE for inference + mask (保持训练一致)
        img_input_bgr_256 = cv2.resize(
            img_input_bgr,
            (self.MODEL_SIZE, self.MODEL_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        # 3) preprocess
        img_pil = Image.fromarray(cv2.cvtColor(img_input_bgr_256, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        # 4) inference with sun vectors
        vec_batch = self.sun_vectors.unsqueeze(0)  # (1, 36)

        with torch.no_grad():
            output_tensor = self.model(input_tensor, vec_batch)

        # 5) postprocess -> BGR uint8 (MODEL_SIZE)
        output_img = output_tensor[0].cpu().permute(1, 2, 0).numpy()
        output_img = (output_img * 0.5 + 0.5) * 255.0
        output_bgr = np.clip(output_img, 0, 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_bgr, cv2.COLOR_RGB2BGR)

        # ✅ 6) build mask (基于 256 输入，保证尺寸对齐)
        valid_mask = self._build_valid_mask_from_input(img_input_bgr_256)
        mask_u8 = (valid_mask.astype(np.uint8) * 255)

        # 7) 颜色重映射
        output_bgr_remapped = self._remap_colors(output_bgr, valid_mask)

        # 8) stats（在 256 上算即可）
        stats = self._calculate_stats_discrete(output_bgr_remapped, valid_mask)

        # 9) debug save（保存 256 版本）
        self._debug_save_files(output_bgr_remapped, img_input_bgr_256, mask_u8, output_bgr)

        # ✅ 10) encode outputs
        # 默认返回给前端 512（NEAREST 放大，不糊边界）
        out_send = output_bgr_remapped
        mask_send = mask_u8

        if self.UPSCALE_TO_FRONTEND_SIZE:
            out_send = cv2.resize(
                output_bgr_remapped,
                (self.FRONTEND_SIZE, self.FRONTEND_SIZE),
                interpolation=cv2.INTER_NEAREST
            )
            mask_send = cv2.resize(
                mask_u8,
                (self.FRONTEND_SIZE, self.FRONTEND_SIZE),
                interpolation=cv2.INTER_NEAREST
            )

        ok1, analysis_png = cv2.imencode(".png", out_send)
        ok2, mask_png = cv2.imencode(".png", mask_send)
        if not ok1 or not ok2:
            return image_bytes, b"", stats

        return analysis_png.tobytes(), mask_png.tobytes(), stats

    def _calculate_stats_discrete(self, output_bgr: np.ndarray, valid_mask: np.ndarray) -> Dict[str, Any]:
        try:
            filtered = cv2.GaussianBlur(output_bgr, (3, 3), 0)
            filtered_lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB).astype(np.float32)

            valid_pixels = filtered_lab[valid_mask]
            if valid_pixels.size == 0:
                print("没有有效像素", flush=True)
                return {"avg": 0, "low_sun": 0, "high_sun": 0, "distribution": [], "debug_pixels": [],
                        "total_valid_pixels": 0}

            diff = valid_pixels[:, None, :] - self.palette_lab_f32[None, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            matched_indices = np.argmin(dists, axis=1)  # 0-8

            real_values = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9])
            matched_values = real_values[matched_indices]

            total_pixels = int(matched_indices.shape[0])
            counts = np.bincount(matched_indices, minlength=9)

            avg = float(np.mean(matched_values))
            low = float(np.sum(matched_values < 2) / total_pixels * 100)
            high = float(np.sum(matched_values > 6) / total_pixels * 100)
            dist = [round(x, 1) for x in (counts / total_pixels * 100).tolist()]

            labels = ["0h", "1h", "2h", "3h", "4h", "6h", "7h", "8h", "9h"]
            log_str = f"\n========== Solar Debug Report ==========\n"
            log_str += f"Time: {datetime.datetime.now()}\n"
            log_str += f"Total Valid Pixels: {total_pixels}\n"
            log_str += "-" * 40 + "\n"
            for i in range(9):
                log_str += f"{labels[i]:<4} : {int(counts[i]):>8} px ({(counts[i] / total_pixels * 100):>6.2f}%)\n"
            log_str += "-" * 40 + "\n"
            log_str += f"Avg: {avg:.2f} | Low(<2h): {low:.1f}% | High(>6h): {high:.1f}%\n"
            print(log_str, flush=True)

            return {
                "avg": round(avg, 1),
                "low_sun": round(low, 1),
                "high_sun": round(high, 1),
                "distribution": dist,
                "debug_pixels": counts.tolist(),
                "total_valid_pixels": total_pixels
            }

        except Exception as e:
            print(f"计算出错: {e}", flush=True)
            return {"avg": 0, "low_sun": 0, "high_sun": 0, "distribution": [], "debug_pixels": [],
                    "total_valid_pixels": 0}

    def _debug_save_files(self, output_bgr: np.ndarray, input_bgr: np.ndarray,
                          mask_u8: np.ndarray, original_output: np.ndarray = None) -> None:
        try:
            cv2.imwrite(os.path.join(self.debug_dir, "debug_ai_output_remapped.png"), output_bgr)
            cv2.imwrite(os.path.join(self.debug_dir, "debug_ai_input.png"), input_bgr)
            cv2.imwrite(os.path.join(self.debug_dir, "debug_mask.png"), mask_u8)
            if original_output is not None:
                cv2.imwrite(os.path.join(self.debug_dir, "debug_ai_output_original.png"), original_output)
            print(f"Debug 图片已保存到: {self.debug_dir}", flush=True)
        except Exception as e:
            print(f"Debug 保存失败: {e}", flush=True)
