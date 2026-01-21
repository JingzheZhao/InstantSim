import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import datetime
from typing import List, Optional, Tuple, Dict, Any

# 导入网络定义
try:
    from app.models.networks import UnetGenerator
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
    from app.models.networks import UnetGenerator


class SolarPredictor:
    """
    负责：
    1) pix2pix 推理输出 analysis 图（512x512）
    2) 基于输入灰度语义生成 mask（背景255=0，其它=255）
    3) 颜色重映射：从训练时的配色 -> 前端新配色
    4) 使用前端传入的 legend palette（11色）在 Lab 空间做 KNN 分类统计
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[InstantSim] AI 引擎启动... 设备: {self.device}", flush=True)

        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "../models/city_sun_model.pth")

        self.model = self._load_model(model_path)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # 训练时使用的原始配色（深蓝 -> 黄）
        self.original_palette_hex = [
            "#0000aa", "#2a0098", "#4b0082", "#a00040", "#d40020",
            "#ff0000", "#ff4500", "#ff8c00", "#ffa500", "#ffd700", "#ffff00"
        ]
        self.original_palette_bgr = np.stack([self._hex_to_bgr_u8(c) for c in self.original_palette_hex], axis=0)
        self.original_palette_lab = cv2.cvtColor(self.original_palette_bgr.reshape(1, 11, 3),
                                                 cv2.COLOR_BGR2LAB).reshape(11, 3).astype(np.float32)

        # palette 由前端传入：11色，对应 0..10h（新配色）
        self.palette_bgr_u8: Optional[np.ndarray] = None  # shape (11,3) uint8 BGR
        self.palette_lab_f32: Optional[np.ndarray] = None  # shape (11,3) float32 Lab

    def _load_model(self, path: str):
        if not os.path.exists(path):
            print(f"模型文件未找到: {path}", flush=True)
            return None

        net = UnetGenerator(
            input_nc=3,
            output_nc=3,
            num_downs=8,
            ngf=64,
            norm_layer=torch.nn.BatchNorm2d,
            use_dropout=True,
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
        """
        输入: "#RRGGBB" 或 "RRGGBB"
        输出: np.array([B,G,R], dtype=uint8)
        """
        s = hex_color.strip()
        if s.startswith("#"):
            s = s[1:]
        if len(s) != 6:
            raise ValueError(f"Invalid hex color: {hex_color}")
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return np.array([b, g, r], dtype=np.uint8)

    def set_palette_from_hex_list(self, hex_colors: List[str]) -> None:
        """
        前端传入 legend 的 11 个颜色（0..10h）- 新配色方案
        """
        if len(hex_colors) != 11:
            raise ValueError(f"Palette must have 11 colors (0..10h). Got: {len(hex_colors)}")

        bgr = np.stack([self._hex_to_bgr_u8(c) for c in hex_colors], axis=0)  # (11,3)
        self.palette_bgr_u8 = bgr

        lab = cv2.cvtColor(bgr.reshape(1, 11, 3), cv2.COLOR_BGR2LAB).reshape(11, 3).astype(np.float32)
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
        """
        颜色重映射：将模型输出的颜色从原始配色映射到新配色

        步骤：
        1. 在 Lab 空间找到每个像素在原始配色中的最近邻（得到 0-10 的索引）
        2. 用该索引在新配色中查找对应颜色
        3. 增强饱和度，让颜色更鲜艳
        4. 仅处理 valid_mask=True 的区域
        """
        if self.palette_bgr_u8 is None:
            print("新配色未设置，跳过颜色重映射", flush=True)
            return output_bgr

        # 转换到 Lab 空间
        output_lab = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # 创建输出图像（先复制原图）
        remapped_bgr = output_bgr.copy()

        # 获取有效区域的像素
        valid_pixels = output_lab[valid_mask]  # shape: (N, 3)

        if valid_pixels.size == 0:
            return remapped_bgr

        # 计算每个像素到原始配色的距离，找到最近的颜色索引
        diff = valid_pixels[:, None, :] - self.original_palette_lab[None, :, :]  # (N, 11, 3)
        dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, 11)
        nearest_indices = np.argmin(dists, axis=1)  # (N,) 值范围 0-10

        # 用新配色替换
        new_colors = self.palette_bgr_u8[nearest_indices]  # (N, 3)

        # 增强饱和度：转到HSV空间，提升S通道
        remapped_bgr[valid_mask] = new_colors

        # 转换到HSV进行饱和度增强
        remapped_hsv = cv2.cvtColor(remapped_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        # 只增强有效区域的饱和度
        valid_hsv = remapped_hsv[valid_mask]
        # 饱和度增强系数 1.4 (可调整：1.2-1.6)
        valid_hsv[:, 1] = np.clip(valid_hsv[:, 1] * 1.4, 0, 255)
        remapped_hsv[valid_mask] = valid_hsv

        # 转回BGR
        remapped_bgr = cv2.cvtColor(remapped_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        print(f"颜色重映射完成（饱和度增强），处理了 {len(valid_pixels)} 个像素", flush=True)
        return remapped_bgr

    def predict(self, image_bytes: bytes) -> Tuple[bytes, bytes, Dict[str, Any]]:
        """
        返回：
        - analysis_png_bytes (重映射后的颜色)
        - mask_png_bytes
        - stats dict
        """
        print("\n[后端] 收到预测请求！正在处理...", flush=True)

        if self.model is None:
            return image_bytes, b"", {}

        if self.palette_lab_f32 is None:
            raise RuntimeError("Legend palette not set. Frontend must send LEGEND_PALETTE first.")

        # 1) decode input
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_input_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_input_bgr is None:
            return image_bytes, b"", {}

        # 2) preprocess
        img_pil = Image.fromarray(cv2.cvtColor(img_input_bgr, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(img_pil)
        input_tensor = (input_tensor - 0.5) / 0.5
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        # 3) inference
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        # 4) postprocess -> BGR uint8
        output_img = output_tensor[0].cpu().permute(1, 2, 0).numpy()
        output_img = (output_img + 1) / 2.0 * 255.0
        output_bgr = np.clip(output_img, 0, 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_bgr, cv2.COLOR_RGB2BGR)

        # 5) build mask (512x512, single channel 0/255)
        valid_mask = self._build_valid_mask_from_input(img_input_bgr)
        mask_u8 = (valid_mask.astype(np.uint8) * 255)

        # 6) 颜色重映射（从原始配色 -> 新配色）
        output_bgr_remapped = self._remap_colors(output_bgr, valid_mask)

        # 7) stats（基于重映射后的颜色）
        stats = self._calculate_stats_discrete(output_bgr_remapped, valid_mask)

        # 8) debug save
        self._debug_save_files(output_bgr_remapped, img_input_bgr, mask_u8, output_bgr)

        # 9) encode outputs
        ok1, analysis_png = cv2.imencode(".png", output_bgr_remapped)
        ok2, mask_png = cv2.imencode(".png", mask_u8)
        if not ok1 or not ok2:
            return image_bytes, b"", stats

        return analysis_png.tobytes(), mask_png.tobytes(), stats

    def _calculate_stats_discrete(self, output_bgr: np.ndarray, valid_mask: np.ndarray) -> Dict[str, Any]:
        """
        - 仅统计 valid_mask=True 的像素
        - 在 Lab 空间做最近邻匹配（比 BGR 更稳）
        """
        try:
            # 轻度平滑
            filtered = cv2.GaussianBlur(output_bgr, (3, 3), 0)

            # 转 Lab
            filtered_lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB).astype(np.float32)

            valid_pixels = filtered_lab[valid_mask]
            if valid_pixels.size == 0:
                print("没有有效像素", flush=True)
                return {"avg": 0, "low_sun": 0, "high_sun": 0, "distribution": [], "debug_pixels": [],
                        "total_valid_pixels": 0}

            # 最近邻匹配到 11 个参考色（新配色）
            diff = valid_pixels[:, None, :] - self.palette_lab_f32[None, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            matched_hours = np.argmin(dists, axis=1)

            total_pixels = int(matched_hours.shape[0])
            counts = np.bincount(matched_hours, minlength=11)

            avg = float(np.mean(matched_hours))
            low = float(np.sum(matched_hours < 2) / total_pixels * 100)  # <2h
            high = float(np.sum(matched_hours > 6) / total_pixels * 100)  # >6h
            dist = [round(x, 1) for x in (counts / total_pixels * 100).tolist()]

            # 控制台日志
            labels = ["0h", "1h", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "10h"]
            log_str = f"\n========== Solar Debug Report ==========\n"
            log_str += f"Time: {datetime.datetime.now()}\n"
            log_str += f"Total Valid Pixels: {total_pixels}\n"
            log_str += "-" * 40 + "\n"
            for i in range(11):
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