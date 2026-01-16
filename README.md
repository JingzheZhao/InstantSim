# 技术架构设计文档 (Technical Design Document)

**项目名称**：InstantSun —— 容器化实时光环境预测 SaaS 平台  
**版本**：1.0.0  
**状态**：开发中 (In Development)  
**作者**：[您的名字]  
**日期**：2026-01-08

---

## 1. 项目概述 (Project Overview)

**InstantSun** 是一个基于深度学习（Deep Learning）和微服务架构（Microservices）的实时建筑日照分析平台。

该项目旨在解决传统建筑物理模拟（如 Ladybug/OpenFOAM）计算耗时长、难以集成到实时设计流程中的痛点。通过构建端到端的 **MLOps 流水线**，项目将 Rhino/Grasshopper 中的几何数据实时传输至云端 **Docker 容器**，利用 **U-Net** 神经网络在毫秒级内推断全年日照辐射量，并通过 **Web Dashboard** 提供实时的决策支持、版本管理与数据可视化功能。

**核心价值**：
* **速度**：将分钟级的模拟时间压缩至毫秒级（<100ms）。
* **交互**：实现了从设计端（Rhino）到 Web 端的实时数字孪生（Digital Twin）交互。
* **工程化**：采用标准的 DevOps 流程，实现了服务的容器化封装与自动化部署。

---

## 2. 技术栈 (Tech Stack)

| 模块 | 技术选型 | 用途说明 |
| :--- | :--- | :--- |
| **编程语言** | **Python 3.9+** | 核心开发语言，覆盖后端、AI 及自动化脚本 |
| **深度学习** | **PyTorch** | 构建 U-Net 神经网络，进行模型训练与推理 |
| **后端框架** | **FastAPI** | 高性能异步 Web 框架，处理 HTTP 请求与 WebSocket |
| **通信协议** | **WebSockets** | 实现 Rhino 与 Web 端的实时双向通信 |
| **基础设施** | **Docker & Docker Compose** | 服务的容器化封装与编排 |
| **数据库** | **SQLite** / PostgreSQL | 存储历史版本快照、元数据与业务指标 |
| **前端** | **HTML5 / JavaScript** | 构建 Web 仪表盘与数据可视化 (ECharts) |
| **数据生成** | **Rhino / Grasshopper / Ladybug** | 生成用于训练的合成数据 (Synthetic Data) |

---

## 3. 系统架构 (System Architecture)

系统采用经典的 **B/S (Browser/Server) + Agent** 架构。

**数据流向逻辑**：
`Rhino (Input)` -> `API Gateway` -> `Docker Container (FastAPI + AI Model)` -> `Web Dashboard (Output)` & `Database (Storage)`

**各层级职责**：
1.  **数据采集层 (The Edge)**: Rhino/Grasshopper 作为客户端，负责几何生成、视口截图与数据发送。
2.  **计算服务层 (The Core)**: 封装在 Docker 中的 Python 后端，负责图像预处理、AI 推理与业务逻辑分发。
3.  **用户表现层 (The View)**: Web Dashboard，负责热力图渲染、图表展示与版本控制。

---

## 4. 核心功能模块 (Core Modules)

### 4.1. 自动化数据生成流水线 (Synthetic Data Pipeline)
构建自动化的 ETL (Extract, Transform, Load) 流程以生成高质量训练数据。

* **输入特征 (Input)**: 归一化的高度图 (Height Map)。
    * 背景：纯黑 (RGB 0,0,0) 代表地面。
    * 前景：灰度值 (0-255) 线性映射建筑高度。
* **输出标签 (Label)**: Ladybug 模拟生成的全年辐射量热力图。
* **自动化脚本**: 基于 Grasshopper Python 脚本实现批量几何生成、视口控制与截图保存。

### 4.2. 实时推理引擎 (Real-time Inference Engine)
基于 U-Net 架构的图像翻译模型。

* **模型架构**: Encoder-Decoder 结构，包含 Skip Connections 以保留边缘细节。
* **推理流程**:
    1.  接收 Base64 图像流。
    2.  预处理：Resize (512x512) -> ToTensor -> Normalize。
    3.  推理：CPU/GPU 快速前向传播。
    4.  后处理：Tensor 转伪彩色图像 (Color Map) -> Base64。

### 4.3. 实时联动系统 (Real-time Synchronization)
基于 WebSocket 的广播机制。

* **Rhino 端**: 监听模型几何变化事件，触发截图并上传。
* **Server 端**: 维护 WebSocket 连接池 (Connection Manager)，接收推理结果并广播给所有活跃的 Web 客户端。
* **Web 端**: 保持长连接，接收到二进制流后直接更新 DOM 元素，实现无刷新显示。

### 4.4. 版本控制与比对 (Version Control & Compare)
提供类似 Git 的设计版本管理功能。

* **快照保存 (Snapshot)**: 将当前推理结果（图片）与性能指标（JSON）持久化存储至数据库。
* **方案比对 (Compare)**: 支持分屏 (Split View) 查看不同历史版本的日照差异。
* **报告导出**: 前端渲染 Canvas 截图，生成高清设计报告。

---

## 5. 部署与 DevOps (Deployment & DevOps)

项目遵循 CI/CD 标准进行工程化实施。

### 5.1. 容器化 (Containerization)
* **Dockerfile**:
    * 基于 `python:3.9-slim` 基础镜像。
    * 分离依赖层与代码层，利用 Layer Cache 优化构建速度。
    * 集成 CPU 版 PyTorch 以减小镜像体积。

### 5.2. 服务编排 (Orchestration)
* **Docker Compose**:
    * 定义 `backend` 服务。
    * 挂载 `volume` 用于持久化存储生成的快照数据。
    * 配置资源限制 (CPU/Memory Limits) 模拟生产环境。

### 5.3. 持续集成 (Mock CI Pipeline)
* **GitHub Actions**:
    * **Lint**: 代码风格检查 (Flake8)。
    * **Test**: 单元测试 (Unit Tests) 验证 API 连通性与模型加载状态。
    * **Build**: 自动构建 Docker 镜像。

---

## 6. 开发路线图 (Roadmap)

### Phase 1: 数据工程 (Data Engineering)
- [ ] 完成 Grasshopper 自动化截图脚本开发。
- [ ] 生成训练集 (Train) 1000 组，验证集 (Val) 100 组。
- [ ] 数据清洗与预处理。

### Phase 2: 模型训练 (Model Training)
- [ ] 搭建 PyTorch 训练环境。
- [ ] 实施数据增强 (旋转、镜像) 以提升泛化能力。
- [ ] 训练 U-Net 模型并导出权重文件 (`.pth`)。

### Phase 3: 后端开发 (Backend Development)
- [ ] 初始化 FastAPI 项目结构。
- [ ] 开发 HTTP 推理接口与 WebSocket 广播服务。
- [ ] 编写 Dockerfile 并验证容器运行。

### Phase 4: 全栈联调 (Fullstack Integration)
- [ ] 开发 Web 前端 (Dashboard, Save, Compare UI)。
- [ ] 联调 Rhino -> Backend -> Web 数据链路。
- [ ] 系统集成测试与性能优化。


# InstantSun: 基于 Pix2Pix 的实时光环境预测系统架构

**版本**: 1.0 (MVP)  
**状态**: 开发中  
**核心逻辑**: Rhino 几何流 -> 高度图提取 -> AI 推理 (PyTorch) -> 后处理修正 -> Web 端纹理映射

---

# InstantSun: 基于 Pix2Pix 的实时光环境预测系统架构

**版本**: 1.0 (MVP)  
**状态**: 开发中  
**核心逻辑**: Rhino 几何流 -> 高度图提取 -> AI 推理 (PyTorch) -> 后处理修正 -> Web 端纹理映射

---

## 1. 系统总览 (High-Level Architecture)

整个系统分为三个解耦的模块：**客户端 (Client)**、**服务端 (Server)** 和 **前端展示 (Frontend)**。

```mermaid
graph LR
    A[Rhino/Grasshopper] -- 1. 发送高度图 (0-255 灰度) --> B(FastAPI Backend);
    B -- 2. PyTorch 推理 & OpenCV 修复 --> C{AI Model};
    C -- 3. 返回彩色日照图 --> D[Web Dashboard / Three.js];
    A -- 4. 同步 3D 几何体 (OBJ/GLTF) --> D;
    D -- 5. 纹理映射 (UV Mapping) --> User((设计师));
2. 详细工作流 (Detailed Workflow)第一阶段：数据采集 (The Input Layer)场景：设计师在 Rhino 中调整建筑方案。监听变化 (Trigger):后台脚本监听 Rhino 的 EndCommand 或 Idle 事件。一旦模型停止变动，触发计算。幽灵扫描 (Ghost Scan):不截图：不使用 ViewCapture，避免光照和分辨率干扰。几何计算：在内存中创建一个 256x256 的网格，从 Z 轴正无穷向 XY 平面发射射线 (Raycasting)。数值映射:Z = 0m (地面) -> 像素值 0 (黑)Z = 100m (最大高度) -> 像素值 255 (白)数据打包:生成 Bitmap 并转换为 Bytes 流。通过 HTTP POST 请求发送至后端 API (/predict)。第二阶段：AI 推理与修复 (The Processing Layer)场景：云端服务器接收请求。预处理 (Pre-processing):Resize: 强制缩放至 256x256。ToTensor: 转换为 PyTorch 张量。Normalize: 将 [0, 255] 映射到 [-1, 1] (Pix2Pix 标准)。模型推理 (Inference):加载模型: unet_256, norm='batch'。执行预测: fake_B = netG(input_A)。后处理修正 (Post-processing & Artifact Removal):针对 1.0 版本模型的工程化修复去噪: 使用 OpenCV 中值滤波 cv2.medianBlur(img, 3) 消除红/蓝噪点。裁边: 切除图像最外圈 2px (img[2:-2, 2:-2]) 消除边缘伪影。恢复: Resize 回 256x256。响应:将结果编码为 Base64 字符串或二进制 PNG 流返回。第三阶段：可视化映射 (The Visualization Layer)场景：网页端实时更新分析结果。双流同步:流 A (几何): 接收 Rhino 传来的简化版 3D 模型 (GLTF/OBJ)。流 B (纹理): 接收 AI 返回的彩色图片。平面投影 (Planar Projection):在 Three.js 中创建一个 Texture 对象加载 AI 图片。UV 映射逻辑: 强制使用 Planar Mapping (Top-down)。即纹理坐标 (u, v) 直接对应世界坐标 (x, y)。材质融合:将纹理贴在几何体表面 (MeshBasicMaterial 或 MeshStandardMaterial)。交互体验:用户在网页旋转视角时，彩色日照图就像“印”在模型上一样随模型转动，实现 3D 沉浸式查看。3. 技术规范 (Technical Specs)模型参数 (Model Configuration)架构: Pix2Pix (U-Net 256)输入通道: 3 (RGB 复制 3 次)输出通道: 3 (RGB)权重文件: city_sun_model.pth (Epoch 200 based)接口定义 (API Contract)POST /predictRequest:file: (Binary) 高度图图片文件Response:Content-Type: image/pngBody: 预测后的彩色日照图4. 常见问题处理 (Edge Case Handling)现象原因诊断解决方案 (工程侧)红点阵列 (Red Dot Matrix)训练用了 Dropout + 测试时 Batch Norm 不稳定后端增加 cv2.medianBlur (中值滤波)边缘蓝线 (Blue Border)卷积填充 (Padding) 导致的边缘效应后端强制 Crop 掉边缘 2px矮楼消失 (Missing Buildings)输入高度图对比度不足，或模型过拟合背景1. 检查 Rhino 射线高度映射范围2. 保持 lambda_L1 权重较高纹理错位Rhino 截图中心与 Three.js 模型中心不重合确保前后端统一使用 Bounding Box 中心对齐