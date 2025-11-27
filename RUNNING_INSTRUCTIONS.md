# Running Instructions / 运行说明

This document provides instructions on how to set up and run the Diffusion Text Visualizer application.
本文档提供设置和运行 Diffusion Text Visualizer 应用程序的说明。

---

## 1. Prerequisites / 前提条件

Before running the application, ensure you have the following installed:
在运行应用程序之前，请确保您已安装以下软件：

*   **Python 3.10+** (Recommended to use `uv` / `pip` for dependency management)
    **Python 3.10+**（建议使用 `uv` / `pip` 进行依赖管理）
*   **Node.js** (LTS version recommended) and **npm** (Node Package Manager)
    **Node.js**（推荐使用 LTS 版本）和 **npm**（Node Package Manager）

---

## 2. Backend Setup and Start / 后端设置与启动

The backend is a FastAPI application that handles the text diffusion model and provides a WebSocket endpoint for the frontend.
后端是一个 FastAPI 应用程序，它处理文本扩散模型并为前端提供 WebSocket 端点。

### Installation / 安装

From the project root directory (`tiny-diffusion/`), install the Python dependencies:
在项目根目录 (`tiny-diffusion/`) 中，安装 Python 依赖：

```bash
uv pip install uvicorn fastapi torch
# If you used 'uv sync' previously, these might already be installed.
# 如果您之前使用过 'uv sync'，这些可能已经安装。
```

### Start the Backend Server / 启动后端服务器

Once dependencies are installed, start the backend server from the project root:
安装完依赖后，从项目根目录启动后端服务器：

```bash
uv run backend/server.py
```

You should see output indicating that the FastAPI server is running, typically on `http://127.0.0.0:8000`.
您应该会看到输出，表明 FastAPI 服务器正在运行，通常在 `http://127.0.0.0:8000`。

---

## 3. Frontend Setup and Start / 前端设置与启动

The frontend is a React application built with Vite that visualizes the diffusion process.
前端是使用 Vite 构建的 React 应用程序，用于可视化扩散过程。

### Installation / 安装

Navigate to the `viz-frontend` directory:
导航到 `viz-frontend` 目录：

```bash
cd viz-frontend
```

Then, install the Node.js (npm) dependencies:
然后，安装 Node.js (npm) 依赖：

```bash
npm install
```
During the development process, the following packages were specifically installed: `three @types/three @react-three/fiber @react-three/drei lucide-react`. If `npm install` doesn't install them, you might need to run:
在开发过程中，专门安装了以下包：`three @types/three @react-three/fiber @react-three/drei lucide-react`。如果 `npm install` 没有安装它们，您可能需要运行：
```bash
npm install three @types/three @react-three/fiber @react-three/drei lucide-react
```

### Start the Frontend Development Server / 启动前端开发服务器

From the `viz-frontend` directory, start the development server:
在 `viz-frontend` 目录中，启动开发服务器：

```bash
npm run dev
```

The frontend application will typically open in your browser at `http://localhost:5173/` (or a similar port).
前端应用程序通常会在您的浏览器中打开，地址为 `http://localhost:5173/`（或类似的端口）。

---

## 4. Usage / 使用

With both the backend and frontend servers running, you can interact with the Diffusion Text Visualizer:
在后端和前端服务器都运行的情况下，您可以与 Diffusion Text Visualizer 进行交互：

1.  **Control Panel:** Use the left-hand control panel to adjust generation parameters like prompt, sequence length, number of steps, temperature, and decoding method.
    **控制面板：** 使用左侧控制面板调整生成参数，例如提示、序列长度、步数、温度和解码方法。
2.  **Start Generation:** Click the "Start Generation" button to begin the text diffusion process.
    **开始生成：** 单击“开始生成”按钮开始文本扩散过程。
3.  **Text Display:** Observe the text generation progress, with masked and decoded tokens highlighted.
    **文本显示：** 观察文本生成进度，其中屏蔽和解码的标记会高亮显示。
4.  **3D Visualization:** Explore the 3D projection of token embeddings, colored by their status (masked, decoded with high/low confidence). You can rotate and zoom using your mouse.
    **3D 可视化：** 探索标记嵌入的 3D 投影，并根据其状态（已屏蔽、高/低置信度解码）进行着色。您可以使用鼠标旋转和缩放。
5.  **Theme Toggle:** Use the "Switch to Light/Dark Mode" button in the header to change the application's theme.
    **主题切换：** 使用标题中的“切换到亮/暗模式”按钮更改应用程序的主题。

Enjoy visualizing the diffusion process!
享受可视化扩散过程！
