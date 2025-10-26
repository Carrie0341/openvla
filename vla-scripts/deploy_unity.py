"""
deploy_unity.py

为 OpenVLA 模型提供一个轻量级服务器实现，可以与 Unity 项目集成，用于控制相机运动。
这个脚本实现了一个 FastAPI 服务器，接收来自 Unity 的图像和指令，并返回相机控制动作。

依赖项:
    => 服务器 (在 GPU 上运行 OpenVLA 模型): `pip install uvicorn fastapi json-numpy`
    => 客户端 (Unity): 使用现有的 DataStreamer.cs 脚本

客户端 (Unity) 使用方式:
    1. 确保 DataStreamer.cs 已添加到您的 Unity 项目中
    2. 设置 serverBaseUrl 为 "http://127.0.0.1:8000"
    3. 设置 telemetryEndpoint 为 "/telemetry"
    4. 调用 Connect() 方法开始流式传输
"""

import os.path
import base64
import io

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Unity 数据模型 ===
class PoseData(BaseModel):
    """从 Unity 接收的相机位姿 (世界坐标系)"""
    px: float
    py: float
    pz: float
    rx: float
    ry: float
    rz: float

class TelemetryData(BaseModel):
    """从 Unity 接收的完整数据"""
    pose: PoseData
    fov: float
    width: int
    height: int
    cam_rgb_b64: Optional[str] = None
    cam_depth_b64: Optional[str] = None

class ResponsePose(BaseModel):
    """返回给 Unity 的控制位姿 (可选字段)"""
    px: Optional[float] = None
    py: Optional[float] = None
    pz: Optional[float] = None
    rx: Optional[float] = None
    ry: Optional[float] = None
    rz: Optional[float] = None

class ResponseData(BaseModel):
    """返回给 Unity 的完整控制指令"""
    apply: bool = False
    isReset: bool = False
    isDone: bool = False
    fov: Optional[float] = None
    pose: ResponsePose = None

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

def base64_to_image(base64_str: str) -> np.ndarray:
    """将 base64 编码的图像转换为 numpy 数组"""
    try:
        # 移除可能的 data URL 前缀
        if ',' in base64_str:
            base64_str = base64_str.split(',', 1)[1]
        
        # 解码 base64 字符串
        image_bytes = base64.b64decode(base64_str)
        
        # 将字节转换为 PIL 图像
        image = Image.open(io.BytesIO(image_bytes))
        
        # 转换为 numpy 数组
        return np.array(image)
    except Exception as e:
        logger.error(f"Error converting base64 to image: {e}")
        raise ValueError(f"Invalid base64 image: {e}")

def action_to_camera_control(action: np.ndarray, current_pose: PoseData) -> ResponsePose:
    """将 OpenVLA 的动作转换为相机控制指令"""
    # 这里需要根据您的 OpenVLA 模型输出的动作格式来实现
    # 假设 action 是一个包含 6 个值的数组：[dx, dy, dz, drx, dry, drz]
    # 即相对于当前位置的增量移动
    
    if len(action) >= 6:
        return ResponsePose(
            px=current_pose.px + action[0],
            py=current_pose.py + action[1],
            pz=current_pose.pz + action[2],
            rx=current_pose.rx + action[3],
            ry=current_pose.ry + action[4],
            rz=current_pose.rz + action[5]
        )
    else:
        # 如果动作维度不匹配，只返回当前位姿
        logger.warning(f"Action dimension mismatch: expected at least 6, got {len(action)}")
        return ResponsePose(
            px=current_pose.px,
            py=current_pose.py,
            pz=current_pose.pz,
            rx=current_pose.rx,
            ry=current_pose.ry,
            rz=current_pose.rz
        )

# === Server Interface ===
class OpenVLAServer:
    def __init__(self, openvla_path: Union[str, Path], instruction: str = "control the camera", attn_implementation: Optional[str] = "flash_attention_2") -> None:
        """
        一个简单的 OpenVLA 服务器，用于与 Unity 集成控制相机。
        
        Args:
            openvla_path: OpenVLA 模型的路径
            instruction: 用于指导模型的指令
            attn_implementation: 注意力实现方式
        """
        self.openvla_path = openvla_path
        self.instruction = instruction
        self.attn_implementation = attn_implementation
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
        # 初始化 FastAPI 应用
        self.app = FastAPI()
        
        # 注册端点
        self.app.post("/telemetry")(self.handle_telemetry)
        
        # 加载 VLA 模型
        logger.info(f"Loading OpenVLA model from {self.openvla_path}")
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        # 加载数据集统计信息（如果存在）
        if os.path.isdir(self.openvla_path):
            stats_path = Path(self.openvla_path) / "dataset_statistics.json"
            if os.path.exists(stats_path):
                logger.info(f"Loading dataset statistics from: {stats_path}")
                with open(stats_path, "r") as f:
                    self.vla.norm_stats = json.load(f)

    async def handle_telemetry(self, data: TelemetryData) -> ResponseData:
        """处理来自 Unity 的遥测数据并返回相机控制指令"""
        try:
            # 检查是否有图像数据
            if not data.cam_rgb_b64:
                logger.warning("No RGB image data received")
                return ResponseData(apply=False)
            
            # 将 base64 编码的图像转换为 numpy 数组
            image = base64_to_image(data.cam_rgb_b64)
            
            # 使用 OpenVLA 模型预测动作
            prompt = get_openvla_prompt(self.instruction, self.openvla_path)
            inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
            
            # 预测动作
            action = self.vla.predict_action(**inputs, unnorm_key=None, do_sample=False)
            
            # 将动作转换为相机控制指令
            camera_control = action_to_camera_control(action, data.pose)
            
            # 返回响应
            return ResponseData(
                apply=True,
                fov=data.fov,  # 保持当前 FOV
                pose=camera_control
            )
            
        except Exception as e:
            logger.error(f"Error processing telemetry: {e}")
            logger.error(traceback.format_exc())
            # 发生错误时返回一个不应用的响应
            return ResponseData(apply=False)

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """启动 FastAPI 服务器"""
        logger.info(f"Starting OpenVLA server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off
    openvla_path: Union[str, Path] = "openvla/openvla-7b"  # HF Hub 路径或本地模型目录
    instruction: str = "control the camera to follow the subject"  # 控制指令
    
    # 服务器配置
    host: str = "0.0.0.0"  # 主机 IP 地址
    port: int = 8000  # 主机端口
    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    """部署 OpenVLA 服务器"""
    logger.info(f"Starting OpenVLA Unity server with model: {cfg.openvla_path}")
    logger.info(f"Using instruction: '{cfg.instruction}'")
    
    server = OpenVLAServer(cfg.openvla_path, cfg.instruction)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()