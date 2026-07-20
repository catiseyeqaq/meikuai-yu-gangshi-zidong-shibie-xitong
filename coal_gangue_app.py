"""
煤矿履带煤炭与矿石智能识别监测系统 - Gradio Web 交互界面.
======================================================
基于 YOLO-GDL 目标检测模型（coal + gangue），实现完整的
"采集 → 识别 → 记录"演示流水线：

  感知阶段: 摄像头对准传送带 → GPU 实时推理（YOLO-GDL）
  识别阶段: 区分煤块 / 矿石 → 语音播报 + 写入日志 + 实时统计
  控制阶段: 下位机(STM32/ESP32) 仅负责履带启停与照明开关
https://yolo.threeflowercat.asia
http://localhost:7860/ - 本地Gradio Web 交互界面
通信协议: USB转TTL (UART), 115200bps, JSON 格式
  上位机→下位机: {"cmd":"belt","action":"start/stop"}
  上位机→下位机: {"cmd":"light","action":"on/off"}

技术手册参考：煤矿履带杂质智能识别与剔除系统技术手册.docx

运行方式：
    D:/miniconda3/envs/yolo/python.exe coal_gangue_app.py

注意：必须使用 yolo conda 环境运行，否则会出现 CUDA DLL 加载错误
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

# 日志配置：级别 INFO，实时刷新输出到终端（stdout）
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None
_log = logging.getLogger("coal_gangue")

import cv2
import gradio as gr
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 配置 matplotlib 中文字体支持
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import torch

from ultralytics import YOLO

# ============================================================================
# 串口通信模块（可选依赖）
# ============================================================================
try:
    import serial as _pyserial

    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARN] pyserial 未安装，串口通信使用模拟模式。")

import json as _json

# ============================================================================
# 语音播报模块（可选依赖）
# ============================================================================
try:
    import pyttsx3

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[WARN] pyttsx3 未安装，语音播报功能不可用。")


class SerialController:
    """串口通信控制器 —— 上位机 → 下位机 JSON 指令收发。.

    技术手册规定：
    - 物理连接: USB 转 TTL (UART)
    - 波特率: 115200
    - 数据格式: JSON 字符串

    未连接硬件时自动使用模拟模式（仅打印日志）。
    """

    def __init__(self) -> None:
        self._ser = None
        self._lock = threading.Lock()
        self.connected = False
        self.port = ""

    def connect(self, port: str = "COM3", baudrate: int = 115200) -> str:
        """连接串口。."""
        if not SERIAL_AVAILABLE:
            self.connected = False
            self.port = port
            return "[模拟模式] pyserial 未安装，串口指令将以日志形式输出"
        try:
            with self._lock:
                if self._ser and self._ser.is_open:
                    self._ser.close()
                self._ser = _pyserial.Serial(port, baudrate, timeout=1)
                self.connected = True
                self.port = port
            return f"串口已连接: {port} @ {baudrate}bps"
        except Exception as e:
            self.connected = False
            return f"串口连接失败: {e}"

    def disconnect(self) -> str:
        """断开串口。."""
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.close()
            self._ser = None
            self.connected = False
        return "串口已断开"

    def send_cmd(self, cmd_dict: dict) -> str:
        """发送 JSON 指令到下位机。."""
        json_str = _json.dumps(cmd_dict, ensure_ascii=False)
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.write((json_str + "\n").encode("utf-8"))
                print(f"[串口TX] {json_str}")
                return f"[已发送] {json_str}"
            else:
                print(f"[模拟TX] {json_str}")
                return f"[模拟] {json_str}"

    def ensure_connected(self, port: str = "COM3", baudrate: int = 115200) -> str:
        """Connect to CH340 if needed before sending hardware commands."""
        with self._lock:
            is_open = bool(self._ser and self._ser.is_open)
        if is_open:
            return f"串口已连接: {self.port or port}"
        return self.connect(port, baudrate)

    def send_belt(self, action: str) -> str:
        """发送履带控制指令 (start/stop)。."""
        return self.send_cmd({"cmd": "belt", "action": action})

    def send_light(self, action: str) -> str:
        """发送照明灯控制指令 (on/off)。."""
        return self.send_cmd({"cmd": "light", "action": action})


class VoiceAlert:
    """异步语音播报引擎 - 队列驱动 + 常驻线程。.

    每次播报创建新 pyttsx3 engine 实例，播完即销毁， 彻底避免 pyttsx3 多线程 runAndWait() 死锁问题。
    """

    def __init__(self) -> None:
        self.tts_available = TTS_AVAILABLE
        import queue

        self._queue: queue.Queue[str] = queue.Queue(maxsize=2)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self) -> None:
        """常驻线程：从队列取消息，新建 engine 播报后销毁。."""
        while True:
            text = self._queue.get()
            if not self.tts_available:
                print(f"[语音播报] {text}")
                continue
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 180)
                engine.setProperty("volume", 0.95)
                voices = engine.getProperty("voices")
                for v in voices:
                    if "chinese" in v.name.lower() or "zh" in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
                print(f"[语音完成] {text}")
            except Exception as e:
                print(f"[语音错误] {e}")

    def speak(self, text: str) -> None:
        """提交播报请求到队列（非阻塞）。队列满时丢弃，避免堆积。."""
        try:
            self._queue.put_nowait(text)
        except Exception:
            pass  # 队列满，静默丢弃


# ============================================================================
# 检测引擎
# ============================================================================
class CoalGangueDetector:
    """YOLOv8 煤矸石检测引擎，封装模型加载与推理。.

    Attributes:
        model (YOLO): 已加载的 YOLO 模型实例。
    """

    def __init__(self, model_path: str, imgsz: int = 1280) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.imgsz = imgsz
        self._infer_lock = threading.Lock()  # 防止多线程同时推理导致 GPU 死锁
        print(f"[INFO] 推理设备: {self.device} | 图像尺寸: {imgsz}")
        _log.info(f"推理设备: {self.device} | 图像尺寸: {imgsz}")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # ── 动态读取模型类别映射 ──
        self.class_names = self.model.names  # {0: 'coal', 1: 'gangue'} 或反过来
        print(f"[INFO] 模型类别: {self.class_names}")
        _log.info(f"模型类别: {self.class_names}")

        # 预热推理：使用训练分辨率消除首帧 GPU kernel 编译延迟
        _warmup_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        self.model(_warmup_img, conf=0.5, verbose=False, device=self.device, imgsz=imgsz)
        print(f"[INFO] 模型预热完成（{imgsz}x{imgsz}）")
        _log.info(f"模型预热完成（{imgsz}x{imgsz}）")

    def _normalize_class(self, cls_id: int) -> tuple[str, str]:
        """将模型原始类别统一映射为系统展示用的煤块/矿石。."""
        raw_name = str(self.class_names.get(cls_id, f"class_{cls_id}")).lower()
        if "coal" in raw_name or "煤" in raw_name:
            return "coal", "煤块"
        return "gangue", "矿石"

    def _parse_boxes(self, boxes, frame_area: int = 0) -> tuple[dict[str, int], list[dict], bool, float]:
        """解析 YOLO 检测框，生成统计、详情和矿石最高置信度。.

        Args:
            boxes: YOLO 检测框结果。
            frame_area: 原始帧总像素面积（宽×高），用于过滤过小检测框。 为 0 时不启用面积过滤（兼容图片检测）。
        """
        stats: dict[str, int] = {"coal": 0, "gangue": 0}
        detections: list[dict] = []
        has_gangue = False
        max_gangue_conf = 0.0

        if boxes is None or len(boxes) == 0:
            return stats, detections, has_gangue, max_gangue_conf

        # 面积阈值：检测框面积需 >= 帧面积的 0.5%，否则视为噪声
        min_box_area = frame_area * 0.005 if frame_area > 0 else 0

        for box in boxes:
            # ── 面积过滤：过滤过小碎框 ──
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_area = (x2 - x1) * (y2 - y1)
            if min_box_area > 0 and box_area < min_box_area:
                continue

            cls_id = int(box.cls[0])
            cls_name, cn_name = self._normalize_class(cls_id)
            conf_score = float(box.conf[0])
            stats[cls_name] += 1
            if cls_name == "gangue":
                has_gangue = True
                max_gangue_conf = max(max_gangue_conf, conf_score)
            detections.append(
                {
                    "class": cn_name,
                    "confidence": round(conf_score, 3),
                    "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
                }
            )

        return stats, detections, has_gangue, max_gangue_conf

    def predict_image(
        self, image: np.ndarray, conf: float = 0.5, iou: float = 0.45
    ) -> tuple[np.ndarray, dict, list[dict]]:
        """对单张图片执行检测。.

        Args:
            image: BGR 通道的 numpy 图像数组。
            conf: 置信度阈值。
            iou: IoU 阈值。

        Returns:
            (annotated, stats, detections):
            annotated: 标注后的图像。
            stats: {"coal": N, "gangue": M} 类别计数。
            detections: 每个检测框的详细信息列表。
        """
        with self._infer_lock:
            results = self.model(image, conf=conf, iou=iou, verbose=False, device=self.device, imgsz=self.imgsz)
        annotated = results[0].plot()
        stats, detections, _, _ = self._parse_boxes(results[0].boxes)
        _log.info(f"检测到 {len(detections)} 个目标: coal={stats['coal']}, gangue={stats['gangue']}")
        return annotated, stats, detections

    def predict_frame(
        self, frame: np.ndarray, conf: float = 0.5, iou: float = 0.45
    ) -> tuple[np.ndarray, dict, bool, float]:
        """对单帧执行检测（视频/摄像头流用）。.

        Args:
            frame: BGR 通道的 numpy 帧数组。
            conf: 置信度阈值。
            iou: IoU 阈值。

        Returns:
            (annotated, stats, has_gangue, max_gangue_conf): 标注帧、类别统计、是否有矿石、矿石最高置信度。
        """
        with self._infer_lock:
            results = self.model(frame, conf=conf, iou=iou, verbose=False, device=self.device, imgsz=self.imgsz)
        annotated = results[0].plot()
        h, w = frame.shape[:2]
        stats, _, has_gangue, max_gangue_conf = self._parse_boxes(results[0].boxes, frame_area=h * w)
        return annotated, stats, has_gangue, max_gangue_conf

    def try_predict_frame(
        self, frame: np.ndarray, conf: float = 0.5, iou: float = 0.45
    ) -> tuple[np.ndarray, dict, bool, float] | None:
        """非阻塞帧推理：推理锁被占用时立即返回 None，避免摄像头线程阻塞等待。.

        用于实时摄像头场景 —— 当图片/视频检测正在占用 GPU 时，
        摄像头线程不必等待，直接跳过本帧推理并显示缓存画面。
        """
        acquired = self._infer_lock.acquire(blocking=False)
        if not acquired:
            return None
        try:
            results = self.model(
                frame,
                conf=conf,
                iou=iou,
                verbose=False,
                device=self.device,
                imgsz=self.imgsz,
            )
        finally:
            self._infer_lock.release()
        annotated = results[0].plot()
        h, w = frame.shape[:2]
        stats, _, has_gangue, max_gangue_conf = self._parse_boxes(results[0].boxes, frame_area=h * w)
        return annotated, stats, has_gangue, max_gangue_conf


# ============================================================================
# 统计可视化
# ============================================================================
def _fig_to_numpy(fig: plt.Figure) -> np.ndarray:
    """将 matplotlib figure 转换为 numpy 数组（RGB）。."""
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 4)[:, :, :3]  # RGBA -> RGB
    return img


def create_pie_chart(stats: dict[str, int]) -> np.ndarray:
    """生成煤块/矿石占比饼图。.

    Args:
        stats: {"coal": N, "gangue": M}。

    Returns:
        RGB 格式的 numpy 图像数组。
    """
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    labels = ["煤块 (Coal)", "矿石 (Gangue)"]
    values = [stats.get("coal", 0), stats.get("gangue", 0)]
    colors = ["#27ae60", "#e74c3c"]

    if sum(values) == 0:
        values = [1, 1]
    explode = (0, 0.06) if values[1] > 0 else (0, 0)

    _wedges, _texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        explode=explode,
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(11)
    ax.set_title("检测结果占比", fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    result = _fig_to_numpy(fig)
    plt.close(fig)
    return result


def create_bar_chart(stats: dict[str, int]) -> np.ndarray:
    """生成各类别数量柱状图。.

    Args:
        stats: {"coal": N, "gangue": M}。

    Returns:
        RGB 格式的 numpy 图像数组。
    """
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    categories = ["煤块 (Coal)", "矿石 (Gangue)"]
    values = [stats.get("coal", 0), stats.get("gangue", 0)]
    colors = ["#27ae60", "#e74c3c"]

    bars = ax.bar(categories, values, color=colors, width=0.45, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(1, max(values) * 0.02),
            str(val),
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )
    ax.set_ylabel("数量", fontsize=10)
    ax.set_title("各类别检测数量", fontsize=12, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(max(values) * 1.25, 5))

    plt.tight_layout()
    result = _fig_to_numpy(fig)
    plt.close(fig)
    return result


# ============================================================================
# 全局状态
# ============================================================================
voice_alert = VoiceAlert()
serial_ctrl = SerialController()
detector: CoalGangueDetector | None = None
loaded_model_name = "YOLO-GDL"
loaded_model_path = ""
loaded_model_classes = ""
PUBLIC_GUEST_MODE = os.environ.get("PUBLIC_GUEST_MODE", "1").strip().lower() not in {"0", "false", "no", "off"}

# 全局日志缓冲区（用于跨 Tab 共享日志）
system_logs: list[str] = []
log_lock = threading.Lock()


def add_log(message: str) -> None:
    """线程安全地向系统日志追加一条记录。."""
    ts = datetime.now().strftime("%H:%M:%S")
    with log_lock:
        system_logs.append(f"[{ts}] {message}")
        if len(system_logs) > 200:
            system_logs.pop(0)


def get_logs() -> str:
    """获取系统日志字符串。."""
    with log_lock:
        return "\n".join(system_logs[-50:])


# ============================================================================
# 回调处理函数
# ============================================================================
def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """将 Gradio 输入图片规范成 RGB uint8 三通道，便于后续统一推理。."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif image.shape[2] != 3:
        image = image[:, :, :3]

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def format_detection_summary(stats: dict[str, int], title: str = "检测结果") -> str:
    """生成煤块/矿石检测结果 Markdown 表格。."""
    coal_cnt = stats.get("coal", 0)
    gangue_cnt = stats.get("gangue", 0)
    return (
        f"## {title}\n\n"
        f"| 类别 | 数量 |\n|------|------|\n"
        f"| 煤块 (Coal) | {coal_cnt} |\n"
        f"| 矿石 (Gangue) | {gangue_cnt} |\n"
        f"| **总计** | **{coal_cnt + gangue_cnt}** |"
    )


def normalize_stats_source(source: dict | list | None) -> dict[str, int]:
    """把检测详情列表或统计字典统一转换成 {"coal": N, "gangue": M}。."""
    stats = {"coal": 0, "gangue": 0}
    if not source:
        return stats

    if isinstance(source, dict):
        stats["coal"] = int(source.get("coal", 0))
        stats["gangue"] = int(source.get("gangue", 0))
        return stats

    for item in source:
        cls_name = str(item.get("class", ""))
        if "煤块" in cls_name:
            stats["coal"] += 1
        elif "矿石" in cls_name:
            stats["gangue"] += 1
    return stats


# ---------- 图像检测 ----------
def handle_image_detect(
    image: np.ndarray | None,
    conf: float,
    iou: float,
    enable_voice: bool,
) -> tuple:
    """处理图片上传检测。.

    Returns:
        (annotated_img, result_text, detections_json, stats)
    """
    if image is None:
        return None, "请先上传图片。", [], None

    global detector
    if detector is None:
        return image, "模型未加载。", [], None

    image = ensure_rgb_uint8(image)

    # ── 核心：Gradio 给的是 RGB，YOLO 内部假定 numpy 输入是 BGR ──
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        annotated, stats, detections = detector.predict_image(image_bgr, conf=conf, iou=iou)
        # YOLO .plot() 输出 BGR，转为 RGB 给 Gradio 显示
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    except Exception as e:
        add_log(f"图片检测异常: {e}")
        return image, f"## 检测异常\n\n{e}", [], None

    coal_cnt = stats.get("coal", 0)
    gangue_cnt = stats.get("gangue", 0)
    result_text = format_detection_summary(stats)

    if enable_voice and gangue_cnt > 0:
        voice_alert.speak(f"检测到{gangue_cnt}个矿石")

    add_log(f"图片检测: 煤块={coal_cnt}, 矿石={gangue_cnt}")
    return annotated, result_text, detections, stats


def handle_guest_image_detect(
    image: np.ndarray | None,
    conf: float,
    iou: float,
    history: list | None,
) -> tuple:
    """Per-browser image/camera detection. No shared logs, voice, or server camera."""
    if image is None:
        return None, "请先上传图片或拍摄一张照片。", [], None, list(history or []), list(history or [])

    global detector
    if detector is None:
        return image, "模型未加载。", [], None, list(history or []), list(history or [])

    image = ensure_rgb_uint8(image)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    try:
        annotated, stats, detections = detector.predict_image(image_bgr, conf=conf, iou=iou)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return image, f"## 检测异常\n\n{e}", [], None, list(history or []), list(history or [])

    result_text = format_detection_summary(stats)
    history = list(history or [])
    if stats:
        history.insert(
            0,
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "coal": int(stats.get("coal", 0)),
                "gangue": int(stats.get("gangue", 0)),
                "total": int(stats.get("coal", 0)) + int(stats.get("gangue", 0)),
            },
        )
        history = history[:30]
    return annotated, result_text, detections, stats, history, history


# ---------- 视频检测 ----------
def handle_video_detect(
    video_path: str | None,
    conf: float,
    iou: float,
    progress=gr.Progress(),
) -> tuple:
    """处理视频上传检测，逐帧推理并输出标注视频。.

    Returns:
        (output_video_path, result_text, stats)
    """
    if video_path is None:
        return None, "请先上传视频。", None

    global detector
    if detector is None:
        return None, "模型未加载。", None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "无法打开视频文件。", None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 输出临时文件
    suffix = os.path.splitext(video_path)[1] or ".mp4"
    out_fd, out_path = tempfile.mkstemp(suffix=suffix)
    os.close(out_fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    total_coal = 0
    total_gangue = 0
    processed_frames = 0

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        processed_frames += 1

        try:
            annotated, stats, _, _ = detector.predict_frame(frame, conf=conf, iou=iou)
        except Exception as e:
            annotated = frame
            stats = {"coal": 0, "gangue": 0}
            add_log(f"视频帧推理异常: {e}")
        out_writer.write(annotated)

        total_coal += stats.get("coal", 0)
        total_gangue += stats.get("gangue", 0)

        progress((frame_idx + 1) / total_frames, desc=f"处理中... {frame_idx + 1}/{total_frames}")

    cap.release()
    out_writer.release()

    if processed_frames == 0:
        return None, "视频无有效帧。", None

    result_text = (
        f"## 视频检测统计\n\n"
        f"| 指标 | 数值 |\n|------|------|\n"
        f"| 总帧数 | {processed_frames} |\n"
        f"| 累计煤块检出 | {total_coal} |\n"
        f"| 累计矿石检出 | {total_gangue} |\n"
        f"| 帧率 (FPS) | {fps:.1f} |"
    )

    add_log(f"视频检测完成: {processed_frames}帧, 煤块={total_coal}, 矿石={total_gangue}")
    return out_path, result_text, {"coal": total_coal, "gangue": total_gangue}


def handle_guest_video_detect(
    video_path: str | None,
    conf: float,
    iou: float,
    history: list | None,
    progress=gr.Progress(),
) -> tuple:
    """Per-browser video detection. History stays in the current Gradio session."""
    if video_path is None:
        history = list(history or [])
        return None, "请先上传视频。", None, history, history

    global detector
    if detector is None:
        history = list(history or [])
        return None, "模型未加载。", None, history, history

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        history = list(history or [])
        return None, "无法打开视频文件。", None, history, history

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    suffix = os.path.splitext(video_path)[1] or ".mp4"
    out_fd, out_path = tempfile.mkstemp(suffix=suffix)
    os.close(out_fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    total_coal = 0
    total_gangue = 0
    processed_frames = 0

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        processed_frames += 1

        try:
            annotated, stats, _, _ = detector.predict_frame(frame, conf=conf, iou=iou)
        except Exception:
            annotated = frame
            stats = {"coal": 0, "gangue": 0}
        out_writer.write(annotated)

        total_coal += stats.get("coal", 0)
        total_gangue += stats.get("gangue", 0)
        progress((frame_idx + 1) / total_frames, desc=f"处理中... {frame_idx + 1}/{total_frames}")

    cap.release()
    out_writer.release()

    if processed_frames == 0:
        history = list(history or [])
        return None, "视频无有效帧。", None, history, history

    stats = {"coal": total_coal, "gangue": total_gangue}
    result_text = (
        f"## 视频检测统计\n\n"
        f"| 指标 | 数值 |\n|------|------|\n"
        f"| 总帧数 | {processed_frames} |\n"
        f"| 累计煤块检出 | {total_coal} |\n"
        f"| 累计矿石检出 | {total_gangue} |\n"
        f"| 帧率 (FPS) | {fps:.1f} |"
    )
    history = list(history or [])
    history.insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "source": "视频检测",
            "coal": total_coal,
            "gangue": total_gangue,
            "total": total_coal + total_gangue,
            "frames": processed_frames,
        },
    )
    history = history[:30]
    return out_path, result_text, stats, history, history


def handle_guest_camera_detect(
    image: np.ndarray | None,
    conf: float,
    iou: float,
    history: list | None,
    snapshots: list | None,
) -> tuple:
    """Browser-camera detection with per-session history and screenshot gallery."""
    if image is None:
        history = list(history or [])
        snapshots = list(snapshots or [])
        return None, "请先打开本机摄像头。", [], None, history, history, snapshots, snapshots

    annotated, result_text, detections, stats, history, history_view = handle_guest_image_detect(
        image, conf, iou, history
    )
    snapshots = list(snapshots or [])
    if stats and int(stats.get("gangue", 0)) > 0 and annotated is not None:
        snapshots.insert(0, annotated)
        snapshots = snapshots[:8]
    return annotated, result_text, detections, stats, history, history_view, snapshots, snapshots


def handle_guest_camera_stream(
    image: np.ndarray | None,
    conf: float,
    iou: float,
    history: list | None,
    snapshots: list | None,
    stream_state: dict | None,
) -> tuple:
    """Realtime visitor webcam stream detection. Camera remains browser-local."""
    history = list(history or [])
    snapshots = list(snapshots or [])
    stream_state = dict(stream_state or {})
    if image is None:
        return None, "等待本机摄像头画面...", [], None, history, history, snapshots, snapshots, stream_state

    global detector
    if detector is None:
        return image, "模型未加载。", [], None, history, history, snapshots, snapshots, stream_state

    frame = ensure_rgb_uint8(image)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    t0 = time.time()
    try:
        result = detector.try_predict_frame(frame_bgr, conf=conf, iou=iou)
        if result is None:
            return (
                gr.update(),
                "推理占用中，已跳过当前帧，请稍候...",
                [],
                None,
                history,
                history,
                snapshots,
                snapshots,
                stream_state,
            )
        annotated_bgr, stats, has_gangue, max_gangue_conf = result
        annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return gr.update(), f"## 实时检测异常\n\n{e}", [], None, history, history, snapshots, snapshots, stream_state

    infer_ms = (time.time() - t0) * 1000
    coal_count = int(stats.get("coal", 0))
    gangue_count = int(stats.get("gangue", 0))
    has_coal = coal_count > 0
    has_gangue = gangue_count > 0

    stream_state["frame_count"] = int(stream_state.get("frame_count", 0)) + 1
    stream_state["coal_streak"] = int(stream_state.get("coal_streak", 0)) + 1 if has_coal else 0
    stream_state["gangue_streak"] = int(stream_state.get("gangue_streak", 0)) + 1 if has_gangue else 0
    confirm_frames = 3
    confirmed_coal = stream_state["coal_streak"] >= confirm_frames
    confirmed_gangue = stream_state["gangue_streak"] >= confirm_frames
    now = time.time()
    last_ts = float(stream_state.get("fps_ts", now))
    last_frame_count = int(stream_state.get("fps_frame_count", 0)) + 1
    elapsed = max(now - last_ts, 0.001)
    if elapsed >= 1.0:
        stream_state["fps"] = round(last_frame_count / elapsed, 1)
        stream_state["fps_ts"] = now
        stream_state["fps_frame_count"] = 0
    else:
        stream_state["fps_frame_count"] = last_frame_count
        stream_state.setdefault("fps_ts", last_ts)
    fps = float(stream_state.get("fps", 0.0))

    result_text = (
        "## 实时检测结果\n\n"
        f"| 类别 | 当前帧数量 |\n|------|------|\n"
        f"| 煤块 (Coal) | {coal_count} |\n"
        f"| 矿石 (Gangue) | {gangue_count} |\n"
        f"| 连续矿石帧 | {stream_state['gangue_streak']} / {confirm_frames} |\n"
        f"| **推理 / FPS** | **{infer_ms:.0f} ms / {fps:.1f}** |"
    )

    last_record_ts = float(stream_state.get("last_record_ts", 0.0))
    should_record = confirmed_gangue and (now - last_record_ts >= 0.5)
    if should_record:
        stream_state["last_record_ts"] = now
        history.insert(
            0,
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "ts": round(now, 3),
                "source": "实时摄像头",
                "coal": coal_count,
                "gangue": gangue_count,
                "total": coal_count + gangue_count,
                "max_gangue_conf": round(float(max_gangue_conf), 3),
            },
        )
        history = history[:30]
        snapshots.insert(0, annotated)
        snapshots = snapshots[:8]

    detections = [
        {
            "class": "煤块",
            "count": coal_count,
            "confirmed": confirmed_coal,
        },
        {
            "class": "矿石",
            "count": gangue_count,
            "max_confidence": round(float(max_gangue_conf), 3),
            "confirmed": confirmed_gangue,
        },
    ]
    alarm_stats = stats if should_record else {"coal": coal_count, "gangue": 0}
    return annotated, result_text, detections, alarm_stats, history, history, snapshots, snapshots, stream_state


def handle_guest_camera_stream_preview(
    image: np.ndarray | None,
    conf: float,
    iou: float,
) -> tuple:
    """Lightweight realtime preview for browser webcam.

    Keep the high-frequency stream focused on the annotated frame and text only. Updating JSON, galleries, and session
    state on every frame can block Gradio's browser renderer and make the right-side image appear frozen.
    """
    if image is None:
        return None, "等待本机摄像头画面..."

    global detector
    if detector is None:
        return image, "模型未加载。"

    frame = ensure_rgb_uint8(image)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    t0 = time.time()
    try:
        annotated_bgr, stats, _has_gangue, _max_gangue_conf = detector.predict_frame(frame_bgr, conf=conf, iou=iou)
    except Exception as e:
        return gr.update(), f"## 实时检测异常\n\n{e}"

    annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    infer_ms = (time.time() - t0) * 1000
    coal_count = int(stats.get("coal", 0))
    gangue_count = int(stats.get("gangue", 0))

    result_text = (
        "## 实时检测结果\n\n"
        f"| 类别 | 当前帧数量 |\n|------|------|\n"
        f"| 煤块 (Coal) | {coal_count} |\n"
        f"| 矿石 (Gangue) | {gangue_count} |\n"
        f"| **推理耗时** | **{infer_ms:.0f} ms** |"
    )
    return annotated, result_text


def refresh_guest_statistics(history: list | None) -> tuple:
    """Build charts from only the current browser session history."""
    history = list(history or [])
    stats = {"coal": 0, "gangue": 0}
    for item in history:
        stats["coal"] += int(item.get("coal", 0))
        stats["gangue"] += int(item.get("gangue", 0))

    pie = create_pie_chart(stats)
    bar = create_bar_chart(stats)
    if sum(stats.values()) == 0:
        summary = "暂无本浏览器检测数据"
    else:
        summary = (
            f"本浏览器会话共检测到 **{sum(stats.values())}** 个目标："
            f"煤块 **{stats['coal']}** 个，矿石 **{stats['gangue']}** 个"
        )
    return pie, bar, summary, history


def format_guest_logs(history: list | None) -> str:
    """Format current browser session history as a private log view."""
    history = list(history or [])
    if not history:
        return "本浏览器暂无检测日志"

    lines = []
    for item in history[:50]:
        source = item.get("source", "图片/摄像头检测")
        line = (
            f"[{item.get('time', '--:--:--')}] {source}: "
            f"煤块={int(item.get('coal', 0))}, "
            f"矿石={int(item.get('gangue', 0))}, "
            f"总计={int(item.get('total', 0))}"
        )
        if "frames" in item:
            line += f", 帧数={int(item.get('frames', 0))}"
        lines.append(line)
    return "\n".join(lines)


def handle_guest_audio_record(audio: tuple | None) -> str:
    """Handle visitor microphone input without writing to shared server logs."""
    if audio is None:
        return "未录制音频"
    sample_rate, data = audio
    duration = len(data) / sample_rate if sample_rate > 0 else 0
    return f"本浏览器录音完成：时长 {duration:.1f} 秒，采样率 {sample_rate} Hz"


def guest_voice_message(text: str) -> str:
    """Guest-side voice should be handled by browser APIs, not server pyttsx3."""
    text = text.strip() or "检测到矿石"
    return f"访客端语音应由浏览器播放：{text}"


# ---------- 摄像头实时流（服务端 OpenCV 采集 + 后台推理线程） ----------
# 使用服务端直接访问摄像头，避免 Gradio stream 的 HTTP 编解码开销。
_cam_thread: threading.Thread | None = None
_cam_stop_event = threading.Event()
_cam_lock = threading.Lock()
# 共享状态：最新检测结果帧 + 统计文本
_cam_latest_frame: np.ndarray | None = None
_cam_latest_stats: str = "等待摄像头启动..."

# 语音告警冷却
_last_gangue_alert_time: float = 0.0
# FPS
_cam_fps: float = 0.0
# 当前摄像头演示累计统计
_cam_total_stats: dict[str, int] = {"coal": 0, "gangue": 0}
# 最近检测截图缓存（最大 8 张，用于 Web 端画廊展示）
_cam_detection_snapshots: list[np.ndarray] = []
# 时序连续性计数器（连续N帧检出才确认，防止单帧幻觉）
_cam_gangue_streak: int = 0
_cam_coal_streak: int = 0
_CAM_CONFIRM_FRAMES: int = 3  # 需连续3帧检出才确认为有效检测


def _cam_worker(cam_index: int, conf: float, iou: float, enable_voice: bool) -> None:
    """后台线程：持续采集传送带画面并执行识别记录。.

    流程: OpenCV 持续读帧 → GPU 推理 → 统计煤块/矿石 → 记录日志与截图。
    """
    global _cam_latest_frame, _cam_latest_stats
    global _last_gangue_alert_time, _cam_fps, _cam_total_stats

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        with _cam_lock:
            _cam_latest_stats = "⚠ 无法打开摄像头"
        print(f"[CAM] 无法打开摄像头 index={cam_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"[CAM] 摄像头已打开 index={cam_index}")
    _log.info(f"摄像头已打开 index={cam_index}")

    frame_count = 0
    fps_ts = time.time()
    _last_annotated_rgb: np.ndarray | None = None  # 缓存上一帧标注画面
    previous_has_coal = False
    previous_has_gangue = False

    while not _cam_stop_event.is_set():
        # 丢弃摄像头缓冲区旧帧（部分驱动 BUFFERSIZE 设置不生效的兜底）
        cap.grab()
        ret, frame_bgr = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        if detector is None:
            time.sleep(0.05)
            continue

        t0 = time.time()
        try:
            result = detector.try_predict_frame(frame_bgr, conf=conf, iou=iou)
        except Exception as e:
            print(f"[CAM] 推理异常: {e}")
            time.sleep(0.05)
            continue

        if result is None:
            # 推理锁被占用（图片/视频检测中），显示缓存帧或原始帧
            fallback = _last_annotated_rgb
            if fallback is None:
                fallback = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            with _cam_lock:
                _cam_latest_frame = fallback
                _cam_latest_stats = f"推理占用中，请稍候...\nFPS: {_cam_fps:.1f}"
            time.sleep(0.01)
            continue

        annotated_bgr, stats, has_gangue, max_gangue_conf = result
        now = time.time()

        gangue_c = stats.get("gangue", 0)
        coal_c = stats.get("coal", 0)
        has_coal = coal_c > 0

        # ── 时序连续性过滤：连续 N 帧检出才确认，防止单帧幻觉 ──
        global _cam_gangue_streak, _cam_coal_streak
        if has_gangue:
            _cam_gangue_streak += 1
        else:
            _cam_gangue_streak = 0
        if has_coal:
            _cam_coal_streak += 1
        else:
            _cam_coal_streak = 0

        # 只有连续帧达标才视为「确认检出」
        confirmed_gangue = _cam_gangue_streak >= _CAM_CONFIRM_FRAMES
        confirmed_coal = _cam_coal_streak >= _CAM_CONFIRM_FRAMES

        coal_event = confirmed_coal and not previous_has_coal
        gangue_event = confirmed_gangue and not previous_has_gangue

        with _cam_lock:
            if coal_event:
                _cam_total_stats["coal"] += coal_c
            if gangue_event:
                _cam_total_stats["gangue"] += gangue_c
            total_coal = _cam_total_stats["coal"]
            total_gangue = _cam_total_stats["gangue"]

        # ===== 识别记录阶段：矿石告警与日志 =====
        if gangue_event and (now - _last_gangue_alert_time > 2.0):
            if enable_voice:
                voice_alert.speak(f"检测到{gangue_c}个矿石")
            _last_gangue_alert_time = now
            add_log(f"矿石检测 -> {gangue_c}个 (最高置信度 {max_gangue_conf:.2f})")
        if coal_event:
            add_log(f"煤块检测 -> {coal_c}个")

        previous_has_coal = confirmed_coal
        previous_has_gangue = confirmed_gangue

        # FPS
        frame_count += 1
        elapsed = now - fps_ts
        if elapsed >= 1.0:
            _cam_fps = frame_count / elapsed
            frame_count = 0
            fps_ts = now

        infer_ms = (now - t0) * 1000

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        _last_annotated_rgb = annotated_rgb  # 缓存标注帧供推理占用时显示
        # 检测截图画廊：仅保存矿石检测帧（限 8 张）
        stats_text = (
            f"当前帧: 煤块 {coal_c} | 矿石 {gangue_c}\n"
            f"经过记录: 煤块 {total_coal} | 矿石 {total_gangue}\n"
            f"推理: {infer_ms:.0f}ms | FPS: {_cam_fps:.1f}"
        )

        with _cam_lock:
            if gangue_event and len(_cam_detection_snapshots) < 8:
                _cam_detection_snapshots.append(annotated_rgb.copy())
            _cam_latest_frame = annotated_rgb
            _cam_latest_stats = stats_text

    cap.release()
    print("[CAM] 摄像头已关闭")
    _log.info("摄像头已关闭")


def start_camera(cam_index: int, conf: float, iou: float, enable_voice: bool) -> str:
    """启动摄像头后台采集线程（采集→识别→记录流水线）。."""
    global _cam_thread, _cam_latest_frame, _cam_latest_stats, _cam_total_stats, _cam_detection_snapshots
    global _cam_gangue_streak, _cam_coal_streak
    stop_camera()  # 先停旧的

    _cam_stop_event.clear()
    with _cam_lock:
        _cam_total_stats = {"coal": 0, "gangue": 0}
        _cam_detection_snapshots = []
        _cam_latest_frame = None
        _cam_latest_stats = "摄像头启动中..."
    _cam_gangue_streak = 0
    _cam_coal_streak = 0

    _cam_thread = threading.Thread(
        target=_cam_worker,
        args=(cam_index, conf, iou, enable_voice),
        daemon=True,
    )
    _cam_thread.start()
    add_log(f"摄像头启动: index={cam_index}, 模式=识别记录")
    return "摄像头已启动（识别记录模式）"


def stop_camera() -> str:
    """停止摄像头后台采集线程。."""
    global _cam_thread
    if _cam_thread is not None and _cam_thread.is_alive():
        _cam_stop_event.set()
        _cam_thread.join(timeout=3)
        _cam_thread = None
        add_log("摄像头已停止")
    with _cam_lock:
        pass  # keep last frame
    return "摄像头已停止"


def poll_camera() -> tuple[np.ndarray | None, str, list[np.ndarray]]:
    """Gradio 定时器回调：拉取最新检测帧、统计和检测截图画廊。."""
    with _cam_lock:
        snapshots = list(_cam_detection_snapshots)
        return _cam_latest_frame, _cam_latest_stats, snapshots


def get_camera_stats() -> dict[str, int]:
    """读取摄像头本次演示的经过记录统计。."""
    with _cam_lock:
        return dict(_cam_total_stats)


# ---------- 统计刷新 ----------
def refresh_statistics(stats_source: dict | list | None) -> tuple:
    """根据检测详情列表刷新统计图表。.

    Args:
        stats_source: 图片检测详情列表，或摄像头累计统计字典。

    Returns:
        (pie_chart_image, bar_chart_image, summary_text)
    """
    stats = normalize_stats_source(stats_source)
    if sum(stats.values()) == 0:
        blank_pie = create_pie_chart({"coal": 0, "gangue": 0})
        blank_bar = create_bar_chart({"coal": 0, "gangue": 0})
        return blank_pie, blank_bar, "暂无检测数据"

    pie = create_pie_chart(stats)
    bar = create_bar_chart(stats)
    summary = f"共检测到 **{sum(stats.values())}** 个目标：煤块 **{stats['coal']}** 个，矿石 **{stats['gangue']}** 个"
    return pie, bar, summary


# ---------- 设备控制 ----------
def ensure_default_serial() -> str:
    """Ensure local CH340 is connected on COM3 before hardware control."""
    return serial_ctrl.ensure_connected("COM3", 115200)


def handle_belt_control(action: str) -> str:
    """履带启停控制 —— 通过串口发送 JSON 指令。."""
    connect_msg = ensure_default_serial()
    result = serial_ctrl.send_belt(action)
    msg = f"{connect_msg}；履带{'启动' if action == 'start' else '停止'} -> {result}"
    add_log(f"设备控制 -> {msg}")
    return msg


def handle_light_control(action: str) -> str:
    """照明灯控制 —— 通过串口发送 JSON 指令。."""
    connect_msg = ensure_default_serial()
    result = serial_ctrl.send_light(action)
    msg = f"{connect_msg}；照明灯{'开启' if action == 'on' else '关闭'} -> {result}"
    add_log(f"设备控制 -> {msg}")
    return msg


def handle_emergency_stop() -> str:
    """应急停止：停止履带并关闭照明。."""
    add_log("!!! 应急停止触发 !!!")
    connect_msg = ensure_default_serial()
    belt_result = serial_ctrl.send_belt("stop")
    light_result = serial_ctrl.send_light("off")
    add_log(f"设备控制 -> 履带停止 -> {belt_result}")
    add_log(f"设备控制 -> 照明关闭 -> {light_result}")
    voice_alert.speak("紧急停止")
    return f"应急停止已触发：{connect_msg}；履带停止 {belt_result}；照明关闭 {light_result}"


def handle_voice_test(text: str) -> str:
    """语音播报测试。."""
    if not text.strip():
        text = "煤矿履带煤炭与矿石识别系统测试"
    voice_alert.speak(text)
    msg = f"语音播报已触发：{text}"
    add_log(msg)
    return msg


def handle_serial_connect(port: str, baudrate: int) -> str:
    """连接串口。."""
    result = serial_ctrl.connect(port, int(baudrate))
    add_log(f"串口 -> {result}")
    return result


def handle_serial_disconnect() -> str:
    """断开串口。."""
    result = serial_ctrl.disconnect()
    add_log(f"串口 -> {result}")
    return result


# ---------- 音频录制播放 ----------
def handle_audio_record(audio: tuple | None) -> str:
    """处理麦克风录音输入。."""
    if audio is None:
        return "未录制音频"
    sample_rate, data = audio
    duration = len(data) / sample_rate if sample_rate > 0 else 0
    msg = f"录音完成：时长 {duration:.1f} 秒，采样率 {sample_rate} Hz"
    add_log(msg)
    return msg


# ============================================================================
# 主应用构建
# ============================================================================
# 自定义 CSS 主题（模块级）
CUSTOM_CSS = """
.main-title { text-align: center; font-size: 2.2em; font-weight: bold; 
              background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
              color: white; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
.sub-title { text-align: center; font-size: 1.1em; color: #7f8c8d; margin-bottom: 20px; }
.detect-box { border: 2px solid #2c3e50; border-radius: 8px; padding: 10px; background: #f8f9fa; }
.stat-box { border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: white; }
.alert-text { color: #e74c3c; font-weight: bold; }
.safe-text { color: #27ae60; font-weight: bold; }
.log-box { font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 0.85em; 
           background: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 6px; 
           max-height: 200px; overflow-y: auto; white-space: pre-wrap; }
.client-status { display: flex; align-items: center; gap: 8px; font-weight: 600; }
.status-dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
.status-on { background: #16a34a; box-shadow: 0 0 0 4px rgba(22, 163, 74, 0.15); }
.status-off { background: #9ca3af; box-shadow: 0 0 0 4px rgba(156, 163, 175, 0.15); }
.status-warn { background: #f59e0b; box-shadow: 0 0 0 4px rgba(245, 158, 11, 0.15); }
"""


GUEST_SERIAL_OFF_HTML = (
    '<div class="client-status"><span class="status-dot status-off"></span><span>未连接本机 CH340</span></div>'
)
GUEST_SERIAL_ON_HTML = (
    '<div class="client-status"><span class="status-dot status-on"></span><span>已连接本机 CH340</span></div>'
)
GUEST_SERIAL_WARN_HTML = (
    '<div class="client-status"><span class="status-dot status-warn"></span><span>需要 HTTPS / 浏览器授权</span></div>'
)


PUBLIC_GUEST_ALARM_JS = """
(stats, enabled) => {
  if (!enabled || !stats) {
    return;
  }
  const gangue = Number(stats.gangue || 0);
  if (gangue <= 0 || !("speechSynthesis" in window)) {
    return;
  }
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(`检测到${gangue}个矿石`);
  utter.lang = "zh-CN";
  utter.rate = 1.0;
  utter.volume = 1.0;
  window.speechSynthesis.speak(utter);
}
"""


GUEST_SERIAL_CONNECT_JS = f"""
async () => {{
  const secure = window.isSecureContext || ["localhost", "127.0.0.1"].includes(location.hostname);
  if (!secure) {{
    return [{_json.dumps(GUEST_SERIAL_WARN_HTML)}, "浏览器串口需要 HTTPS 域名，公网请使用 https://yolo.cat.com 后再连接。"];
  }}
  if (!("serial" in navigator)) {{
    return [{_json.dumps(GUEST_SERIAL_WARN_HTML)}, "当前浏览器不支持 Web Serial，请使用 Chrome 或 Edge。"];
  }}
  try {{
    if (window.yoloGdlSerialPort && window.yoloGdlSerialPort.readable) {{
      return [{_json.dumps(GUEST_SERIAL_ON_HTML)}, "本机 CH340 已连接。"];
    }}
    const port = await navigator.serial.requestPort();
    await port.open({{ baudRate: 115200 }});
    window.yoloGdlSerialPort = port;
    window.yoloGdlSerialRxBuffer = "";
    window.yoloGdlSerialReading = true;
    window.yoloGdlSerialReadTask = (async () => {{
      const decoder = new TextDecoder();
      while (window.yoloGdlSerialReading && window.yoloGdlSerialPort && window.yoloGdlSerialPort.readable) {{
        const reader = window.yoloGdlSerialPort.readable.getReader();
        window.yoloGdlSerialReader = reader;
        try {{
          while (window.yoloGdlSerialReading) {{
            const {{ value, done }} = await reader.read();
            if (done) {{
              break;
            }}
            if (value) {{
              window.yoloGdlSerialRxBuffer += decoder.decode(value, {{ stream: true }});
              window.yoloGdlSerialRxBuffer = window.yoloGdlSerialRxBuffer.slice(-4000);
            }}
          }}
        }} catch (readErr) {{
          break;
        }} finally {{
          try {{
            reader.releaseLock();
          }} catch (releaseErr) {{}}
          if (window.yoloGdlSerialReader === reader) {{
            window.yoloGdlSerialReader = null;
          }}
        }}
      }}
    }})();
    return [{_json.dumps(GUEST_SERIAL_ON_HTML)}, "本机 CH340 已连接，波特率 115200。"];
  }} catch (err) {{
    return [{_json.dumps(GUEST_SERIAL_WARN_HTML)}, "连接失败或已取消授权：" + err.message];
  }}
}}
"""


GUEST_SERIAL_DISCONNECT_JS = f"""
async () => {{
  try {{
    if (window.yoloGdlSerialPort) {{
      window.yoloGdlSerialReading = false;
      if (window.yoloGdlSerialReader) {{
        try {{
          await window.yoloGdlSerialReader.cancel();
        }} catch (cancelErr) {{}}
      }}
      await window.yoloGdlSerialPort.close();
      window.yoloGdlSerialPort = null;
    }}
    window.yoloGdlSerialReader = null;
    window.yoloGdlSerialRxBuffer = "";
    return [{_json.dumps(GUEST_SERIAL_OFF_HTML)}, "已断开本机 CH340。"];
  }} catch (err) {{
    window.yoloGdlSerialPort = null;
    window.yoloGdlSerialReader = null;
    window.yoloGdlSerialReading = false;
    return [{_json.dumps(GUEST_SERIAL_OFF_HTML)}, "断开时出现异常：" + err.message];
  }}
}}
"""


GUEST_SERIAL_SEND_CUSTOM_JS = f"""
async (payloadText) => {{
  if (!window.yoloGdlSerialPort || !window.yoloGdlSerialPort.writable) {{
    return [{_json.dumps(GUEST_SERIAL_WARN_HTML)}, "请先连接访问者本机 CH340。"];
  }}
  let payload;
  try {{
    payload = JSON.parse(payloadText || "{{}}");
  }} catch (err) {{
    return [{_json.dumps(GUEST_SERIAL_WARN_HTML)}, "JSON 格式错误：" + err.message];
  }}
  const encoder = new TextEncoder();
  let writer;
  const startLen = (window.yoloGdlSerialRxBuffer || "").length;
  try {{
    writer = window.yoloGdlSerialPort.writable.getWriter();
    await writer.write(encoder.encode(JSON.stringify(payload) + "\\n"));
  }} catch (err) {{
    return [{_json.dumps(GUEST_SERIAL_WARN_HTML)}, "测试包发送失败：" + err.message];
  }} finally {{
    if (writer) {{
      writer.releaseLock();
    }}
  }}
  await new Promise((resolve) => setTimeout(resolve, 800));
  const received = (window.yoloGdlSerialRxBuffer || "").slice(startLen).trim();
  if (received) {{
    return [{_json.dumps(GUEST_SERIAL_ON_HTML)}, "测试包已发送，并收到回包：" + received];
  }}
  return [{_json.dumps(GUEST_SERIAL_ON_HTML)}, "测试包已发送到访问者本机 CH340，800ms 内未收到回包。"];
}}
"""


def make_guest_serial_send_js(payloads: list[dict], success_message: str, speak_text: str = "") -> str:
    """Generate browser-side Web Serial JS. It never calls Python serial APIs."""
    payload_json = _json.dumps(payloads, ensure_ascii=False)
    on_html = _json.dumps(GUEST_SERIAL_ON_HTML)
    warn_html = _json.dumps(GUEST_SERIAL_WARN_HTML)
    success_json = _json.dumps(success_message, ensure_ascii=False)
    speak_json = _json.dumps(speak_text, ensure_ascii=False)
    return f"""
async () => {{
  if (!window.yoloGdlSerialPort || !window.yoloGdlSerialPort.writable) {{
    return [{warn_html}, "请先点击“连接本机 CH340”，并在浏览器授权自己的串口。"];
  }}
  const payloads = {payload_json};
  const encoder = new TextEncoder();
  let writer;
  try {{
    writer = window.yoloGdlSerialPort.writable.getWriter();
    for (const payload of payloads) {{
      await writer.write(encoder.encode(JSON.stringify(payload) + "\\n"));
    }}
  }} catch (err) {{
    return [{warn_html}, "发送失败：" + err.message];
  }} finally {{
    if (writer) {{
      writer.releaseLock();
    }}
  }}
  const speakText = {speak_json};
  if (speakText && "speechSynthesis" in window) {{
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(speakText);
    utter.lang = "zh-CN";
    window.speechSynthesis.speak(utter);
  }}
  return [{on_html}, {success_json}];
}}
"""


GUEST_VOICE_TEST_JS = """
(text) => {
  const msg = (text || "检测到矿石").trim() || "检测到矿石";
  if (!("speechSynthesis" in window)) {
    return "当前浏览器不支持本机语音播放。";
  }
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(msg);
  utter.lang = "zh-CN";
  utter.rate = 1.0;
  utter.volume = 1.0;
  window.speechSynthesis.speak(utter);
  return "本机语音已播放：" + msg;
}
"""


GUEST_BROWSER_ENV_JS = """
() => {
  const secure = window.isSecureContext || ["localhost", "127.0.0.1"].includes(location.hostname);
  const camera = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  const serial = !!navigator.serial;
  const lines = [
    `安全上下文: ${secure ? "是" : "否"}`,
    `摄像头 API: ${camera ? "可用" : "不可用"}`,
    `Web Serial: ${serial ? "可用" : "不可用"}`,
    `当前地址: ${location.href}`,
  ];
  if (!secure) {
    lines.push("公网 HTTP 下浏览器会拦截摄像头和串口，请使用 HTTPS 域名访问。");
  }
  return lines.join("\\n");
}
"""


def build_app() -> gr.Blocks:
    """构建 Gradio Blocks 应用。.

    Returns:
        配置完成的 gr.Blocks 实例。
    """
    model_display_path = loaded_model_path or "模型尚未加载"
    model_display_classes = loaded_model_classes or "coal / gangue"

    with gr.Blocks(
        title="煤矿履带煤炭与矿石智能识别监测系统",
    ) as demo:
        # ============ 页头 ============
        gr.HTML('<div class="main-title">煤矿履带煤炭与矿石智能识别监测系统</div>')
        gr.HTML(
            f'<div class="sub-title">基于 {loaded_model_name} | 煤块(Coal) / 矿石(Gangue) 检测 | Gradio Web 控制台</div>'
        )
        gr.Markdown(
            f"**当前搭载模型**: `{loaded_model_name}`  \n"
            f"**当前权重路径**: `{model_display_path}`  \n"
            f"**检测类别**: `{model_display_classes}`"
        )

        # ============ 全局状态 ============
        detection_state = gr.State(None)  # 存储最近一次图片检测统计
        guest_history_state = gr.State([])
        gr.State([])
        gr.State({})

        # ============ 主 Tab 布局 ============
        with gr.Tabs():
            # ------------------------------------------------------------------
            # Tab 1: 图像检测
            # ------------------------------------------------------------------
            with gr.TabItem("图像检测", id="tab_image"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        gr.Markdown("### 上传图片或使用本机摄像头检测")
                        image_input = gr.Image(
                            label="图片 / 本机摄像头",
                            type="numpy",
                            sources=["upload", "webcam", "clipboard"],
                            image_mode="RGB",
                        )
                        with gr.Row():
                            with gr.Column(scale=1):
                                image_conf = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="置信度阈值")
                            with gr.Column(scale=1):
                                image_iou = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label="IoU 阈值")
                        with gr.Row():
                            image_voice = gr.Checkbox(
                                value=False,
                                label="启用语音播报",
                                visible=True,
                            )
                            image_detect_btn = gr.Button("开始检测", variant="primary", size="lg")

                    with gr.Column(scale=3):
                        gr.Markdown("### 检测结果")
                        image_output = gr.Image(label="标注结果", type="numpy")
                        image_result_text = gr.Markdown("等待检测...")

                with gr.Accordion("检测详情", open=False):
                    image_detections_json = gr.JSON(label="检测框详情", scale=1)
                    guest_history_json = gr.JSON(
                        label="本浏览器历史记录",
                        value=[],
                        visible=PUBLIC_GUEST_MODE,
                    )

                # 图片检测绑定（防重复点击：处理期间禁用按钮）
                if PUBLIC_GUEST_MODE:
                    image_detect_btn.click(
                        fn=handle_guest_image_detect,
                        inputs=[image_input, image_conf, image_iou, guest_history_state],
                        outputs=[
                            image_output,
                            image_result_text,
                            image_detections_json,
                            detection_state,
                            guest_history_state,
                            guest_history_json,
                        ],
                    )
                else:
                    image_detect_btn.click(
                        fn=lambda: gr.update(interactive=False),
                        outputs=[image_detect_btn],
                    ).then(
                        fn=handle_image_detect,
                        inputs=[image_input, image_conf, image_iou, image_voice],
                        outputs=[image_output, image_result_text, image_detections_json, detection_state],
                    ).then(
                        fn=lambda: gr.update(interactive=True),
                        outputs=[image_detect_btn],
                    )

            # ------------------------------------------------------------------
            # Tab 2: 视频检测
            # ------------------------------------------------------------------
            with gr.TabItem("视频检测", id="tab_video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 上传视频文件")
                        video_input = gr.Video(label="选择视频 (MP4/MOV/AVI)")
                        with gr.Row():
                            video_conf = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="置信度阈值")
                            video_iou = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label="IoU 阈值")
                        video_detect_btn = gr.Button("开始检测", variant="primary", size="lg")
                        gr.Markdown("> 处理时间取决于视频长度，请耐心等待。")

                    with gr.Column(scale=2):
                        gr.Markdown("### 检测结果视频")
                        video_output = gr.Video(label="标注后的视频")
                        video_result_text = gr.Markdown("等待检测...")

                if PUBLIC_GUEST_MODE:
                    video_detect_btn.click(
                        fn=lambda: gr.update(interactive=False),
                        outputs=[video_detect_btn],
                    ).then(
                        fn=handle_guest_video_detect,
                        inputs=[video_input, video_conf, video_iou, guest_history_state],
                        outputs=[
                            video_output,
                            video_result_text,
                            detection_state,
                            guest_history_state,
                            guest_history_json,
                        ],
                    ).then(
                        fn=lambda: gr.update(interactive=True),
                        outputs=[video_detect_btn],
                    )
                else:
                    video_detect_btn.click(
                        fn=lambda: gr.update(interactive=False),
                        outputs=[video_detect_btn],
                    ).then(
                        fn=handle_video_detect,
                        inputs=[video_input, video_conf, video_iou],
                        outputs=[video_output, video_result_text, detection_state],
                    ).then(
                        fn=lambda: gr.update(interactive=True),
                        outputs=[video_detect_btn],
                    )

            if PUBLIC_GUEST_MODE:
                # ------------------------------------------------------------------
                # Guest Tab 3: browser-local realtime camera stream detection
                # ------------------------------------------------------------------
                with gr.TabItem("实时摄像头", id="tab_guest_webcam"):
                    gr.Markdown(
                        "### 本机摄像头实时检测\n"
                        "使用访问者浏览器自己的摄像头连续传帧到 YOLO-GDL 推理，结果和历史只保存在当前浏览器会话中。"
                    )
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=2):
                            guest_cam_input = gr.Image(
                                label="本机摄像头",
                                type="numpy",
                                sources=["webcam"],
                                image_mode="RGB",
                                streaming=True,
                                webcam_options=gr.WebcamOptions(
                                    mirror=False,
                                    constraints={
                                        "video": {
                                            "width": {"ideal": 640},
                                            "height": {"ideal": 480},
                                            "frameRate": {"ideal": 30, "min": 30},
                                        }
                                    },
                                ),
                            )
                            with gr.Row():
                                guest_cam_conf = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="置信度阈值")
                                guest_cam_iou = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label="IoU 阈值")
                            gr.Checkbox(value=True, label="启用本机语音报警")
                            guest_env_check = gr.Textbox(
                                label="浏览器环境自检",
                                interactive=False,
                                lines=5,
                            )
                            gr.Button("检查摄像头/串口权限环境", variant="secondary").click(
                                fn=None,
                                outputs=[guest_env_check],
                                js=GUEST_BROWSER_ENV_JS,
                                queue=False,
                            )

                        with gr.Column(scale=3):
                            guest_cam_output = gr.Image(label="检测结果", type="numpy")
                            guest_cam_result = gr.Markdown("等待检测...")
                            gr.JSON(label="检测详情")
                            gr.Gallery(
                                label="本浏览器最近矿石截图",
                                columns=4,
                                rows=2,
                                object_fit="contain",
                                height="300px",
                                show_label=True,
                            )

                    guest_cam_timer = gr.Timer(value=0.5, active=True)
                    guest_cam_timer.tick(
                        fn=handle_guest_camera_stream_preview,
                        inputs=[
                            guest_cam_input,
                            guest_cam_conf,
                            guest_cam_iou,
                        ],
                        outputs=[
                            guest_cam_output,
                            guest_cam_result,
                        ],
                        queue=False,
                        trigger_mode="always_last",
                        concurrency_limit=1,
                        concurrency_id="guest_webcam_stream",
                    )

                # ------------------------------------------------------------------
                # Guest Tab 4: session-local statistics
                # ------------------------------------------------------------------
                with gr.TabItem("检测统计", id="tab_guest_stats"):
                    gr.Markdown("### 本浏览器检测统计")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            guest_stats_pie = gr.Image(label="类别占比饼图", type="numpy")
                        with gr.Column(scale=1):
                            guest_stats_bar = gr.Image(label="数量柱状图", type="numpy")
                    guest_stats_summary = gr.Markdown("暂无本浏览器检测数据")
                    guest_stats_json = gr.JSON(label="本浏览器历史记录", value=[])
                    guest_stats_refresh = gr.Button("刷新本浏览器历史", variant="secondary")
                    guest_stats_refresh.click(
                        fn=refresh_guest_statistics,
                        inputs=[guest_history_state],
                        outputs=[
                            guest_stats_pie,
                            guest_stats_bar,
                            guest_stats_summary,
                            guest_stats_json,
                        ],
                    )

                # ------------------------------------------------------------------
                # Guest Tab 5: browser-side hardware placeholder
                # ------------------------------------------------------------------
                with gr.TabItem("设备控制", id="tab_guest_device"):
                    gr.Markdown("### 本机 CH340 硬件控制")
                    gr.Markdown(
                        "公网访客的设备控制调用访问者浏览器授权的本机 CH340，服务端不会访问你的 COM3。"
                        "公网串口权限需要 HTTPS 域名，建议用 https://yolo.cat.com 访问。"
                    )
                    guest_serial_status = gr.HTML(value=GUEST_SERIAL_OFF_HTML, label="串口状态")
                    guest_device_msg = gr.Textbox(label="操作反馈", interactive=False)
                    with gr.Row():
                        gr.Button("连接本机 CH340", variant="primary").click(
                            fn=None,
                            outputs=[guest_serial_status, guest_device_msg],
                            js=GUEST_SERIAL_CONNECT_JS,
                            queue=False,
                        )
                        gr.Button("断开本机 CH340", variant="stop").click(
                            fn=None,
                            outputs=[guest_serial_status, guest_device_msg],
                            js=GUEST_SERIAL_DISCONNECT_JS,
                            queue=False,
                        )
                    with gr.Accordion("CH340 预留接口测试", open=True):
                        gr.Markdown(
                            "访问者插入自己的 USB-CH340 后，先点击连接，再发送测试包。"
                            "测试包只走访问者本机浏览器串口，不经过服务器串口。"
                        )
                        guest_serial_payload = gr.Textbox(
                            label="测试 JSON",
                            value='{"cmd":"ping","source":"web-client"}',
                        )
                        gr.Button("发送串口测试包", variant="secondary").click(
                            fn=None,
                            inputs=[guest_serial_payload],
                            outputs=[guest_serial_status, guest_device_msg],
                            js=GUEST_SERIAL_SEND_CUSTOM_JS,
                            queue=False,
                        )
                    with gr.Row():
                        gr.Button("履带启动", variant="primary").click(
                            fn=None,
                            outputs=[guest_serial_status, guest_device_msg],
                            js=make_guest_serial_send_js(
                                [{"cmd": "belt", "action": "start"}],
                                "已向本机 CH340 发送：履带启动",
                            ),
                            queue=False,
                        )
                        gr.Button("履带停止", variant="stop").click(
                            fn=None,
                            outputs=[guest_serial_status, guest_device_msg],
                            js=make_guest_serial_send_js(
                                [{"cmd": "belt", "action": "stop"}],
                                "已向本机 CH340 发送：履带停止",
                            ),
                            queue=False,
                        )
                    with gr.Row():
                        gr.Button("照明开启", variant="primary").click(
                            fn=None,
                            outputs=[guest_serial_status, guest_device_msg],
                            js=make_guest_serial_send_js(
                                [{"cmd": "light", "action": "on"}],
                                "已向本机 CH340 发送：照明开启",
                            ),
                            queue=False,
                        )
                        gr.Button("照明关闭", variant="stop").click(
                            fn=None,
                            outputs=[guest_serial_status, guest_device_msg],
                            js=make_guest_serial_send_js(
                                [{"cmd": "light", "action": "off"}],
                                "已向本机 CH340 发送：照明关闭",
                            ),
                            queue=False,
                        )
                    gr.Button("紧急停止", variant="stop", size="lg").click(
                        fn=None,
                        outputs=[guest_serial_status, guest_device_msg],
                        js=make_guest_serial_send_js(
                            [
                                {"cmd": "belt", "action": "stop"},
                                {"cmd": "light", "action": "off"},
                            ],
                            "已向本机 CH340 发送：紧急停止、履带停止、照明关闭",
                            "紧急停止",
                        ),
                        queue=False,
                    )
                    with gr.Accordion("音频功能", open=False):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("#### 麦克风录音")
                                guest_mic_input = gr.Audio(
                                    sources=["microphone"],
                                    type="numpy",
                                    label="录制语音指令",
                                )
                                guest_mic_record_btn = gr.Button("提交录音", variant="secondary")
                                guest_mic_result = gr.Textbox(label="录音信息", interactive=False)
                        guest_mic_record_btn.click(
                            fn=handle_guest_audio_record,
                            inputs=[guest_mic_input],
                            outputs=[guest_mic_result],
                        )
                    guest_voice_input = gr.Textbox(label="语音文本", value="检测到矿石")
                    gr.Button("本机语音测试", variant="secondary").click(
                        fn=guest_voice_message,
                        inputs=[guest_voice_input],
                        outputs=[guest_device_msg],
                        js=GUEST_VOICE_TEST_JS,
                        queue=False,
                    )

                # ------------------------------------------------------------------
                # Guest Tab 6: session-local log
                # ------------------------------------------------------------------
                with gr.TabItem("系统日志", id="tab_guest_logs"):
                    gr.Markdown("### 本浏览器日志")
                    guest_log_output = gr.Textbox(
                        label="本浏览器运行日志",
                        interactive=False,
                        lines=20,
                        max_lines=20,
                        elem_classes=["log-box"],
                        value="本浏览器暂无检测日志",
                    )
                    guest_log_refresh = gr.Button("刷新本浏览器日志", variant="secondary")
                    guest_log_refresh.click(
                        fn=format_guest_logs,
                        inputs=[guest_history_state],
                        outputs=[guest_log_output],
                    )

            if not PUBLIC_GUEST_MODE:
                # ------------------------------------------------------------------
                # Tab 3: 实时摄像头（服务端 OpenCV 采集 + 定时轮询）
                # ------------------------------------------------------------------
                with gr.TabItem("实时摄像头", id="tab_webcam"):
                    gr.Markdown(
                        "### 传送带实时识别与记录\n"
                        "> **工作流程**: 摄像头对准传送带 → GPU 实时推理 → "
                        "煤块/矿石识别 → 日志记录与统计展示"
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### 参数设置")
                            cam_index_input = gr.Number(
                                value=0,
                                label="摄像头序号 (0=默认)",
                                precision=0,
                            )
                            cam_conf = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="置信度阈值")
                            cam_iou = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label="IoU 阈值")
                            cam_voice = gr.Checkbox(value=True, label="语音播报")
                            with gr.Row():
                                cam_start_btn = gr.Button("启动检测", variant="primary")
                                cam_stop_btn = gr.Button("停止检测", variant="stop")
                            cam_status = gr.Textbox(label="运行状态", interactive=False, value="未启动")
                            cam_stats_text = gr.Textbox(
                                label="实时统计（煤块/矿石/推理）",
                                interactive=False,
                                value="等待启动...",
                            )

                        with gr.Column(scale=3):
                            cam_output = gr.Image(
                                label="传送带实时检测画面",
                                type="numpy",
                            )
                            cam_gallery = gr.Gallery(
                                label="最近检测截图（矿石检出时自动保存）",
                                columns=4,
                                rows=2,
                                object_fit="contain",
                                height="300px",
                                show_label=True,
                            )

                    cam_start_btn.click(
                        fn=lambda: (gr.update(interactive=False), gr.update(interactive=False)),
                        outputs=[cam_start_btn, cam_stop_btn],
                    ).then(
                        fn=start_camera,
                        inputs=[cam_index_input, cam_conf, cam_iou, cam_voice],
                        outputs=[cam_status],
                    ).then(
                        fn=lambda: (gr.update(interactive=True), gr.update(interactive=True)),
                        outputs=[cam_start_btn, cam_stop_btn],
                    )
                    cam_stop_btn.click(
                        fn=lambda: (gr.update(interactive=False), gr.update(interactive=False)),
                        outputs=[cam_start_btn, cam_stop_btn],
                    ).then(
                        fn=stop_camera,
                        outputs=[cam_status],
                    ).then(
                        fn=lambda: (gr.update(interactive=True), gr.update(interactive=True)),
                        outputs=[cam_start_btn, cam_stop_btn],
                    )

                    cam_timer = gr.Timer(value=0.08, active=True)
                    cam_timer.tick(
                        fn=poll_camera,
                        outputs=[cam_output, cam_stats_text, cam_gallery],
                        queue=False,
                        trigger_mode="always_last",
                    )

                # ------------------------------------------------------------------
                # Tab 4: 检测统计
                # ------------------------------------------------------------------
                with gr.TabItem("检测统计", id="tab_stats"):
                    gr.Markdown("### 检测数据可视化")
                    with gr.Row():
                        stats_refresh_btn = gr.Button("刷新统计图表", variant="secondary")

                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            stats_pie = gr.Image(label="类别占比饼图", type="numpy")
                        with gr.Column(scale=1):
                            stats_bar = gr.Image(label="数量柱状图", type="numpy")

                    stats_summary = gr.Markdown("暂无统计数据，请先进行检测。")

                    stats_refresh_btn.click(
                        fn=lambda state: refresh_statistics(state or get_camera_stats()),
                        inputs=[detection_state],
                        outputs=[stats_pie, stats_bar, stats_summary],
                    )

                # ------------------------------------------------------------------
                # Tab 5: 设备控制
                # ------------------------------------------------------------------
                with gr.TabItem("设备控制", id="tab_device"):
                    gr.Markdown("### 下位机设备远程控制（串口 JSON 协议）")
                    gr.Markdown(
                        "> 通信协议：USB转TTL (UART)，波特率 115200，数据格式 JSON\n"
                        "> 连接串口后可真实控制下位机；未连接时为模拟模式。"
                    )

                    with gr.Accordion("串口连接管理", open=True):
                        with gr.Row():
                            serial_port = gr.Textbox(
                                value="COM3",
                                label="串口号 (如 COM3)",
                            )
                            serial_baud = gr.Number(
                                value=115200,
                                label="波特率",
                                precision=0,
                            )
                            serial_connect_btn = gr.Button("连接串口", variant="primary")
                            serial_disconnect_btn = gr.Button("断开串口", variant="stop")
                        serial_status = gr.Textbox(
                            label="串口状态",
                            interactive=False,
                            value="未连接（模拟模式）",
                        )
                        serial_connect_btn.click(
                            fn=handle_serial_connect,
                            inputs=[serial_port, serial_baud],
                            outputs=[serial_status],
                        )
                        serial_disconnect_btn.click(
                            fn=handle_serial_disconnect,
                            outputs=[serial_status],
                        )

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### 履带控制")
                            with gr.Row():
                                belt_start_btn = gr.Button("履带启动", variant="primary")
                                belt_stop_btn = gr.Button("履带停止", variant="stop")

                        with gr.Column(scale=1):
                            gr.Markdown("#### 照明控制")
                            with gr.Row():
                                light_on_btn = gr.Button("照明开启", variant="primary")
                                light_off_btn = gr.Button("照明关闭", variant="stop")

                    with gr.Row():
                        with gr.Column(scale=1):
                            emergency_btn = gr.Button("紧急停止", variant="stop", size="lg")
                            emergency_msg = gr.Textbox(label="操作反馈", interactive=False)

                        with gr.Column(scale=2):
                            gr.Markdown("#### 语音播报测试")
                            with gr.Row():
                                voice_test_input = gr.Textbox(
                                    label="播报文本",
                                    value="检测到矿石",
                                    placeholder="输入要播报的文字...",
                                )
                                voice_test_btn = gr.Button("播报", variant="primary")

                    with gr.Accordion("音频功能", open=False):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("#### 麦克风录音")
                                mic_input = gr.Audio(sources=["microphone"], type="numpy", label="录制语音指令")
                                mic_record_btn = gr.Button("提交录音", variant="secondary")
                                mic_result = gr.Textbox(label="录音信息", interactive=False)

                    belt_start_btn.click(
                        fn=lambda: handle_belt_control("start"),
                        outputs=[emergency_msg],
                    )
                    belt_stop_btn.click(
                        fn=lambda: handle_belt_control("stop"),
                        outputs=[emergency_msg],
                    )
                    light_on_btn.click(
                        fn=lambda: handle_light_control("on"),
                        outputs=[emergency_msg],
                    )
                    light_off_btn.click(
                        fn=lambda: handle_light_control("off"),
                        outputs=[emergency_msg],
                    )
                    emergency_btn.click(
                        fn=handle_emergency_stop,
                        outputs=[emergency_msg],
                    )
                    voice_test_btn.click(
                        fn=handle_voice_test,
                        inputs=[voice_test_input],
                        outputs=[emergency_msg],
                    )
                    mic_record_btn.click(
                        fn=handle_audio_record,
                        inputs=[mic_input],
                        outputs=[mic_result],
                    )

                # ------------------------------------------------------------------
                # Tab 6: 系统日志
                # ------------------------------------------------------------------
                with gr.TabItem("系统日志", id="tab_logs"):
                    gr.Markdown("### 系统运行日志")
                    log_output = gr.Textbox(
                        label="实时日志",
                        interactive=False,
                        lines=20,
                        max_lines=20,
                        elem_classes=["log-box"],
                        value=get_logs(),
                    )
                    log_refresh_btn = gr.Button("刷新日志", variant="secondary")
                    log_refresh_btn.click(fn=get_logs, outputs=[log_output])

                    # 日志自动定时刷新（每0.5秒）
                    log_timer = gr.Timer(value=0.5, active=True)
                    log_timer.tick(fn=get_logs, outputs=[log_output])

        # ============ 页脚 ============
        gr.Markdown(
            "---\n"
            "**煤矿与矿石智能识别监测系统** | 上位机端 (AI视觉 + Web控制台)\n\n"
            "采集→识别→记录: 摄像头实时推理 → 煤块/矿石统计 → 日志与图表展示\n\n"
            f"技术栈: Python + Gradio + {loaded_model_name} + OpenCV + pyttsx3 + pyserial\n\n"
            f"通信协议: UART串口 JSON (115200bps) | 模型类别: {model_display_classes}"
        )

    return demo


def get_available_port(default_port: int = 7860, max_port: int = 7899) -> int:
    """Return the requested Gradio port, or the next free port if it is busy."""
    env_port = os.environ.get("GRADIO_SERVER_PORT")
    if env_port:
        return int(env_port)

    for port in range(default_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port

    raise OSError(f"Cannot find empty port in range: {default_port}-{max_port}.")


def write_frpc_config(frpc_dir: Path, local_port: int) -> Path:
    """Write frpc config that exposes the local Gradio port through Aliyun ECS."""
    config_path = frpc_dir / "frpc.toml"
    frp_server = os.environ.get("COAL_FRP_SERVER", "YOUR_FRP_SERVER_IP")
    frp_token = os.environ.get("COAL_FRP_TOKEN", "YOUR_FRP_TOKEN")
    config_path.write_text(
        "\n".join(
            [
                f'serverAddr = "{frp_server}"',
                "serverPort = 7000",
                "",
                'auth.method = "token"',
                f'auth.token = "{frp_token}"',
                "",
                "[[proxies]]",
                'name = "gradio-yolo-gdl"',
                'type = "tcp"',
                'localIP = "127.0.0.1"',
                f"localPort = {local_port}",
                "remotePort = 8080",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _get_coal_frpc_pid_file() -> Path:
    """Per-app PID file so each app only manages its own frpc instance."""
    return Path(__file__).resolve().parent / ".gradio" / "frpc_coal_gangue.pid"


def _stop_own_frpc() -> None:
    """Kill only *this app's* previous frpc process — never touch other apps."""
    pid_file = _get_coal_frpc_pid_file()
    if not pid_file.exists():
        return
    try:
        old_pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        return
    try:
        subprocess.run(
            ["taskkill", "/PID", str(old_pid), "/F"],
            capture_output=True,
            text=True,
            encoding="gbk",
            errors="ignore",
        )
        print(f"[INFO] 已清理本 App 旧 frpc 进程 (PID={old_pid})")
        _log.info(f"已清理本 App 旧 frpc 进程 (PID={old_pid})")
    except Exception:
        pass
    finally:
        pid_file.unlink(missing_ok=True)


def start_frpc_tunnel(project_root: Path, local_port: int) -> subprocess.Popen | None:
    """Start frpc in a separate Windows console for public access."""
    frpc_dir = project_root / "frp_windows" / "frp_0.68.1_windows_amd64"
    frpc_exe = frpc_dir / "frpc.exe"
    if not frpc_exe.exists():
        print(f"[WARN] 未找到 frpc.exe，跳过公网端启动: {frpc_exe}")
        _log.warning(f"未找到 frpc.exe，跳过公网端启动: {frpc_exe}")
        add_log(f"未找到 frpc.exe，跳过公网端启动: {frpc_exe}")
        return None

    config_path = write_frpc_config(frpc_dir, local_port)
    _stop_own_frpc()  # only kill our own previous instance

    # Windows 下 CREATE_NEW_CONSOLE 在 IDE 终端环境中不可靠，
    # 改为直接后台静默启动 frpc.exe，stdout/stderr 重定向避免阻塞。
    try:
        proc = subprocess.Popen(
            [str(frpc_exe), "-c", str(config_path)],
            cwd=str(frpc_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception as e:
        print(f"[WARN] frpc 启动失败: {e}")
        _log.warning(f"frpc 启动失败: {e}")
        add_log(f"frpc 启动失败: {e}")
        return None

    # 等待半秒确认进程是否存活
    time.sleep(0.5)
    if proc.poll() is not None:
        print(f"[WARN] frpc 启动后立即退出，退出码: {proc.returncode}")
        _log.warning(f"frpc 启动后立即退出，退出码: {proc.returncode}")
        add_log(f"frpc 启动后立即退出，退出码: {proc.returncode}")
        return None

    # Remember PID so next launch can clean it up
    _get_coal_frpc_pid_file().parent.mkdir(parents=True, exist_ok=True)
    _get_coal_frpc_pid_file().write_text(str(proc.pid))

    print(f"[INFO] 公网访问: http://{os.environ.get('COAL_FRP_SERVER', 'YOUR_FRP_SERVER_IP')}:8080")
    print(f"[INFO] frpc 已启动，本地端口 {local_port} -> 阿里云 8080")
    _log.info(f"frpc 已启动，本地端口 {local_port} -> 阿里云 8080")
    add_log(f"公网访问: http://{os.environ.get('COAL_FRP_SERVER', 'YOUR_FRP_SERVER_IP')}:8080")
    return proc


# ============================================================================
# 入口
# ============================================================================
def main() -> None:
    """应用入口：初始化检测引擎、构建界面、启动服务。."""
    global detector, loaded_model_name, loaded_model_path, loaded_model_classes

    print("=" * 60)
    print("  煤矿履带煤炭与矿石智能识别监测系统 - 启动中...")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent
    default_model_path = project_root / "runs" / "train" / "YOLO-GDL" / "weights" / "best.pt"

    # 模型路径优先级：命令行参数 > COAL_MODEL_PATH 环境变量 > YOLO-GDL 最终权重 > yolov8n.pt
    model_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get(
            "COAL_MODEL_PATH",
            str(default_model_path),
        )
    )

    if not os.path.exists(model_path):
        fallback = project_root / "yolov8n.pt"
        if os.path.exists(fallback):
            model_path = str(fallback)
            print(f"[WARN] YOLO-GDL best.pt 未找到，回退到预训练权重: {model_path}")
        else:
            print(f"[ERROR] 模型文件不存在: {model_path}")
            sys.exit(1)

    model_path = str(Path(model_path).resolve())
    loaded_model_path = model_path
    loaded_model_name = "YOLO-GDL" if "YOLO-GDL" in model_path else Path(model_path).stem

    print(f"[INFO] 加载模型: {model_path}")
    _log.info(f"加载模型: {model_path}")
    detector = CoalGangueDetector(model_path)
    loaded_model_classes = ", ".join(str(v) for v in detector.model.names.values())
    print(f"[INFO] 模型已加载，类别: {detector.model.names}")
    _log.info(f"模型已加载，类别: {detector.model.names}")

    add_log("系统启动")
    _log.info("系统启动完成")
    add_log(f"模型路径: {model_path}")
    add_log(f"检测类别: {detector.model.names}")

    voice_alert.speak("煤炭与矿石识别系统已启动")

    demo = build_app()
    server_port = get_available_port()
    if server_port != 7860:
        print(f"[WARN] 7860 端口已被占用，自动改用端口: {server_port}")
        _log.warning(f"7860 端口已被占用，自动改用端口: {server_port}")
        add_log(f"7860 端口已被占用，自动改用端口: {server_port}")
    else:
        print("[INFO] Gradio 服务端口: 7860")
        _log.info("Gradio 服务端口: 7860")

    start_frpc_tunnel(project_root, server_port)

    # 公网访问使用阿里云 frp，关闭 Gradio 官方 share 中转以避免重复代理和上传卡顿。
    # Gradio 5.x 校验 .then() 链中所有被取消函数的 queue 属性，fn=None 的函数默认 queue=False。
    # 需要手动将所有函数的 queue 设为 True 以通过 validate_queue_settings 校验。
    for fn in demo.fns.values():
        fn.queue = True
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=False,
        show_error=True,
        inbrowser=True,
        quiet=False,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
