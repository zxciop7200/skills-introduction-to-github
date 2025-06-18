import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import time
import hashlib
from functools import lru_cache
from werkzeug.utils import secure_filename


# 配置日志系统
def setup_logging():
    # 创建根记录器并设置日志级别
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 定义日志格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 创建控制台处理器并添加到记录器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 创建文件处理器（支持日志轮转）并添加到记录器
    if not os.path.exists('logs'):
        os.makedirs('logs')
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10 * 1024 * 1024,  # 单个日志文件最大10MB
        backupCount=5  # 保留最多5个备份文件
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# 初始化日志记录器并记录服务启动信息
logger = setup_logging()
logger.info("口罩检测系统后端服务启动")


# 模型初始化函数
def load_model(model_path):
    try:
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载YOLO模型
        logger.info(f"正在加载模型: {model_path}")
        model = YOLO(model_path)

        # 确定运行设备（GPU或CPU）并将模型移至该设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # 记录模型加载成功及设备信息
        logger.info(f"模型加载完成，设备: {device}")
        if device == 'cuda':
            logger.info(f"GPU信息: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")

        return model, device
    except Exception as e:
        # 记录模型加载异常信息
        logger.error(f"模型加载失败: {str(e)}", exc_info=True)
        raise


# 从环境变量获取模型路径，默认为models/best.pt
model_path = os.getenv('MODEL_PATH', 'models/best.pt')
try:
    model, device = load_model(model_path)
except Exception as e:
    # 模型初始化失败时记录致命错误并退出程序
    logger.fatal(f"模型初始化失败，服务无法启动: {str(e)}")
    exit(1)


# 模型性能监控类
class ModelPerformanceMonitor:
    def __init__(self):
        # 存储推理时间（毫秒）
        self.inference_times = []
        # 存储每次检测到的目标数量
        self.detection_counts = []
        # 记录总请求数
        self.request_counts = 0

    def record_inference(self, time_ms, detection_count):
        # 记录单次推理的时间和检测目标数
        self.inference_times.append(time_ms)
        self.detection_counts.append(detection_count)
        self.request_counts += 1

        # 每100次请求记录一次性能统计信息
        if self.request_counts % 100 == 0:
            self._log_statistics()

    def _log_statistics(self):
        # 计算最近100次请求的性能统计
        if not self.inference_times:
            return

        recent_times = self.inference_times[-100:]
        recent_detections = self.detection_counts[-100:]
        avg_time = sum(recent_times) / len(recent_times)
        avg_detections = sum(recent_detections) / len(recent_detections)

        logger.info(f"性能统计 - 最近{len(recent_times)}次请求: "
                    f"平均推理时间 {avg_time:.2f}ms, "
                    f"平均检测目标数 {avg_detections:.2f}")


# 初始化性能监控器
performance_monitor = ModelPerformanceMonitor()


# 图像预测函数
def predict(image, confidence_threshold=0.5):
    # 记录推理开始时间
    start_time = time.time()

    try:
        # 获取图像尺寸并检查有效性
        img_width, img_height = image.size
        if img_width == 0 or img_height == 0:
            logger.warning("输入图像尺寸为0")
            return [], [], []

        # 执行模型推理
        results = model(image)

        # 处理无检测结果的情况
        if not results or not results[0].boxes:
            logger.info("未检测到目标")
            end_time = time.time()
            performance_monitor.record_inference((end_time - start_time) * 1000, 0)
            return [], [], []

        # 提取边界框坐标、标签和置信度分数
        boxes = results[0].boxes.xyxy.cpu().numpy()  # 边界框坐标[x1, y1, x2, y2]
        labels = results[0].boxes.cls.cpu().numpy().astype(int)  # 类别标签
        scores = results[0].boxes.conf.cpu().numpy()  # 置信度分数

        # 置信度过滤，保留高于阈值的检测结果
        high_confidence = scores > confidence_threshold
        boxes = boxes[high_confidence]
        labels = labels[high_confidence]
        scores = scores[high_confidence]

        # 边界框坐标校验，确保坐标在图像范围内
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1 = max(0, min(img_width, x1))
            y1 = max(0, min(img_height, y1))
            x2 = max(x1, min(img_width, x2))
            y2 = max(y1, min(img_height, y2))
            boxes[i] = [x1, y1, x2, y2]

        # 记录推理结束时间和性能数据
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000
        performance_monitor.record_inference(inference_time_ms, len(boxes))

        return boxes, labels, scores

    except Exception as e:
        # 异常处理并记录错误信息
        logger.error(f"预测过程出错: {str(e)}", exc_info=True)
        end_time = time.time()
        performance_monitor.record_inference((end_time - start_time) * 1000, 0)
        return [], [], []


# 计算图像的MD5哈希值
def calculate_image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()


# 带缓存的预测函数，使用LRU缓存机制避免重复计算
@lru_cache(maxsize=128)
def cached_predict(image_hash, image_data):
    # 从字节数据打开图像并转换为RGB格式
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return predict(image)


# 初始化Flask应用
app = Flask(__name__)
# 设置最大请求大小限制为10MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
# 允许的上传文件扩展名，新增支持的格式
app.config['UPLOAD_EXTENSIONS'] = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

# 配置CORS（跨域资源共享）
CORS(app, resources={
    r"/predict": {
        # 允许特定来源访问API（开发环境配置）
        "origins": ["http://localhost:5000", "http://127.0.0.1:5000"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})


# 请求前处理中间件，记录请求开始时间
@app.before_request
def before_request():
    request.start_time = time.time()


# 请求后处理中间件，记录请求处理时间
@app.after_request
def after_request(response):
    duration = (time.time() - request.start_time) * 1000
    logger.info(f"{request.method} {request.path} - 状态: {response.status_code} - 耗时: {duration:.2f}ms")
    return response


# 口罩检测API端点
@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        # 检查请求中是否包含图像文件
        if 'image' not in request.files:
            return jsonify({"error": "未上传图像文件"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "未选择图像文件"}), 400

        # 安全处理文件名，防止路径遍历攻击
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower()

        # 检查文件扩展名是否允许，根据配置的扩展名判断
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            return jsonify({"error": "仅支持JPG、JPEG、PNG、BMP、GIF、TIFF、WEBP格式"}), 400

        # 读取图像文件内容
        image_bytes = file.read()

        # 验证图像文件有效性
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logger.warning(f"无效的图像文件: {filename}, 错误: {str(e)}")
            return jsonify({"error": "无效的图像文件"}), 400

        # 图像尺寸限制，防止过大图像导致性能问题
        max_size = (1280, 720)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image = image.resize(max_size, Image.LANCZOS)
            logger.info(f"图像已调整大小: {image.size}")
            # 将调整后的图像转换回字节流，根据原文件扩展名推导保存格式
            img_byte_arr = io.BytesIO()
            # 格式映射，根据文件扩展名选择保存格式
            format_map = {
                '.jpg': 'JPEG',
                '.jpeg': 'JPEG',
                '.png': 'PNG',
                '.bmp': 'BMP',
                '.gif': 'GIF',
                '.tiff': 'TIFF',
                '.webp': 'WEBP'
            }
            save_format = format_map.get(file_ext, 'JPEG')  # 默认用JPEG兜底
            image.save(img_byte_arr, format=save_format)
            image_bytes = img_byte_arr.getvalue()

        # 计算图像哈希用于缓存
        image_hash = calculate_image_hash(image_bytes)

        # 调用缓存的预测函数获取检测结果
        boxes, labels, scores = cached_predict(image_hash, image_bytes)

        # 构建API响应数据
        results = []
        for box, label, score in zip(boxes, labels, scores):
            results.append({
                "bbox": box.tolist(),  # 边界框坐标
                "label": int(label),  # 类别标签（0或1）
                "score": float(score),  # 置信度分数
                "label_name": "未戴口罩" if label == 0 else "已戴口罩"  # 标签名称
            })

        return jsonify(results), 200

    except Exception as e:
        # 异常处理并返回错误响应
        logger.error(f"API处理错误: {str(e)}", exc_info=True)
        return jsonify({"error": "服务器内部错误"}), 500


# 提供前端页面
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# 提供静态文件服务
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


# 健康检查接口，用于监控服务状态
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "OK",
        "model": model_path,
        "device": str(device),
        "uptime": time.time()
    }), 200


# 批量预测接口，支持同时处理多个图像
@app.route('/batch_predict', methods=['POST'])
def batch_predict_api():
    try:
        # 获取请求中的所有图像文件
        files = request.files.getlist('images')
        if not files or len(files) == 0:
            return jsonify({"error": "未上传图像文件"}), 400

        batch_results = []
        for file in files:
            # 安全处理文件名并检查扩展名
            filename = secure_filename(file.filename)
            file_ext = os.path.splitext(filename)[1].lower()

            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                batch_results.append({
                    "filename": filename,
                    "error": "不支持的文件格式"
                })
                continue

            try:
                # 处理单个图像文件
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # 图像尺寸调整
                max_size = (1280, 720)
                if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                    image = image.resize(max_size, Image.LANCZOS)
                    img_byte_arr = io.BytesIO()
                    # 格式映射，根据文件扩展名选择保存格式
                    format_map = {
                        '.jpg': 'JPEG',
                        '.jpeg': 'JPEG',
                        '.png': 'PNG',
                        '.bmp': 'BMP',
                        '.gif': 'GIF',
                        '.tiff': 'TIFF',
                        '.webp': 'WEBP'
                    }
                    save_format = format_map.get(file_ext, 'JPEG')  # 默认用JPEG兜底
                    image.save(img_byte_arr, format=save_format)
                    image_bytes = img_byte_arr.getvalue()

                # 计算哈希并进行预测
                image_hash = calculate_image_hash(image_bytes)
                boxes, labels, scores = cached_predict(image_hash, image_bytes)

                # 构建单个图像的检测结果
                detections = []
                for box, label, score in zip(boxes, labels, scores):
                    detections.append({
                        "bbox": box.tolist(),
                        "label": int(label),
                        "score": float(score),
                        "label_name": "未戴口罩" if label == 0 else "已戴口罩"
                    })

                batch_results.append({
                    "filename": filename,
                    "detections": detections
                })

            except Exception as e:
                # 记录单个图像处理错误
                logger.error(f"批量处理文件 {filename} 时出错: {str(e)}")
                batch_results.append({
                    "filename": filename,
                    "error": "处理过程中出错"
                })

        return jsonify(batch_results), 200

    except Exception as e:
        # 批量处理异常
        logger.error(f"批量预测错误: {str(e)}", exc_info=True)
        return jsonify({"error": "服务器内部错误"}), 500


# 应用程序入口点
if __name__ == '__main__':
    # 从环境变量获取调试模式配置，默认为False
    debug_mode = os.getenv('DEBUG', 'False') == 'True'
    # 从环境变量获取主机和端口配置
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '5000'))

    # 记录服务启动信息
    logger.info(f"服务启动 - 模式: {'调试' if debug_mode else '生产'}, 地址: {host}:{port}")
    # 启动Flask应用
    app.run(host=host, port=port, debug=debug_mode)