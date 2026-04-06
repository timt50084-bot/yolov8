from ultralytics import YOLO
import torch
import os
import tkinter as tk
from tkinter import filedialog

# ===================== 核心配置（只需改这里）=====================
MODEL_PATH = r'C:\Users\20379\Desktop\best.pt'
CONF_THRESHOLD = 0.5  # 置信度阈值
USE_GPU = True  # 是否使用GPU推理


# ===================== 手动选择文件/文件夹 =====================
def select_source_path(infer_type):
    """
    根据推理类型弹出窗口，手动选择路径
    infer_type: single_img/batch_img/video
    """
    # 隐藏tkinter主窗口
    root = tk.Tk()
    root.withdraw()

    if infer_type == "single_img":
        # 选择单张图片
        file_path = filedialog.askopenfilename(
            title="选择要检测的单张图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif"), ("所有文件", "*.*")]
        )
    elif infer_type == "batch_img":
        # 选择文件夹
        file_path = filedialog.askdirectory(
            title="选择要批量检测的图片文件夹"
        )
    elif infer_type == "video":
        # 选择视频文件（或输入摄像头ID）
        choice = input("请选择：1-检测视频文件  2-检测摄像头（输入数字）：")
        if choice == "1":
            file_path = filedialog.askopenfilename(
                title="选择要检测的视频文件",
                filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
            )
        else:
            cam_id = input("请输入摄像头ID（默认0）：") or "0"
            file_path = cam_id

    if not file_path:
        raise ValueError("❌ 未选择任何文件/文件夹，退出推理")

    print(f"\n✅ 已选择：{file_path}")
    return file_path


# ===================== 初始化模型 =====================
def init_model():
    """加载训练好的模型"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在：{MODEL_PATH}")

    model = YOLO(MODEL_PATH)
    # 设置推理设备
    device = 0 if (USE_GPU and torch.cuda.is_available()) else 'cpu'
    print(f"✅ 模型加载成功，使用 {device} 推理")
    return model, device


# ===================== 方法1：单张图片推理 =====================
def infer_single_image(model, device, source_path):
    """对单张图片推理，输出详细检测结果"""
    print(f"\n📸 开始检测单张图片...")
    results = model(
        source=source_path,
        device=device,
        conf=CONF_THRESHOLD,
        iou=0.7,
        save=True,  # 保存检测后的图片
        show=False  # Windows下建议设为False
    )

    # 解析检测结果
    for r in results:
        boxes = r.boxes
        print(f"\n🔍 检测结果：共识别 {len(boxes)} 个目标")
        for idx, box in enumerate(boxes):
            cls_idx = box.cls.item()
            cls_name = model.names[cls_idx]
            conf = box.conf.item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  目标{idx + 1}：{cls_name} | 置信度：{conf:.2f} | 坐标：[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    print(f"\n💾 检测后的图片已保存到：runs/detect/predict")


# ===================== 方法2：批量图片推理 =====================
def infer_batch_images(model, device, source_path):
    """对文件夹下所有图片批量推理，保存结果和标注文件"""
    print(f"\n📁 开始批量检测文件夹...")
    results = model(
        source=source_path,
        device=device,
        conf=CONF_THRESHOLD,
        iou=0.7,
        save=True,  # 保存检测后的图片
        save_txt=True,  # 保存检测框坐标到txt
        save_conf=True,  # txt文件中包含置信度
        batch=8,  # 批量大小（根据GPU显存调整）
        show=False
    )

    print(f"\n📊 批量检测完成：共处理 {len(results)} 张图片")
    print(f"💾 结果保存路径：")
    print(f"  - 检测图片：runs/detect/predict")
    print(f"  - 标注文件：runs/detect/predict/labels")


# ===================== 方法3：视频/摄像头推理 =====================
def infer_video_or_camera(model, device, source_path):
    """检测视频文件或实时摄像头"""
    is_camera = source_path.isdigit()  # 判断是否是摄像头（source=0/1等）
    if is_camera:
        print(f"\n🎥 开始实时摄像头检测（摄像头ID：{source_path}）")
    else:
        print(f"\n🎬 开始检测视频文件...")

    results = model(
        source=source_path,
        device=device,
        conf=CONF_THRESHOLD,
        iou=0.7,
        save=True,  # 保存检测后的视频（摄像头模式不生效）
        show=False,  # 实时显示窗口（摄像头模式建议设为True）
        vid_stride=1  # 视频帧步长（1=逐帧检测）
    )

    if not is_camera:
        print(f"\n💾 检测后的视频已保存到：runs/detect/predict")
    else:
        print(f"\n⚠️  按 'q' 键退出摄像头检测")


# ===================== 主函数（手动选择+一键切换）=====================
if __name__ == '__main__':
    try:
        # 选择推理类型
        print("===== 无人机跨模态车辆检测推理工具 =====")
        print("请选择推理方式：")
        print("1 - 单张图片检测")
        print("2 - 批量图片检测")
        print("3 - 视频/摄像头检测")
        infer_choice = input("输入数字（1/2/3）：") or "1"

        infer_type_map = {"1": "single_img", "2": "batch_img", "3": "video"}
        infer_type = infer_type_map.get(infer_choice, "single_img")

        # 手动选择路径
        source_path = select_source_path(infer_type)

        # 初始化模型
        model, device = init_model()

        # 执行推理
        if infer_type == "single_img":
            infer_single_image(model, device, source_path)
        elif infer_type == "batch_img":
            infer_batch_images(model, device, source_path)
        elif infer_type == "video":
            infer_video_or_camera(model, device, source_path)

        print("\n🎉 推理完成！")

    except Exception as e:
        print(f"\n❌ 推理出错：{e}")