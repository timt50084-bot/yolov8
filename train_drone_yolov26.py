from ultralytics import YOLO
import torch
import os
from tkinter import Tk, filedialog


def select_detect_path():
    """弹出弹窗选择检测的图片/文件夹/视频路径（权重路径固定）"""
    # 隐藏tkinter主窗口
    root = Tk()
    root.withdraw()

    # 选择检测类型（新增视频选项）
    print("请选择检测类型：")
    print("1 - 检测单张图片（红外/可见光）")
    print("2 - 检测图片文件夹（批量图片）")
    print("3 - 检测单个视频文件")
    print("4 - 检测视频文件夹（批量视频）")

    while True:
        choice = input("输入 1/2/3/4：").strip()
        if choice in ['1', '2', '3', '4']:
            break
        print("❌ 输入错误，请输入 1、2、3 或 4！")

    # 弹窗选择对应路径
    if choice == '1':
        # 选单张图片
        detect_path = filedialog.askopenfilename(
            title="选择要检测的图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )
    elif choice == '2':
        # 选图片文件夹
        detect_path = filedialog.askdirectory(title="选择图片文件夹")
    elif choice == '3':
        # 选单个视频
        detect_path = filedialog.askopenfilename(
            title="选择要检测的视频",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv"), ("所有文件", "*.*")]
        )
    else:
        # 选视频文件夹
        detect_path = filedialog.askdirectory(title="选择视频文件夹")

    # 校验是否选择了路径
    if not detect_path:
        print("❌ 未选择任何检测路径，程序退出！")
        exit()
    return detect_path, choice  # 返回路径+检测类型


def main():
    # 1. 确认 GPU 可用
    print(f"GPU 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"使用显卡: {torch.cuda.get_device_name(0)}")

    # 2. 固定加载50+50轮的权重文件（无需选择）
    model_path = r'D:\project\ultralytics-main\runs\detect\drone_vehicle_train\yolov8_drone_2\weights\best.pt'

    if not os.path.exists(model_path):
        print(f"❌ 50+50轮权重文件不存在：{model_path}")
        print("请检查权重文件路径是否正确！")
        return

    model = YOLO(model_path)
    print(f"✅ 成功加载50+50轮权重：{model_path}")

    # 3. 弹窗选择要检测的路径（图片/视频）
    print("\n📌 请选择要检测的目标：")
    source, detect_type = select_detect_path()
    print(f"✅ 检测目标路径：{source}")

    # 校验检测路径是否存在
    if not os.path.exists(source):
        print(f"❌ 检测目标不存在：{source}")
        return

    # 4. 执行检测（针对视频优化参数）
    # 视频检测参数优化：增加视频帧率、保存视频结果
    results = model(
        source=source,
        device=0,  # 强制用GPU（视频检测更高效）
        conf=0.3,  # 红外/视频推荐0.3，可见光图片可改0.5
        iou=0.7,
        save=True,  # 保存检测结果（图片/视频）
        save_txt=True,  # 保存txt标注（仅图片/视频帧）
        show=False,  # 不实时弹窗显示
        imgsz=640,
        vid_stride=1,  # 视频帧采样步长（1=逐帧检测）
        stream=True  # 视频流式检测，降低内存占用
    )

    # 5. 打印检测结果（区分图片/视频）
    print("\n✅ 检测完成！")
    print(f"📂 检测结果保存位置：{model.predictor.save_dir}")

    if detect_type in ['1', '2']:
        # 图片检测结果统计
        total_targets = 0
        for result in results:
            img_name = os.path.basename(result.path)
            target_num = len(result.boxes)
            total_targets += target_num
            print(f"  - {img_name}：检测到 {target_num} 个目标")
        print(f"📊 总计检测到 {total_targets} 个目标")
    else:
        # 视频检测结果提示
        if detect_type == '3':
            video_name = os.path.basename(source)
            print(f"  - 视频 {video_name} 检测完成，已保存带检测框的视频文件")
        else:
            print(f"  - 视频文件夹检测完成，所有视频已保存检测结果")
        print("📌 视频检测结果说明：")
        print("   - 带检测框的完整视频保存在结果文件夹中")
        print("   - 每帧的检测标注保存在 labels 文件夹中（按帧编号命名）")


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()