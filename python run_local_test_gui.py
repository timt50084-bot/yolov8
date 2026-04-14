# -*- coding: utf-8 -*-
"""
项目根目录直接运行的本地可视化测试脚本

功能：
1. 固定加载权重：C:\\Users\\20379\\Desktop\\rgbir_temporal\\weights\\best.pt
2. 图片 / 图片文件夹：
   - conf 固定为 0.7
   - 显示框 + 类别 + 概率
3. 视频：
   - conf 固定为 0.3
   - 只显示边框，不显示类别和概率
4. 通过图形界面手动选择：
   - 单张图片
   - 图片文件夹
   - 视频
5. 自动保存预测结果到项目目录下 outputs/local_gui_test/
6. 预测结束后自动打开输出目录

运行方式：
    python run_local_test_gui.py
"""

import os
import sys
import threading
import traceback
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# =========================
# 基础配置（按你的要求固定）
# =========================
WEIGHTS_PATH = r"C:\Users\20379\Desktop\rgbir_small\weights\best.pt"
IMAGE_CONF = 0.5
VIDEO_CONF = 0.3
IMGSZ = 1024

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "local_gui_test"

# 为了优先使用当前项目里的 ultralytics 代码
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError(
        "导入 ultralytics 失败，请确认你是在 yolov8 项目根目录下运行此脚本，"
        "并且当前环境已正确安装项目依赖。原始错误：\n" + str(e)
    )


class YOLOLocalTestGUI:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("YOLO 本地测试工具")
        self.master.geometry("780x540")
        self.master.minsize(740, 520)

        self.model = None
        self.is_running = False
        self.last_output_dir = None

        self.device = self._get_device()
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

        self._build_ui()
        self._log("程序已启动。")
        self._log(f"项目目录：{PROJECT_ROOT}")
        self._log(f"权重路径：{WEIGHTS_PATH}")
        self._log(f"输出目录：{OUTPUT_ROOT}")
        self._log(f"设备：{self.device}")
        self._log(f"图片/文件夹参数：conf={IMAGE_CONF}, imgsz={IMGSZ}")
        self._log(f"视频参数：conf={VIDEO_CONF}, imgsz={IMGSZ}")

    def _get_device(self):
        try:
            if torch is not None and torch.cuda.is_available():
                return 0
            return "cpu"
        except Exception:
            return "cpu"

    def _build_ui(self):
        main = ttk.Frame(self.master, padding=16)
        main.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            main,
            text="YOLO 本地测试工具",
            font=("Microsoft YaHei", 16, "bold")
        )
        title.pack(anchor="w", pady=(0, 10))

        info_frame = ttk.LabelFrame(main, text="当前配置", padding=12)
        info_frame.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(info_frame, text=f"权重文件：{WEIGHTS_PATH}").pack(anchor="w", pady=2)
        ttk.Label(info_frame, text=f"图片/文件夹 conf：{IMAGE_CONF}").pack(anchor="w", pady=2)
        ttk.Label(info_frame, text=f"视频 conf：{VIDEO_CONF}").pack(anchor="w", pady=2)
        ttk.Label(info_frame, text=f"输入尺寸 imgsz：{IMGSZ}").pack(anchor="w", pady=2)
        ttk.Label(info_frame, text=f"设备：{self.device}").pack(anchor="w", pady=2)
        ttk.Label(info_frame, text=f"输出根目录：{OUTPUT_ROOT}").pack(anchor="w", pady=2)

        select_frame = ttk.LabelFrame(main, text="选择测试输入", padding=12)
        select_frame.pack(fill=tk.X, pady=(0, 12))

        btn_frame = ttk.Frame(select_frame)
        btn_frame.pack(fill=tk.X)

        self.btn_image = ttk.Button(
            btn_frame, text="选择单张图片", command=self.select_image
        )
        self.btn_image.pack(side=tk.LEFT, padx=(0, 10), pady=4)

        self.btn_folder = ttk.Button(
            btn_frame, text="选择图片文件夹", command=self.select_folder
        )
        self.btn_folder.pack(side=tk.LEFT, padx=(0, 10), pady=4)

        self.btn_video = ttk.Button(
            btn_frame, text="选择视频文件", command=self.select_video
        )
        self.btn_video.pack(side=tk.LEFT, padx=(0, 10), pady=4)

        self.btn_open_output = ttk.Button(
            btn_frame, text="打开输出目录", command=self.open_output_dir
        )
        self.btn_open_output.pack(side=tk.LEFT, padx=(0, 10), pady=4)

        self.btn_exit = ttk.Button(
            btn_frame, text="退出", command=self.master.destroy
        )
        self.btn_exit.pack(side=tk.RIGHT, pady=4)

        status_frame = ttk.LabelFrame(main, text="运行日志", padding=12)
        status_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            status_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        bottom_frame = ttk.Frame(main)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_var = tk.StringVar(value="状态：空闲")
        self.status_label = ttk.Label(bottom_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT)

    def _set_buttons_state(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.btn_image.config(state=state)
        self.btn_folder.config(state=state)
        self.btn_video.config(state=state)
        self.btn_open_output.config(state=state)

    def _log(self, message: str):
        def _append():
            now = datetime.now().strftime("%H:%M:%S")
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"[{now}] {message}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        self.master.after(0, _append)

    def _set_status(self, message: str):
        self.master.after(0, lambda: self.status_var.set(f"状态：{message}"))

    def _load_model(self):
        if self.model is None:
            if not os.path.exists(WEIGHTS_PATH):
                raise FileNotFoundError(f"权重文件不存在：{WEIGHTS_PATH}")
            self._log("正在加载模型，请稍候...")
            self.model = YOLO(WEIGHTS_PATH)
            self._log("模型加载完成。")

    def select_image(self):
        if self.is_running:
            return
        path = filedialog.askopenfilename(
            title="选择单张图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("所有文件", "*.*")
            ]
        )
        if path:
            self.start_predict(path, "image")

    def select_folder(self):
        if self.is_running:
            return
        path = filedialog.askdirectory(title="选择图片文件夹")
        if path:
            self.start_predict(path, "folder")

    def select_video(self):
        if self.is_running:
            return
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.mpeg *.mpg"),
                ("所有文件", "*.*")
            ]
        )
        if path:
            self.start_predict(path, "video")

    def start_predict(self, source_path: str, source_type: str):
        self.is_running = True
        self._set_buttons_state(False)
        self._set_status("正在运行")
        self._log("=" * 70)
        self._log(f"开始测试，输入类型：{source_type}")
        self._log(f"输入路径：{source_path}")

        thread = threading.Thread(
            target=self._predict_worker,
            args=(source_path, source_type),
            daemon=True
        )
        thread.start()

    def _predict_worker(self, source_path: str, source_type: str):
        try:
            self._load_model()

            run_name = f"{source_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            save_dir = OUTPUT_ROOT / run_name

            if source_type == "video":
                current_conf = VIDEO_CONF
                show_labels = False
                show_conf = False
            else:
                # image 和 folder 都按图片逻辑处理
                current_conf = IMAGE_CONF
                show_labels = True
                show_conf = True

            self._log("开始推理...")
            self._log(f"本次输出目录：{save_dir}")
            self._log(f"本次置信度：{current_conf}")
            self._log(
                f"显示设置：show_boxes=True, show_labels={show_labels}, show_conf={show_conf}"
            )

            # 使用 stream=True，避免视频长时一次性占满内存
            results = self.model.predict(
                source=source_path,
                conf=current_conf,
                imgsz=IMGSZ,
                device=self.device,
                save=True,
                project=str(OUTPUT_ROOT),
                name=run_name,
                exist_ok=True,
                stream=True,
                verbose=True,
                show_boxes=True,
                show_labels=show_labels,
                show_conf=show_conf
            )

            result_count = 0
            actual_save_dir = None

            for r in results:
                result_count += 1
                if actual_save_dir is None:
                    try:
                        actual_save_dir = Path(r.save_dir)
                    except Exception:
                        actual_save_dir = save_dir

                if result_count == 1:
                    self._log("已开始生成结果文件...")
                elif result_count % 20 == 0:
                    self._log(f"当前已处理：{result_count}")

            if actual_save_dir is None:
                actual_save_dir = save_dir

            self.last_output_dir = actual_save_dir

            self._log(f"推理完成，共处理：{result_count}")
            self._log(f"结果已保存到：{actual_save_dir}")
            self._set_status("运行完成")

            self._open_dir(actual_save_dir)

            self.master.after(
                0,
                lambda: messagebox.showinfo(
                    "完成",
                    f"推理完成！\n\n"
                    f"输入类型：{source_type}\n"
                    f"处理数量：{result_count}\n"
                    f"本次置信度：{current_conf}\n"
                    f"输出目录：\n{actual_save_dir}"
                )
            )

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            detail = traceback.format_exc()
            self._log("运行失败。")
            self._log(err_msg)
            self._log(detail)
            self._set_status("运行失败")

            self.master.after(
                0,
                lambda: messagebox.showerror(
                    "错误",
                    f"运行失败：\n{err_msg}"
                )
            )

        finally:
            self.is_running = False
            self.master.after(0, lambda: self._set_buttons_state(True))

    def _open_dir(self, path: Path):
        try:
            if os.name == "nt":
                os.startfile(str(path))
            else:
                self._log(f"当前系统不是 Windows，请手动打开目录：{path}")
        except Exception as e:
            self._log(f"自动打开输出目录失败：{e}")

    def open_output_dir(self):
        target = self.last_output_dir if self.last_output_dir else OUTPUT_ROOT
        self._open_dir(target)


def main():
    root = tk.Tk()

    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass

    app = YOLOLocalTestGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()