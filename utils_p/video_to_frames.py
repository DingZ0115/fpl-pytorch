import cv2
import os


def video_to_frames(video_path, output_folder, fps=10):
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Failed to open video.")
        return

    # 获取视频帧率
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # 计算每隔多少帧取一帧
    frame_interval = int(video_fps / fps)

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 循环读取视频帧并保存为图像文件
    frame_count = 0
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 只保留每秒的第 fps 帧
        if frame_count % frame_interval == 0:
            # 将 BGR 图像转换为 RGB 图像
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 保存图像文件
            output_path = os.path.join(output_folder, f"{cnt}.png")
            cv2.imwrite(output_path, rgb_frame)
            cnt += 1

        frame_count += 1

    cap.release()


# 定义输入视频文件路径和输出文件夹路径
video_path = r"C:\Users\dy\Desktop\数据例子\video_0002.mp4"
output_folder = r"\video_0002_image"

# 将视频分割为每秒 10 帧的 RGB 图像
video_to_frames(video_path, output_folder, fps=10)
