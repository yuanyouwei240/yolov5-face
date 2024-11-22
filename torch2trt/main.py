import time
import os
import sys
import cv2
import copy
import torch
import argparse
import subprocess

root_path = os.path.dirname(
    os.path.abspath(os.path.dirname(__file__))
)  # 项目根路径：获取当前路径，再上级路径
sys.path.append(root_path)  # 将项目根路径写入系统路径
from utils.general import (
    check_img_size,
    non_max_suppression_face,
    scale_coords,
    xyxy2xywh,
)
from utils.datasets import letterbox
from detect_face import scale_coords_landmarks, show_results
from torch2trt.trt_model import TrtModel

cur_path = os.path.abspath(os.path.dirname(__file__))


def img_process(orgimg, long_side=640, stride_max=32):
    """
    图像预处理
    """
    img0 = copy.deepcopy(orgimg)
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = long_side / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(long_side, s=stride_max)  # check img_size

    img = letterbox(img0, new_shape=imgsz, auto=False)[
        0
    ]  # auto True最小矩形   False固定尺度
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img, orgimg


def img_vis(img, orgimg, pred, vis_thres=0.6):
    """
    预测可视化
    vis_thres: 可视化阈值
    """

    # print('img.shape: ', img.shape)
    # print('orgimg.shape: ', orgimg.shape)

    no_vis_nums = 0
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        ]  # normalization gain landmarks
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = scale_coords_landmarks(
                img.shape[2:], det[:, 5:15], orgimg.shape
            ).round()

            for j in range(det.size()[0]):
                if det[j, 4].cpu().numpy() < vis_thres:
                    no_vis_nums += 1
                    continue

                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:15].view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()
                orgimg = show_results(orgimg, xyxy, conf, landmarks, class_num)

    return orgimg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp_url", type=str, required=True, help="RTSP URL")
    parser.add_argument(
        "--trt_path", type=str, required=True, help="TensorRT model path"
    )
    parser.add_argument(
        "--output_shape",
        type=list,
        default=[1, 25200, 16],
        help="Output shape of the model",
    )
    parser.add_argument(
        "--rtmp_url", type=str, default="rtmp://localhost/live/stream", help="RTMP URL"
    )
    opt = parser.parse_args()

    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TrtModel(opt.trt_path)

    # 打开 RTSP 流
    cap = cv2.VideoCapture(opt.rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        sys.exit()

    # 获取视频流的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 使用 FFmpeg 推送到 RTMP 流
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "ultrafast",
        "-f",
        "flv",
        opt.rtmp_url,
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # 记录推理开始时间
        # inference_start_time = time.time()
        img, orgimg = img_process(frame)
        # 记录推理结束时间
        # inference_end_time = time.time()
        # inference_time = inference_end_time - inference_start_time
        # print(f"Inference time: {inference_time:.4f} seconds")

        pred = model(img.numpy()).reshape(opt.output_shape)  # forward

        # Apply NMS
        pred = non_max_suppression_face(
            torch.from_numpy(pred), conf_thres=0.3, iou_thres=0.5
        )

        # 可视化
        result_img = img_vis(img, orgimg, pred)

        # 计算并显示帧率
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(
            result_img,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # 推送到 RTMP 流
        process.stdin.write(result_img.tobytes())

        # 控制处理速度不超过视频流帧率
        time.sleep(max(1.0 / 25 - elapsed_time, 0))

        # 显示结果
        # cv2.imshow('RTSP Stream', result_img)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    model.destroy()
    process.stdin.close()
    process.wait()
