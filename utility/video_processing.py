import os
import cv2 as cv
import pandas as pd
import random


def sample_video(
    video_file: str,
    output_img_dir: str,
    img_number: int = -1,
    crop: list[int] | None = None,
):
    """Sample input video and generate images from it.

    Args:
        video_file (str):
        output_img_dir (str):
        img_number (int, optional): Number of images to create. Defaults to -1 for all frames.
        sampling_level (int, optional): How frequently sample the video. Defaults to 10.
    """
    basename = os.path.basename(video_file).rsplit(".", 1)[0]
    idx = 0
    saved_img_cnt = 0

    cap = cv.VideoCapture(video_file)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    prob = img_number / frame_count
    print(f"Sampling probability={prob} FRAME_COUNT={frame_count}")
    while cap.isOpened():
        ret, frame = cap.read()
        idx += 1
        # if frame is read correctly ret is True
        if not ret:
            break

        if img_number == -1 or (random.random() <= prob):
            frame_name = os.path.join(output_img_dir, f"{basename}_frame{idx:07}.png")
            if crop is not None:
                x1, y1, width, height = crop
                frame = frame[y1 : y1 + height, x1 : x1 + width]

            cv.imwrite(frame_name, frame)
            saved_img_cnt += 1
        if img_number != -1 and saved_img_cnt >= img_number:
            break

    cap.release()


def play_video(video_input, annotation):
    print(f"Showing image annotation {annotation}")
    annotation_df = pd.read_csv(annotation, sep="\t", header=None, dtype=float)
    cap = cv.VideoCapture(video_input)
    frames = []
    idx = 0

    cv.startWindowThread()

    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames.append(gray)
        try:
            x1, y1, width, height = [
                int(val) for val in annotation_df.iloc[idx].tolist()
            ]
            cv.rectangle(gray, (x1, y1), (x1 + width, y1 + height), (255, 0, 0), 2)
        except Exception as e:
            print(x1, y1, width, height, e)
        cv.imshow("frame", gray)
        idx += 1
        if cv.waitKey(1) == ord("q"):
            break
        # if cv.getWindowProperty('frame', cv.WND_PROP_VISIBLE) < 1:
        #     break
    cap.release()
    cv.destroyAllWindows()
    for i in range(10):
        cv.waitKey(1)
