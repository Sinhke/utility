import glob
import math
import os
import shutil
import tempfile
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from ultralytics import YOLO
import yaml


def get_obb_coordinates(bboxes):
    """
    Extract oriented bounding box coordinates from model predictions.

    Args:
        bboxes: List of bounding box predictions from YOLO model, containing coordinates
               in the format of [x1,y1, x2,y2, x3,y3, x4,y4]

    Returns:
        list: List of numpy arrays containing the coordinates of oriented bounding boxes,
              where each array has shape (4,2) representing 4 corner points
    """
    bbox_pos = []
    for bbox in bboxes:
        rectangle = bbox.numpy().reshape((-1, 2))
        bbox_pos.append(rectangle)
    print(f"Found {len(bbox_pos)} bounding boxes")
    return bbox_pos


def get_bbox_coordinates(bboxes, width, height):
    """
    Extract bounding box coordinates from model predictions.

    Args:
        bboxes: List of bounding box predictions from YOLO model, containing coordinates
               in the format of [xcenter, ycenter, xwidth, ywidth]
        width: Width of the image
        height: Height of the image

    Returns:
        list: List of numpy arrays containing the coordinates of bounding boxes,
              where each array has shape (4,2) representing 4 corner points
    """
    bbox_pos = []
    for bbox in bboxes:
        xcenter, ycenter, xwidth, ywidth = bbox.tolist()
        rel_xcenter = xcenter / width
        rel_ycenter = ycenter / height
        rel_width = xwidth / width
        rel_height = ywidth / height
        bbox_pos.append([rel_xcenter, rel_ycenter, rel_width, rel_height])
    return bbox_pos


def get_result_coordinates(results):
    """
    Extract bounding box coordinates from model predictions.

    Args:
        results: List of model predictions from YOLO tracking results

    Returns:
        list: List of numpy arrays containing the coordinates of bounding boxes,
              where each array has shape (4,2) representing 4 corner points
    """
    bbox_pos_list = []
    for result in results:
        if result.obb.xyxyxyxyn is not None:
            bbox_pos = get_obb_coordinates(result.obb.xyxyxyxyn)
        elif result.boxes.xywh is not None:
            bbox_pos = get_bbox_coordinates(
                result.boxes.xywh, result.width, result.height
            )
        bbox_pos_list.append(bbox_pos)

    return bbox_pos_list


def get_initial_prediction(
    images, model_path, conf=0.5, device="cuda", persist=False, predict=False
):
    model = YOLO(model_path)
    # Run batched inference on a list of images
    if predict:
        results = model.predict(images, conf=conf, device=device)
    else:
        results = model.track(
            images, conf=conf, device=device, persist=persist, stream=True
        )
    realized_result = {}
    bbox_coordinates = get_result_coordinates(results)
    # Process results list
    for img_file, result in zip(images, bbox_coordinates):
        # result should contain a list of coordinates for each object in the image
        realized_result[img_file] = result

    return realized_result


def create_cvat_importable_annotations(
    image_dir, model_path, outdir, conf=0.5, label_name="shrimp", max_images=None
):
    images = sorted(
        glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.PNG"))
    )
    if max_images != -1:
        images = images[:max_images]

    logger.info(f"Annotating {len(images)} images")

    os.makedirs(outdir, exist_ok=True)

    base_train_str = "Train"
    # create data.yaml file
    data_yaml = {
        "train": "images/train",
        "names": {0: label_name},
        "path": ".",
    }

    with open(os.path.join(outdir, "data.yaml"), "w") as fp:
        yaml.dump(data_yaml, fp)

    # Move images to images/train directory
    image_outdir = os.path.join(outdir, "images/train")
    os.makedirs(image_outdir, exist_ok=True)
    for image in images:
        shutil.copy(image, os.path.join(image_outdir, os.path.basename(image)))

    realized_coordinates = get_initial_prediction(
        images=images, model_path=model_path, conf=conf
    )
    # Write labels to labels/train directory
    labels_outdir = os.path.join(outdir, "labels/train")
    os.makedirs(labels_outdir, exist_ok=True)
    for img_file, result in realized_coordinates.items():
        label_file = os.path.join(
            labels_outdir, os.path.basename(img_file).replace(".png", ".txt")
        )
        with open(label_file, "w") as fp:
            for box_coords in result:
                coord_str = ""
                for coord in box_coords:
                    coord_str += f"{coord[0]} {coord[1]} "
                fp.write(f"0 {coord_str}\n")

    # Zip the outdir in its parent directory
    parent_dir = os.path.dirname(outdir)
    outdir_name = os.path.basename(outdir)
    zip_file_name = os.path.join(parent_dir, outdir_name)
    shutil.make_archive(
        zip_file_name,
        "zip",
        parent_dir,
        outdir_name,
    )
    zip_file_name = zip_file_name + ".zip"
    logger.info(f"Created cvat importable annotations at {zip_file_name}")
    return zip_file_name


def show_tracked_video(images, bboxes):
    cv2.startWindowThread()
    tracking_frame = defaultdict(list)
    for idx, frame_bbox in enumerate(bboxes):
        img_file = images[idx]
        basename = os.path.basename(img_file)

        frame = cv2.imread(img_file)

        width, height, _ = frame.shape
        for idx, bbox in enumerate(frame_bbox):
            xcenter, ycenter, xwidth, ywidth = [int(elem) for elem in bbox.tolist()]
            x1 = xcenter - xwidth // 2
            y1 = ycenter - ywidth // 2
            cv2.rectangle(frame, (x1, y1), (x1 + xwidth, y1 + ywidth), (255, 0, 0), 2)
            tracking_frame[idx].append([xcenter, ycenter])
            for path_idx, curr_line_center in enumerate(tracking_frame[idx]):
                cv2.circle(frame, curr_line_center, 5, (0, 0, 255), -1)
                prev_line_center = tuple(tracking_frame[idx][path_idx - 1])
                if (
                    path_idx > 0
                    and math.sqrt(
                        (prev_line_center[0] - curr_line_center[0]) ** 2
                        + (prev_line_center[1] - curr_line_center[1]) ** 2
                    )
                    < 50
                ):
                    cv2.line(frame, prev_line_center, curr_line_center, (0, 0, 255), 5)
            if len(tracking_frame[idx]) > 10:
                tracking_frame[idx].pop(0)

        if cv2.waitKey(33) & 0xFF == ord(
            "q"
        ):  # Wait for specified ms or until 'q' is pressed
            break

        cv2.imshow("frame", frame)
        idx += 1

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def get_object_speeds(results, speed_thresholds):
    """Receives results dictionary from YOLO model tracking result.

    Args:
        results (_type_):

    Returns:
        _type_: _description_
    """
    obj_locations = defaultdict(list)
    obj_frame_locations = []
    for _, result in results.items():
        obj_frame_locations.append(result)
        if result.boxes.id is None:
            continue
        for idx, obj_id in enumerate(result.boxes.id):
            obj_locations[int(obj_id)].append(result.boxes.xywh[idx].tolist())

    obj_dist_delta = {}
    for obj_id, locations in obj_locations.items():
        loc_df = pd.DataFrame(locations)
        loc_df.columns = ["xcenter", "ycenter", "width", "height"]
        loc_df["x_delta"] = loc_df["xcenter"].diff()
        loc_df["y_delta"] = loc_df["ycenter"].diff()
        loc_df["dist_delta"] = np.sqrt(loc_df["x_delta"] ** 2 + loc_df["y_delta"] ** 2)
        obj_dist_delta[obj_id] = loc_df["dist_delta"]

    obj_dist_delta = pd.DataFrame.from_dict(obj_dist_delta)
    obj_dist_delta_summary = obj_dist_delta.describe().T
    obj_dist_delta_summary["color"] = obj_dist_delta_summary.apply(
        lambda x: get_speed_color(x["mean"], speed_thresholds), axis=1
    )
    return obj_dist_delta_summary


def get_speed_color(speed, speed_thresholds):
    if speed <= speed_thresholds["slow"]:
        return 6  # red not moving much
    elif speed <= speed_thresholds["fast"]:
        return 2  # white normal movement
    elif speed > speed_thresholds["fast"]:
        return 0  # blue fast

    return 3  # aquamarine for unknown


class TemporaryDirectory:
    def __enter__(self):
        self.dir = tempfile.mkdtemp()
        return self.dir

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.dir)


def create_video_from_images(frame_folder, output_video_path):
    frames = [f for f in os.listdir(frame_folder) if f.endswith(".png")]
    frames.sort(
        key=lambda f: int(f.split("_")[-1].split(".")[0])
    )  # Sorting numerically based on frame number

    # Read the first frame to get frame size
    frame = cv2.imread(os.path.join(frame_folder, frames[0]))
    height, width, layers = frame.shape

    out = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
    )

    for frame in frames:
        img = cv2.imread(os.path.join(frame_folder, frame))
        out.write(img)  # Write each frame to the video

    out.release()
    cv2.destroyAllWindows()


def generate_speed_color_coded_video(
    results: dict,
    output_video_path: str,
    speed_thresholds: dict = {"slow": 5, "fast": 10},
):
    obj_dist_delta_summary = get_object_speeds(results, speed_thresholds)
    speed_color = obj_dist_delta_summary["color"].to_dict()
    # Write annotated images into temp directory
    with TemporaryDirectory() as temp_dir:
        for frame_idx, (fname, result) in enumerate(results.items()):
            basename = os.path.basename(fname)
            annot_fname = os.path.join(
                temp_dir, basename.replace(".png", f"_frame_{frame_idx:04d}.png")
            )
            result.plot(
                save=True,
                filename=annot_fname,
                line_width=2,
                custom_color_ids=speed_color,
            )

        create_video_from_images(temp_dir, output_video_path)
    return obj_dist_delta_summary
