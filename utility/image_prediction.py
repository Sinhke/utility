import glob
import math
import os
import shutil
import tempfile
from collections import defaultdict

import cv2
from ultralytics import YOLO


def get_initial_prediction(images, model_path, conf=0.5, device="cpu", persist=False):
    model = YOLO(model_path)
    # Run batched inference on a list of images
    results = model.track(
        images, conf=conf, stream=True, device=device, persist=persist
    )  # return a list of Results objects
    realized_result = {}
    # Process results list
    for img_file, result in zip(images, results):
        realized_result[img_file] = result

    return realized_result


def create_cvat_importable_annotations(
    image_dir, model_path, outdir, conf=0.5, label_name="shrimp"
):
    # Add base files
    with open(os.path.join(outdir, "obj.data"), "w") as fp:
        fp.write(
            "classes = 1\ntrain = data/train.txt\n\nnames = data/obj.names\nbackup = backup/"
        )
    with open(os.path.join(outdir, "obj.names"), "w") as fp:
        fp.write(label_name)

    images = sorted(glob.glob(f"{image_dir}*.png") + glob.glob(f"{image_dir}*.PNG"))

    with open(os.path.join(outdir, "train.txt")) as fp:
        for image_file in images:
            fp.write(f"data/obj_train_data/{os.path.basename(image_file)}\n")

    get_initial_prediction(
        images=images, model_path=model_path, outdir=outdir, conf=conf
    )


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


def get_object_speeds(results):
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
        lambda x: get_speed_color(x["mean"]), axis=1
    )
    return obj_dist_delta_summary


def get_speed_color(speed):
    if speed <= 5:
        return 6  # red
    elif speed <= 10:
        return 7  # yellow
    elif speed > 10:
        return 13  # green

    return 0  # blue


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


def generate_speed_color_coded_video(results: dict, output_video_path: str):
    obj_dist_delta_summary = get_object_speeds(results)
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
                line_width=4,
                custom_color_ids=speed_color,
            )

        create_video_from_images(temp_dir, output_video_path)
