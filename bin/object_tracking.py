import os
import click
import pandas as pd
from ultralytics import YOLO


@click.command()
@click.option("--input_video", "-v", type=click.Path(exists=True))
@click.option("--model_path", "-m", type=click.Path(exists=True))
@click.option(
    "--save_dir",
    type=click.Path(exists=True),
    required=True,
    help="Save the tracking video to this directory.",
)
@click.option(
    "--confidence-threshold",
    "-c",
    type=click.FloatRange(0.0, 1.0),
    default=0.5,
    help="Minimum confidence score for detections to be kept.",
)
@click.option(
    "--tracker",
    "-tr",
    type=str,
    default="bytetrack.yaml",
    help="Tracking algo to use",
)
@click.option(
    "--max_frames",
    "-mf",
    type=int,
    default=-1,
    help="Maximum frames to process. -1 for all frame.",
)
@click.option(
    "--device",
    "-d",
    type=str,
    default="cuda",
    help="Device to run the model on. Default is cuda.",
)
def process_video(
    input_video,
    model_path,
    save_dir: str,
    confidence_threshold,
    tracker: str = "bytesort.yaml",
    max_frames: int = -1,
    device: str = "cuda",
):
    model = YOLO(model=model_path)
    detection_result = os.path.join(
        save_dir, f"{os.path.basename(input_video).removesuffix('.mp4')}_detection.csv"
    )
    tracking_parameters = dict(
        source=input_video,
        conf=confidence_threshold,
        save=True,
        stream=True,
        verbose=False,
        project=f"{save_dir}",
        name=f"{os.path.basename(input_video).split('.')[0]}",
        save_dir=save_dir,
        tracker=tracker,
        device=device,
        line_width=1,
    )

    results = model.track(**tracking_parameters)  # Default tracker is ByteTrack
    # results is a generator and need to realized in order to save the video

    bbox_pos = []
    for idx, result in enumerate(results):
        if max_frames != -1 and idx >= max_frames:
            break

        for bbox in result.obb.xyxyxyxy:
            frame_and_coord = [idx]
            frame_and_coord = frame_and_coord + bbox.tolist()
            bbox_pos.append(frame_and_coord)

    bbox_df = pd.DataFrame(bbox_pos, columns=["frame", "x", "y", "width", "height"])
    bbox_df.to_csv(detection_result, index=False)


if __name__ == "__main__":
    process_video()
