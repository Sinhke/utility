import os
import sys
import click
from loguru import logger
import pandas as pd
from ultralytics import YOLO
from utility.image_prediction import get_result_coordinates


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
@click.option(
    "--debug",
    "-db",
    type=bool,
    default=False,
    help="Debug mode.",
)
def process_video(
    input_video,
    model_path,
    save_dir: str,
    confidence_threshold,
    tracker: str = "bytesort.yaml",
    max_frames: int = -1,
    device: str = "cuda",
    debug: bool = False,
):
    logger.remove()
    if debug:
        logger.add(sys.stdout, level="DEBUG")
        logger.info("Debug mode is enabled. Setting log level to DEBUG.")
    else:
        logger.add(sys.stdout, level="INFO")

    logger.info("Starting object tracking...")

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
        name=f"{os.path.basename(input_video).removesuffix('.mp4')}",
        save_dir=save_dir,
        tracker=tracker,
        device=device,
        line_width=1,
    )

    results = model.track(**tracking_parameters)  # Default tracker is ByteTrack
    bbox_pos = get_result_coordinates(results, max_frames=max_frames)
    converted_bbox_pos = []
    for idx, obj in enumerate(bbox_pos):
        if idx % 100 == 0:
            logger.info(f"Processed {idx} frames")

        obj = [[idx] + elem.reshape(-1).tolist() for elem in obj if len(elem) > 1]
        for elem in obj:
            if len(elem) > 1:
                converted_bbox_pos.append(elem)
    try:
        bbox_df = pd.DataFrame(converted_bbox_pos, columns=["frame", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"])
        bbox_df.to_csv(detection_result, index=False)
    except ValueError as e:
        logger.warning("No bounding boxes found in the video.")


if __name__ == "__main__":
    process_video()
