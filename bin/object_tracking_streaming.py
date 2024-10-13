import os
import click
from ultralytics import YOLO


@click.command()
@click.option("--input_video", "-v", type=click.Path(exists=True))
@click.option("--model_path", "-m", type=click.Path(exists=True))
@click.option("--save_video", type=str, required=True, help="Save the tracking video")
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
def process_video(
    input_video,
    model_path,
    save_video: str,
    confidence_threshold,
    tracker: str = "bytesort.yaml",
):
    model = YOLO(model=model_path)
    project, fname = os.path.split(save_video)
    detection_result = f"{fname.removesuffix('.mp4')}_detection.csv"
    tracking_parameters = dict(
        source=input_video,
        conf=confidence_threshold,
        save=True,
        stream=True,
        verbose=False,
        project=project,
        name=f"{os.path.basename(fname).split('.')[0]}",
        tracker=tracker,
        device="mps",
        line_width=1,
    )

    results = model.track(**tracking_parameters)  # Default tracker is ByteTrack
    # results is a generator and need to realized in order to save the video

    for elem in results:
        bboxes = [result.boxes.xywh for result in results]
        print(type(elem), elem)
        # detection_result


if __name__ == "__main__":
    process_video()
