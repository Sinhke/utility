import os
from ultralytics import YOLO
import click


@click.command()
@click.option("--input_video", "-v", type=click.Path(exists=True))
@click.option("--model_path", "-m", type=click.Path(exists=True))
@click.option(
    "--confidence-threshold",
    "-c",
    type=click.FloatRange(0.0, 1.0),
    default=0.5,
    help="Minimum confidence score for detections to be kept.",
)
@click.option("--show", is_flag=True, default=False, help="Show video")
@click.option("--save_video", type=str, default=None, help="Save the tracking video")
def process_video(
    input_video,
    model_path,
    confidence_threshold,
    show: bool = False,
    save_video: str | None = None,
):
    """
    Processes a video using a provided model, filtering detections below a confidence threshold.

    Args:
        input_video (str): Path to the input video file.
        model_path (str): Path to the model file.
        confidence_threshold (float): Minimum confidence score for detections.
    """

    # Load the model
    model = YOLO(model=model_path)
    tracking_parameters = dict(
        source=input_video,
        conf=confidence_threshold,
        show=show,
        stream=True,
    )
    if save_video:
        project, fname = os.path.split(save_video)
        tracking_parameters["save"] = True
        tracking_parameters["project"] = project
        tracking_parameters["name"] = f"{os.path.basename(fname).split('.')[0]}"

    model.track(**tracking_parameters)  # Default tracker is ByteTrack


if __name__ == "__main__":
    process_video()
