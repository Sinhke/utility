import glob
import logging
import os
import click
from utility.image_prediction import get_initial_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--image_dir", "-v", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    "-o",
    type=str,
    help="Output dir where CVAT importable annotations will be saved.",
)
@click.option("--model_path", "-m", type=click.Path(exists=True))
@click.option(
    "--conf",
    "-c",
    type=click.FloatRange(0.0, 1.0),
    default=0.5,
    help="Minimum confidence score for detections to be kept.",
)
def create_cvat_importable_annotations(image_dir, out_dir, model_path, conf=0.5):
    """This CLI runs YOLO inference on images inside the image_dir and output result into out_dir.
    It will create folder structure compatible with CVAT. You can export this output dir into CVAT
    so initial prediction will be set.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Add base files
    with open(os.path.join(out_dir, "obj.data"), "w") as fp:
        fp.write(
            "classes = 1\ntrain = data/train.txt\n\nnames = data/obj.names\nbackup = backup/"
        )
    with open(os.path.join(out_dir, "obj.names"), "w") as fp:
        fp.write("shrimp")

    images = sorted(
        glob.glob(f"{image_dir.rstrip('/')}/*.png")
        + glob.glob(f"{image_dir.rstrip('/')}/*.PNG")
    )

    logger.info(f"Found {len(images)} images in {image_dir}")
    with open(os.path.join(out_dir, "train.txt"), "w") as fp:
        for image_file in images:
            fp.write(f"data/obj_train_data/{os.path.basename(image_file)}\n")

    get_initial_prediction(
        images=images, model_path=model_path, out_dir=out_dir, conf=conf
    )


if __name__ == "__main__":
    create_cvat_importable_annotations()
