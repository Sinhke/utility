import glob
import logging
import os
import click
from ultralytics import YOLO
from utility.image_prediction import get_result_coordinates, get_initial_prediction
import shutil
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TODO FIX THIS and merge with get_initial_prediction in image_prediction.py
def write_initial_predictions(images, model_path, out_dir, conf=0.5):
    os.makedirs(os.path.join(out_dir, "obj_train_data"), exist_ok=True)
    obj_inference_dir = os.path.join(out_dir, "obj_train_data")
    model = YOLO(model_path)
    # Run batched inference on a list of images
    results = model.predict(images, conf=conf)  # return a list of Results objects

    bbox_coordinates = get_result_coordinates(results, use_normalized_coordinates=True)

    # Process results list
    for img_file, bbox_coord in zip(images, bbox_coordinates):
        # Copy the image file to the obj_train_data directory
        shutil.copy(img_file, obj_inference_dir)
        basename = os.path.basename(img_file)
        with open(
            os.path.join(obj_inference_dir, f"{basename.removesuffix('.png')}.txt"), "w"
        ) as fp:
            for bbox in bbox_coord:
                line = " ".join([str(x) for x in bbox])
                fp.write(f"0 {line}\n")


@click.command()
@click.option("--image_dir", "-v", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    "-o",
    type=click.Path(exists=True),
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

    write_initial_predictions(
        images=images, model_path=model_path, out_dir=out_dir, conf=conf
    )


if __name__ == "__main__":
    create_cvat_importable_annotations()
