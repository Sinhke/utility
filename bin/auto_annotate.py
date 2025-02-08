import os
import subprocess

import click

from utility.image_prediction import *


@click.command()
@click.option(
    "--image-dir", required=True, help="Directory containing images to annotate"
)
@click.option("--model-path", required=True, help="Path to YOLO model weights")
@click.option(
    "--output-dir", required=True, help="Directory to save annotation outputs"
)
@click.option(
    "--out-gcs-path", default=None, help="Optional GCS path to upload results"
)
@click.option("--conf", default=0.5, help="Confidence threshold for YOLO model")
@click.option(
    "--max-images",
    default=-1,
    type=int,
    help="Maximum number of images to annotate. -1 for all images.",
)
def auto_annotate(image_dir, model_path, output_dir, out_gcs_path, conf, max_images):
    """
    Run auto-annotation on images and optionally upload to GCS

    Args:
        image_dir: Directory containing images to annotate
        model_path: Path to YOLO model weights
        output_dir: Directory to save annotation outputs
        out_gcs_path: GCS path to upload results (if None, skip upload)
    """

    # model_path = "/home/jay_sinhke_com/obb_models/bloo_model1/train/weights/best.pt" # this is the best model so far
    # image_dir = "/mnt/big_disk/obb_training_data/beta2_training_data/images/Jay"
    # outdir = "/mnt/big_disk/obb_training_data/beta2_training_data/yolo_annotations/Jay/cvat_importable_best_model"

    if os.path.exists(output_dir):
        raise ValueError(f"Output directory {output_dir} already exists")

    out_zipfile = create_cvat_importable_annotations(
        image_dir=image_dir,
        model_path=model_path,
        outdir=output_dir,
        conf=conf,
        max_images=max_images,
    )

    if out_gcs_path:
        subprocess.run(["gsutil", "cp", out_zipfile, out_gcs_path], check=True)


if __name__ == "__main__":
    auto_annotate()
