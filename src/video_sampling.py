import click
from loguru import logger
from utility.video_processing import sample_video


@click.command()
@click.option("--input_video", "-i", type=click.Path(exists=True))
@click.option("--output_dir", "-o", help="Output directory")
@click.option(
    "--image_count",
    "-m",
    default=100,
    help="Number of images to sample from the video. Use -1 for all frames to be outputted.",
)
def sample_video_for_images(
    input_video: str, output_dir: str, image_count: int
) -> None:
    logger.info(f"Sampling video {input_video} to {output_dir}")
    sample_video(input_video, output_dir, image_count)


if __name__ == "__main__":
    sample_video()
