"""
Training and fine tuning
"""

from ultralytics import YOLO
import click


@click.command()
@click.option(
    "--config_yaml", required=True, type=str, help="Path to the config YAML file."
)
@click.option(
    "--epochs", required=True, type=int, help="Number of epochs to train the model."
)
@click.option(
    "--model_path",
    required=False,
    type=str,
    help="Path to the model file to resume training.",
)
@click.option("--verbose", is_flag=True, help="Enable verbose mode.")
@click.option(
    "--resume_training",
    required=False,
    type=str,
    help="Path to the model to resume training.",
)
@click.option(
    "--device",
    required=False,
    type=str,
    default="cuda",
    help="Device to train the model on. Default is cuda.",
)
def train_custom_model(
    config_yaml: str,
    epochs: int,
    model_path: str,
    verbose: bool = False,
    resume_training: str = None,
    device: str = "cuda",
):
    """
    Training YOLO model.
    Fine tune the latest YOLO model or resume training of a given model.
    """

    if resume_training:
        model = YOLO(resume_training)
    else:
        model = YOLO(model_path)

    if verbose:
        model.info()

    # Train the model
    model.train(
        data=config_yaml,
        epochs=epochs,
        device=device,
        fraction=0.8,
        verbose=verbose,
        weight_decay=0.05,
        lr0=0.1,
        dropout=0.1,
        # patience=30,  # Number of epochs to wait to stop if no improvement.
        resume=True if resume_training else False,
    )


if __name__ == "__main__":
    train_custom_model()
