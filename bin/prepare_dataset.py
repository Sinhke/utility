import os
import click
import yaml
import glob

from sklearn.model_selection import train_test_split
import shutil


# label_path = "/mnt/big_disk/freshwater_test/labels/"
# image_path = "/mnt/big_disk/freshwater_test/images/"


def get_image_and_label_list(image_path, label_path):
    img_list = sorted(glob.glob(os.path.join(image_path, "*.png")))
    label_list = sorted(glob.glob(os.path.join(label_path, "*.txt")))
    img_and_labels_list = []
    for img, label in zip(img_list, label_list):
        assert os.path.basename(img).removesuffix(".png") == os.path.basename(
            label
        ).removesuffix(
            ".txt"
        ), f"Image and Labels not matching in {image_path} and {label_path}"
        img_and_labels_list.append((img, label))
    return img_and_labels_list


@click.command()
@click.option(
    "--image-path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Path to the directory containing images.",
)
@click.option(
    "--label-path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Path to the directory containing labels.",
)
@click.option(
    "-o",
    "--out_training_data_dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Output directory to store the training dataset.",
)
@click.option(
    "--config-name",
    required=True,
    type=str,
    help="Name of the configuration YAML file to be created.",
)
@click.option(
    "--config-template",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the YAML configuration template.",
)
def prepare_training_dataset(
    image_path, label_path, out_training_data_dir, config_name, config_template
):
    img_and_label_list = get_image_and_label_list(image_path, label_path)
    train_data, test_data = train_test_split(
        img_and_label_list, test_size=0.3, random_state=42
    )
    test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=42)
    config_yaml = os.path.join(out_training_data_dir, config_name)

    with open(config_template) as in_fp:
        yaml_config = yaml.safe_load(in_fp)
        yaml_config["path"] = out_training_data_dir
        with open(config_yaml, "w") as out_fp:
            yaml.safe_dump(yaml_config, out_fp)

        for subdir, img_and_label_list in zip(
            ["train", "test", "val"], [train_data, test_data, val_data]
        ):
            images_dir = os.path.join(out_training_data_dir, yaml_config[subdir])
            labels_dir = os.path.join(
                out_training_data_dir, yaml_config[subdir].replace("images", "labels")
            )
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            for img_file, label_file in img_and_label_list:
                shutil.copy(img_file, images_dir)
                shutil.copy(label_file, labels_dir)


if __name__ == "__main__":
    prepare_training_dataset()
