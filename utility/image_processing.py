import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def get_pixels(img_file: str) -> np.array:
    """Get np.array pixel values

    Args:
        img_file (str):

    Returns:
        np.array:
    """
    img = Image.open(img_file)
    pixels = np.asarray(img)
    return pixels


def get_rect_box_yolo(
    rel_x: float,
    rel_y: float,
    rel_bbox_width: float,
    rel_bbox_height: float,
    width: int,
    height: int,
) -> patches.Rectangle:
    """Get YOLO rectangle bounding box

    Args:
        rel_x (float):
        rel_y (float):
        rel_bbox_width (float):
        rel_bbox_height (float):
        width (int):
        height (int):

    Returns:
        patches.Rectangle:
    """
    xmin = width * (rel_x - rel_bbox_width / 2)
    ymin = height * (rel_y - rel_bbox_height / 2)
    patch_width = rel_bbox_width * width
    patch_height = rel_bbox_height * height
    # Create a Rectangle patch
    rect_patch = patches.Rectangle(
        (xmin, ymin),
        patch_width,
        patch_height,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )

    return rect_patch


def get_yolo_style_bbox_rect(
    bbox_coord_file: str, width: int, height: int
) -> list[patches.Rectangle]:
    """Get YOLO bounding box from file

    Args:
        bbox_coord_file (str):
        width (int):
        height (int):

    Returns:
        list[patches.Rectangle]:
    """
    bbox_coord_df = pd.read_csv(bbox_coord_file, sep=" ", header=None)
    bbox_coord_df.columns = ["label", "x", "y", "w", "h"]

    rect_patch_list = []
    for _, row in bbox_coord_df.iterrows():
        rel_x, rel_y, rel_bbox_width, rel_bbox_height = (
            row["x"],
            row["y"],
            row["w"],
            row["h"],
        )
        rect_patch_list.append(
            get_rect_box_yolo(
                rel_x, rel_y, rel_bbox_width, rel_bbox_height, width, height
            )
        )
    return rect_patch_list


def get_rect_box(
    rel_xmin: float,
    rel_xmax: float,
    rel_ymin: float,
    rel_ymax: float,
    width: int,
    height: int,
) -> patches.Rectangle:
    """_summary_

    Args:
        rel_xmin (float): _description_
        rel_xmax (float): _description_
        rel_ymin (float): _description_
        rel_ymax (float): _description_
        width (int): _description_
        height (int): _description_

    Returns:
        patches.Rectangle: _description_
    """
    xmin = width * rel_xmin
    ymin = height * rel_ymin
    patch_width = (rel_xmax - rel_xmin) * width
    patch_height = (rel_ymax - rel_ymin) * height
    # Create a Rectangle patch
    rect_patch = patches.Rectangle(
        (xmin, ymin),
        patch_width,
        patch_height,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )

    return rect_patch


def show_img_with_bounding_box(img_file: str, bbox_coord_file: str) -> None:
    """Show an image with bounding box

    Args:
        img_file (str):
        bbox_coord_file (str):
    """
    pixels = get_pixels(img_file)
    height, width, _ = pixels.shape
    bbox_coord_df = pd.read_csv(bbox_coord_file, sep=" ", header=None)

    rect_patch_list = []
    for _, row in bbox_coord_df.iterrows():
        rel_xmin, rel_xmax, rel_ymin, rel_ymax = (
            row[1],
            row[2],
            row[3],
            row[4],
        )
        rect_patch_list.append(
            get_rect_box_yolo(rel_xmin, rel_xmax, rel_ymin, rel_ymax, width, height)
        )

    _, ax = plt.subplots()
    ax.imshow(pixels)
    # Add the patch to the Axes
    for rect in rect_patch_list:
        ax.add_patch(rect)
    plt.show()
