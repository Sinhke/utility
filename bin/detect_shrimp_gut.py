import glob
import os
import shutil
import click
from loguru import logger

import cv2
import numpy as np
import pandas as pd


def calculate_angle(p1, p2, p3):
    """
    Calculate the angle between three points.
    Points should be tuples or arrays of (x,y) coordinates.
    Returns angle in degrees.
    """
    # Get vectors from p2 to p1 and p2 to p3
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    # Calculate angle using dot product formula
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    # Convert to degrees
    return np.degrees(angle)


def get_potential_eyes(df, gut, max_angle=100):
    already_calculated = set()
    potential_eyes = []
    for i, row in df.iterrows():
        for j, row2 in df.iterrows():
            if i == j:
                continue
            if tuple(sorted([i, j])) in already_calculated:
                continue

            already_calculated.add(tuple(sorted([i, j])))

            angle = calculate_angle(
                (row["x"], row["y"]), (gut["x"], gut["y"]), (row2["x"], row2["y"])
            )
            # print(f"Angle between point {i} - gut - point {j}: {angle:.2f} degrees")
            if angle <= max_angle:
                potential_eyes.append((i, j))
            else:
                continue

    return potential_eyes


def shrimp_eye_gut_detection(
    image_path,
    min_percentile=5,
    min_area_ratio=0.001,
    max_area_ratio=0.3,
    max_angle=100,
    max_distance_ratio=0.3,
):
    logger.info(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # guts and eyes should be black and should be lower end of the color value
    low = np.percentile(img_gray, min_percentile)
    img_thresholded = 255 * (img_gray <= low).astype(np.uint8)

    area = img_thresholded.shape[0] * img_thresholded.shape[1]
    min_area = int(area * min_area_ratio)
    max_area = int(area * max_area_ratio)

    logger.info(f"min_area: {min_area}, max_area: {max_area}")
    # Create blob detector
    # For irregular polygons, we need to use MSER detector instead of blob detector
    detector = cv2.MSER_create(min_area=min_area, max_area=max_area)

    keypoints = []
    try:
        # Detect blobs
        keypoints = detector.detect(img_thresholded)
        logger.info(f"Detected {len(keypoints)} blobs")
        if len(keypoints) <= 1:
            logger.warning("NO KEYPOINTS FOUND; Bad image")
            return pd.DataFrame(columns=["x", "y", "diameter"]), None

        keypoints = sorted(keypoints, key=lambda x: x.size, reverse=True)
        gut_keypoint = keypoints[0]  # First keypoint (gut) in different color

        keypoints_df = pd.DataFrame(
            [(*elem.pt, elem.size) for elem in keypoints], columns=["x", "y", "diameter"]
        )
        keypoints_df = keypoints_df.sort_values(by="diameter", ascending=False).reset_index(
            drop=True
        )

        gut = keypoints_df.iloc[0]
        keypoints_df = keypoints_df.drop(0)
        keypoints_df["distance_to_gut"] = keypoints_df.apply(
            lambda x: np.sqrt((x["x"] - gut["x"]) ** 2 + (x["y"] - gut["y"]) ** 2),
            axis=1,
        )
        # display(keypoints_df)
        length_side = max(img_gray.shape)
        # eyes cannot be too far from the gut
        max_distance = length_side * max_distance_ratio
        keypoints_df = keypoints_df[keypoints_df["distance_to_gut"] <= max_distance]
        potential_eyes = get_potential_eyes(keypoints_df, gut, max_angle=max_angle)

        if len(potential_eyes) == 0:
            logger.warning("NO POTENTIAL EYES FOUND; Bad image")
            return pd.DataFrame(columns=["x", "y", "diameter"]), None

    except Exception as e:
        logger.error(f"Error detecting keypoints: {e}")
        return pd.DataFrame(columns=["x", "y", "diameter"]), None

    return keypoints_df, gut_keypoint


@click.command()
@click.option(
    "--image-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to input directory",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(exists=False),
    help="Path to output directory",
)
@click.option(
    "--min-percentile",
    default=5,
    type=float,
    help="Minimum percentile for thresholding. Default is 5. This is needed to remove the background.",
)
@click.option(
    "--min-area-ratio",
    default=0.001,
    type=float,
    help="Minimum area ratio for blob detection. Default is 0.001.",
)
@click.option(
    "--max-area-ratio",
    default=0.3,
    type=float,
    help="Maximum area ratio for blob detection. Default is 0.3. Max liver size is 30% of the image.",
)
@click.option(
    "--max-angle",
    default=100,
    type=float,
    help="Maximum angle between potential eyes. Default is 100.",
)
@click.option(
    "--max-distance-ratio",
    default=0.3,
    type=float,
    help="Maximum distance ratio from gut to eyes. Default is 30% of the image.",
)
def classify_shrimp_images(
    image_dir,
    output_dir,
    min_percentile,
    min_area_ratio,
    max_area_ratio,
    max_angle,
    max_distance_ratio,
):
    """
    Process images in an image_dir for shrimp eye and gut detection.
    Outputs good and bad images in output_dir.

    This function does the following:
    - Remove background from the image by thresholding at a certain percentile; eye and gut are usually darker than the background.
    - Detects the blobs in the image.
    - Sorts the blobs by diameter.
    - Selects the largest blob as the gut.
    - Calculates the distance of the other blobs to the gut.
    - Selects the blobs within a certain distance as potential eyes.
    - Calculates the angle between the gut and the potential eyes.
    - Selects the potential eyes within a certain angle.
    - Saves the good and bad images in the output_dir

    Args:
        image_dir (str): Path to input directory
        output_dir (str): Path to output directory
        min_percentile (float): Minimum percentile for thresholding. Default is 5. This is needed to remove the background.
        min_area_ratio (float): Minimum area ratio for blob detection. Default is 0.001.
        max_area_ratio (float): Maximum area ratio for blob detection. Default is 0.3. Max liver size is 30% of the image.
        max_angle (float): Maximum angle between potential eyes. Default is 100.
    """
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    logger.info(f"Found {len(image_files)} images to process")

    bad_image_dir = os.path.join(output_dir, "bad_images")
    os.makedirs(bad_image_dir, exist_ok=True)
    good_image_dir = os.path.join(output_dir, "good_images")
    os.makedirs(good_image_dir, exist_ok=True)

    for image_file in image_files:
        keypoints_df, gut_keypoint = shrimp_eye_gut_detection(
            image_file,
            min_percentile=min_percentile,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            max_angle=max_angle,
            max_distance_ratio=max_distance_ratio,
        )
        if gut_keypoint is None:
            shutil.copy(
                image_file, os.path.join(bad_image_dir, os.path.basename(image_file))
            )
        else:
            img = cv2.imread(image_file)
            img = cv2.drawKeypoints(
                img,
                [gut_keypoint],
                np.array([]),
                (0, 255, 255),
            )
            cv2.imwrite(
                os.path.join(good_image_dir, os.path.basename(image_file)),
                img,
            )
            keypoints_df.loc[len(keypoints_df)] = [gut_keypoint.pt[0], gut_keypoint.pt[1], gut_keypoint.size, 0]
            keypoints_df.sort_values(by="diameter", ascending=False, inplace=True)
            keypoints_df.to_csv(
                os.path.join(good_image_dir, os.path.basename(image_file).replace(".png", ".csv")),
                index=False,
            )


if __name__ == "__main__":
    classify_shrimp_images()
