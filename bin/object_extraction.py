import cv2
import numpy as np
import os
import pandas as pd
from ultralytics import YOLO
import click

from utility.image_prediction import get_rectangle_area, get_result_coordinates


def get_center_point(coords):
    """
    Calculate the center point of a rectangle given its coordinates

    Args:
        coords (list): List of [x,y] coordinates for the rectangle vertices

    Returns:
        list: [x,y] coordinates of the center point
    """
    x_coords = [p[0] for p in coords]
    y_coords = [p[1] for p in coords]
    center_x = sum(x_coords) / len(coords)
    center_y = sum(y_coords) / len(coords)
    return [center_x, center_y]


def scale_rectangle(coords, scale_factor):
    """
    Scale a rectangle from its center point

    Args:
        coords (list): List of [x,y] coordinates for the rectangle vertices
                      in the order: [x1,y1], [x2,y2], [x3,y3], [x4,y4]
        scale_factor (float): Factor to scale the rectangle by (>1 enlarges, <1 shrinks)

    Returns:
        list: New coordinates of the scaled rectangle
    """
    # Validate input
    if len(coords) != 4:
        raise ValueError("Expected 4 coordinates but got {}".format(len(coords)))
    if scale_factor <= 0:
        raise ValueError("Scale factor must be positive")

    # Get center point
    center = get_center_point(coords)

    # Scale each point relative to center
    scaled_coords = []
    for point in coords:
        # Vector from center to point
        vector = [point[0] - center[0], point[1] - center[1]]

        # Scale vector
        scaled_vector = [v * scale_factor for v in vector]

        # New point position
        new_point = [
            int(center[0] + scaled_vector[0]),
            int(center[1] + scaled_vector[1]),
        ]
        scaled_coords.append(new_point)

    return np.array(scaled_coords), center


def order_points(pts):
    """
    Order points in clockwise order starting from top-left
    Args:
        pts: numpy array of shape (4, 2) containing rectangle coordinates
    Returns:
        numpy array of same shape with ordered points
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # Get sum and diff of coordinates
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    # Top-left point will have smallest sum
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have largest sum
    rect[2] = pts[np.argmax(s)]

    # Top-right point will have smallest difference
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left point will have largest difference
    rect[3] = pts[np.argmax(diff)]

    return rect


def get_destination_dimensions(pts):
    """
    Calculate width and height of the rectified image
    Args:
        pts: numpy array of shape (4, 2) containing ordered rectangle coordinates
    Returns:
        tuple of (width, height)
    """
    # Calculate width as max of top and bottom edge lengths
    width_top = np.linalg.norm(pts[1] - pts[0])
    width_bottom = np.linalg.norm(pts[2] - pts[3])
    width = max(int(width_top), int(width_bottom))

    # Calculate height as max of left and right edge lengths
    height_left = np.linalg.norm(pts[3] - pts[0])
    height_right = np.linalg.norm(pts[2] - pts[1])
    height = max(int(height_left), int(height_right))

    return width, height


def rectify_image(image, points):
    """
    Transform a rotated rectangular region in an image to a front-facing view

    Args:
        image: numpy array containing the source image
        points: list of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] defining the rectangle

    Returns:
        numpy array containing the rectified image
    """
    # Convert points to numpy array
    pts = np.array(points, dtype=np.float32)

    # Order points in clockwise order starting from top-left
    rect = order_points(pts)

    # Get dimensions of destination image
    width, height = get_destination_dimensions(rect)

    # Define destination points for perspective transform
    dst = np.array(
        [
            [0, 0],  # Top-left
            [width - 1, 0],  # Top-right
            [width - 1, height - 1],  # Bottom-right
            [0, height - 1],  # Bottom-left
        ],
        dtype=np.float32,
    )

    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)

    # Apply perspective transformation
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped


def crop_image(image, ratio=None, points=None):
    """
    Crop an image array using either a ratio from edges or specific points

    Args:
        image: numpy array of shape (height, width, channels) or (height, width)
        ratio: float between 0 and 1, amount to crop from each edge (optional)
        points: list of [x,y] coordinates defining the rectangle to crop (optional)

    Returns:
        numpy array containing the cropped image
    """
    height, width = image.shape[:2]

    if ratio is not None:
        if not 0 <= ratio < 1:
            raise ValueError("Ratio must be between 0 and 1")

        # Calculate pixels to crop from each edge
        crop_x = int(width * ratio)
        crop_y = int(height * ratio)

        # Crop the image from all edges
        cropped = image[crop_y : height - crop_y, crop_x : width - crop_x]

    elif points is not None:
        # Convert points to numpy array if not already
        pts = np.array(points)

        # Get bounding rectangle
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)

        # Convert to integers
        x_min, y_min = int(x_min), int(y_min)
        x_max, y_max = int(x_max), int(y_max)

        # Ensure coordinates are within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)

        # Crop the image
        cropped = image[y_min:y_max, x_min:x_max]

    else:
        raise ValueError("Either ratio or points must be provided")

    return cropped


def draw_blob_detection(image, outdir):
    """Draw blob detection on an image

    Args:
        image (_type_): _description_
        outdir (_type_): _description_
    """
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    # Convert RGBA to grayscale for blob detection
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)

    # Set up blob detection parameters
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 1
    params.maxThreshold = 150
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 100

    # Create blob detector
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(gray)
    # Draw detected blobs as red circles with smaller radius
    # Use cv2.DRAW_MATCHES_FLAGS_DEFAULT to draw simple circles instead of rich keypoints
    img_with_keypoints = cv2.drawKeypoints(
        gray, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )
    # Convert back to RGBA
    img = img_with_keypoints
    cv2.imwrite(os.path.join(outdir, os.path.basename(image)), img)


def run_prediction_and_extract_image(
    image_path, model_path, output_dir, confidence_threshold=0.5, scale=1.3, save_extracted_image=False
):
    model = YOLO(model=model_path)
    images = [image_path]
    tracking_parameters = dict(
        source=images,
        conf=confidence_threshold,
        device="cpu",
        line_width=1,
    )

    results = model.track(**tracking_parameters)
    # For bounding box extraction, we use the original coordinates
    bbox_coordinates = get_result_coordinates(results, use_normalized_coordinates=False)

    # Process results list
    for img_file, bbox_coord in zip(images, bbox_coordinates):
        basename = os.path.basename(img_file).removesuffix(".png")
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detected_objects = []
        for idx, coord in enumerate(bbox_coord):
            flattened_coord = list(np.array(coord).flatten())
            # Create a mask from the polygon coordinates
            # Enlarge polygon based on scale factor
            # [x1, y1], [x2, y2], [x3, y3], [x4, y4]
            scaled_coord, center = scale_rectangle(coord, scale)
            flattened_coord.append(get_rectangle_area(flattened_coord))
            flattened_coord.extend(center)
            detected_objects.append(flattened_coord)

            if save_extracted_image:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [scaled_coord], 1)

                # Extract the pixels within the polygon using the mask
                masked_img = cv2.bitwise_and(img, img, mask=mask)

                # Get the bounding rectangle of the polygon
                x, y, w, h = cv2.boundingRect(scaled_coord)
                # Crop to just the bounding rectangle area
                cropped = masked_img[y : y + h, x : x + w]
                # Adjust coordinates relative to cropped image by subtracting x,y offset

                cropped_coord = scaled_coord.copy()
                cropped_coord[:, 0] = scaled_coord[:, 0] - x  # Adjust x coordinates
                cropped_coord[:, 1] = scaled_coord[:, 1] - y  # Adjust y coordinates
                try:
                    rectified_cropped = rectify_image(cropped, cropped_coord)
                    rectified_cropped = crop_image(rectified_cropped, ratio=0.03)
                    output_path = os.path.join(output_dir, f"{basename}_obj_{idx}.png")
                    cv2.imwrite(output_path, rectified_cropped)

                except Exception as e:
                    print("Error in rectifying image", e)
        detected_objects = pd.DataFrame(detected_objects)
        detected_objects.columns = [
            "x1",
            "y1",
            "x2",
            "y2",
            "x3",
            "y3",
            "x4",
            "y4",
            "cx",
            "cy",
            "area",
        ]
        detected_objects.index.name = "object_id"
        detected_objects.to_csv(
            os.path.join(output_dir, f"{basename}_obj_coordinates.csv")
        )


@click.command()
@click.option("--image_path", required=True, type=str, help="Path to input image")
@click.option(
    "--output_dir", required=True, type=str, help="Directory to save extracted objects"
)
@click.option(
    "--model_path", required=True, type=str, help="Path to YOLO model weights"
)
@click.option(
    "--confidence",
    default=0.5,
    type=float,
    help="Confidence threshold for detection (default: 0.5)",
)
@click.option(
    "--scale",
    default=1.3,
    type=float,
    help="Scale factor for enlarging detected objects; this add more leeway to the bounding box (default: 1.3)",
)
@click.option(
    "--save_extracted_image",
    is_flag=True,
    default=False,
    help="Save extracted images (default: False)",
)
def object_extraction(image_path, output_dir, model_path, confidence, scale, save_extracted_image=False):
    """Run object detection and extract images"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run prediction and extraction
    run_prediction_and_extract_image(
        image_path=image_path,
        output_dir=output_dir,
        model_path=model_path,
        confidence_threshold=confidence,
        scale=scale,
        save_extracted_image=save_extracted_image,
    )


@click.command()
@click.option("--image_path", required=True, type=str, help="Path to input image")
@click.option(
    "--output_path",
    required=True,
    type=str,
    help="Path to save output image with detected blobs",
)
@click.option(
    "--min_threshold",
    default=10,
    type=int,
    help="Minimum threshold for blob detection (default: 10)",
)
@click.option(
    "--max_threshold",
    default=200,
    type=int,
    help="Maximum threshold for blob detection (default: 200)",
)
@click.option(
    "--min_area",
    default=100,
    type=float,
    help="Minimum area of blob (default: 100)",
)
@click.option(
    "--max_area",
    default=5000,
    type=float,
    help="Maximum area of blob (default: 5000)",
)
@click.option(
    "--min_circularity",
    default=0.1,
    type=float,
    help="Minimum circularity of blob (default: 0.1)",
)
@click.option(
    "--min_convexity",
    default=0.87,
    type=float,
    help="Minimum convexity of blob (default: 0.87)",
)
@click.option(
    "--min_inertia",
    default=0.01,
    type=float,
    help="Minimum inertia ratio of blob (default: 0.01)",
)
def draw_blob_detection(
    image_path,
    output_path,
    min_threshold,
    max_threshold,
    min_area,
    max_area,
    min_circularity,
    min_convexity,
    min_inertia,
):
    """Run blob detection on image and save visualization"""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Set up the detector with parameters
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = min_threshold
    params.maxThreshold = max_threshold

    # Filter by Area
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = min_circularity

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = min_convexity

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = min_inertia

    # Create detector with parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(image)

    # Draw detected blobs as red circles
    image_with_keypoints = cv2.drawKeypoints(
        image,
        keypoints,
        np.array([]),
        (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    # Save the output image
    cv2.imwrite(output_path, image_with_keypoints)
    print(f"Detected {len(keypoints)} blobs")
    print(f"Output saved to {output_path}")


@click.group()
def main():
    """Object detection and extraction utilities"""
    pass


main.add_command(object_extraction)
main.add_command(draw_blob_detection)


if __name__ == "__main__":
    main()
