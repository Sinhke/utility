# Utility repo for processing video and image annotation data

## Setting up the environment on GCP

* Start the persistent GCP instance from [google console](TBD).
* Run tmux session. 
* Inside tmux session, run `jupyter lab`. This will launch a JupyterLab notebook server and you can access it from 
`http://<IP ADDRESS OF THE INSTANCE>:8888`. 

## How to install utility repo
```
# install poetry if not installed
> pip install poetry
# clone utility repo and cd into it
> poetry install
```

## If you want to install it in a venv
If you want to install it in a venv, you can do the following:
```
> poetry shell 
> poetry install

```

## Available executables
### `sample_video` = Sample images from video
```
Usage: sample_video [OPTIONS]
Randomly sample images from video.

Options:
  -i, --input_video PATH
  -o, --output_dir TEXT      Output directory
  -m, --image_count INTEGER  Number of images to sample from the video. Use -1
                             for all frames to be outputted.
  --help                     Show this message and exit.
```
---
### `prepare_dataset` = Prepare YOLO training dataset
```
Usage: prepare_dataset [OPTIONS]

  This CLI tool prepares the training dataset for YOLO training. It takes the
  path to the images and labels directory. Image names and label names should
  match. It splits the dataset into train, test, and validation sets and
  creates a configuration YAML file for the training.

Options:
  --image-path DIRECTORY          Path to the directory containing images.
                                  [required]
  --label-path DIRECTORY          Path to the directory containing labels.
                                  [required]
  -o, --out_training_data_dir DIRECTORY
                                  Output directory to store the training
                                  dataset.  [required]
  --config-name TEXT              Name of the configuration YAML file to be
                                  created.  [required]
  --config-template FILE          Path to the YAML configuration template.
                                  [required]
  --help                          Show this message and exit.
```
---
### `train_yolo` = Train YOLO model
```
Usage: train_yolo [OPTIONS]

  Training YOLO v11 Fine tune the latest YOLO v11 or resume training of a given
  model. Use `-obb.pt` for Oriented Bounding Box model.


Options:
  --config_yaml TEXT      Path to the config YAML file.  [required]
  --epochs INTEGER        Number of epochs to train the model.  [required]
  --model_path TEXT       Path to the model file to resume training.
  --verbose               Enable verbose mode.
  --resume_training TEXT  Path to the model to resume training.
  --device TEXT           Device to train the model on. Default is cuda.
  --help                  Show this message and exit.
```
---
### `yolo_inference` 
```
Usage: yolo_inference [OPTIONS]

  This CLI runs YOLO inference on images inside the image_dir and output
  result into out_dir. It will create folder structure compatible with CVAT.
  You can export this output dir into CVAT so initial prediction will be set.

Options:
  -v, --image_dir PATH
  -o, --out_dir TEXT      Output dir where CVAT importable annotations will be
                          saved.
  -m, --model_path PATH
  -c, --conf FLOAT RANGE  Minimum confidence score for detections to be kept.
                          [0.0<=x<=1.0]
  --help                  Show this message and exit.
```
---
### `yolo_track` = Run tracking using existing model on input video
```
Usage: yolo_track [OPTIONS]

Options:
  -v, --input_video PATH
  -m, --model_path PATH
  --save_dir PATH                 Save the tracking video to this directory.
                                  [required]
  -c, --confidence-threshold FLOAT RANGE
                                  Minimum confidence score for detections to
                                  be kept.  [0.0<=x<=1.0]
  -tr, --tracker TEXT             Tracking algo to use
  -mf, --max_frames INTEGER       Maximum frames to process. -1 for all frame.
  -d, --device TEXT               Device to run the model on. Default is cuda. 
                                  Use cpu if there is no GPU. 
                                  Use mps if you have access to Apple Silicon.
  --help                          Show this message and exit.
```
---
### [DEPRECATED] `yolo_track_stream` = Run tracking in realtime and shows the tracked video.
```
Usage: yolo_track_stream [OPTIONS]

  Processes a video using a provided model, filtering detections below a
  confidence threshold.

  Args:     input_video (str): Path to the input video file.     model_path
  (str): Path to the model file.     confidence_threshold (float): Minimum
  confidence score for detections.

Options:
  -v, --input_video PATH
  -m, --model_path PATH
  -c, --confidence-threshold FLOAT RANGE
                                  Minimum confidence score for detections to
                                  be kept.  [0.0<=x<=1.0]
  -tr, --tracker TEXT             Tracking algo to use
  -d, --device TEXT               Device to run the model on. Default is cuda.
  --help                          Show this message and exit.
```
---
### `yolo_object_extraction` = Extract objects from images
```
Usage: yolo_object_extraction [OPTIONS] COMMAND [ARGS]...


  Object detection and extraction utilities

Options:
  --help  Show this message and exit.

Commands:
  draw-blob-detection  Run blob detection on image and save visualization
  object-extraction    Run object detection and extract images
```

---
### `yolo_object_extraction object-extraction` = Run object detection and extract images
```
Usage: object_extraction [OPTIONS]

Options:
  --image_path TEXT   Path to input image  [required]
  --output_dir TEXT   Directory to save extracted objects  [required]
  --model_path TEXT   Path to YOLO model weights  [required]
  --confidence FLOAT  Confidence threshold for detection (default: 0.5)
  --scale FLOAT       Scale factor for enlarging detected objects; this add
                      more leeway to the bounding box (default: 1.3)
  --save_extracted_image  Save extracted images (default: False)
  --help              Show this message and exit.
```
---
### `yolo_object_extraction draw-blob-detection` = Draw blob detection on image
```
Usage: yolo_object_extraction draw-blob-detection [OPTIONS]

  Run blob detection on image and save visualization

Options:
  --image_path TEXT        Path to input image  [required]
  --output_path TEXT       Path to save output image with detected blobs
                           [required]
  --min_threshold INTEGER  Minimum threshold for blob detection (default: 10)
  --max_threshold INTEGER  Maximum threshold for blob detection (default: 200)
  --min_area FLOAT         Minimum area of blob (default: 100)
  --max_area FLOAT         Maximum area of blob (default: 5000)
  --min_circularity FLOAT  Minimum circularity of blob (default: 0.1)
  --min_convexity FLOAT    Minimum convexity of blob (default: 0.87)
  --min_inertia FLOAT      Minimum inertia ratio of blob (default: 0.01)
  --help                   Show this message and exit.
```
---
### `auto_annotate` = Automatically annotate images
```
Usage: auto_annotate [OPTIONS]

  Run auto-annotation on images and optionally upload to GCS

  Args:     image_dir: Directory containing images to annotate     model_path:
  Path to YOLO model weights     output_dir: Directory to save annotation
  outputs     out_gcs_path: GCS path to upload results (if None, skip upload)

Options:
  --image-dir TEXT      Directory containing images to annotate  [required]
  --model-path TEXT     Path to YOLO model weights  [required]
  --output-dir TEXT     Directory to save annotation outputs  [required]
  --out-gcs-path TEXT   Optional GCS path to upload results
  --conf FLOAT          Confidence threshold for YOLO model
  --max-images INTEGER  Maximum number of images to annotate. -1 for all images.
  --help                Show this message and exit.
```
Example:
```
> auto_annotate --image-dir <IMAGE_DIR> --model-path <MODEL_PATH> --output-dir <OUTPUT_DIR> --out-gcs-path <GCS_PATH>
```

---
### `detect_shrimp_gut` = Detect shrimp gut and eyes
```
Usage: detect_shrimp_gut [OPTIONS]

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
        max_distance_ratio (float): Maximum distance ratio from gut to eyes. Default is 30% of the image.
Options:
  --image-dir PATH            Path to input directory  [required]
  --output-dir PATH           Path to output directory  [required]
  --min-percentile FLOAT      Minimum percentile for thresholding. Default is
                              5. This is needed to remove the background.
  --min-area-ratio FLOAT      Minimum area ratio for blob detection. Default
                              is 0.001.
  --max-area-ratio FLOAT      Maximum area ratio for blob detection. Default
                              is 0.3. Max liver size is 30% of the image.
  --max-angle FLOAT           Maximum angle between potential eyes. Default is
                              100.
  --max-distance-ratio FLOAT  Maximum distance ratio from gut to eyes. Default
                              is 30% of the image.
  --help                      Show this message and exit.
```
Example:
```
> detect_shrimp_gut --image-dir <IMAGE_DIR> --output-dir <OUTPUT_DIR>
```

---