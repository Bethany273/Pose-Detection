from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


import cv2

img = cv2.imread("pose.jpg")  # Changed to match your image file
if img is None:
    print("Error: Could not read image file 'pose.jpg'. Please check if the file exists.")
    exit(1)
    
cv2.imshow("Input Image", img)
cv2.waitKey(1)  # Brief pause to show image

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# Check if model file exists
model_path = 'pose_landmarker_full.task'
import os
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found. Please make sure it's in the same directory.")
    cv2.destroyAllWindows()
    exit(1)

try:
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
except Exception as e:
    print(f"Error loading the model: {e}")
    cv2.destroyAllWindows()
    exit(1)

# STEP 3: Load the input image.
try:
    image = mp.Image.create_from_file("pose.jpg")  # Using the same image file as before
except Exception as e:
    print(f"Error loading image: {e}")
    cv2.destroyAllWindows()
    exit(1)

# STEP 4: Detect pose landmarks from the input image.
try:
    detection_result = detector.detect(image)
    if not detection_result.pose_landmarks:
        print("No pose landmarks detected in the image. Please try an image with a clearly visible person.")
        cv2.destroyAllWindows()
        exit(1)
except Exception as e:
    print(f"Error detecting pose: {e}")
    cv2.destroyAllWindows()
    exit(1)

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imshow("Pose Detection", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
cv2.imshow("Segmentation Mask", visualized_mask.astype(np.uint8))

# Wait for a key press and then close all windows
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()  # Clean up the windows
