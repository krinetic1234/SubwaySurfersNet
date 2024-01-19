import cv2 as cv
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sys
import argparse
import copy
import pyautogui
import datetime

# Faster pyautogui
pyautogui.PAUSE = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.FAILSAFE = False

# Key actions
KEY_LEFT = "left"
KEY_RIGHT = "right"
KEY_UP = "up"
KEY_DOWN = "down"


def log_info(message):
    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{date_time}] {message}")


def get_args():
    """
    Get arguments passed to script
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0, help="Camera device")
    parser.add_argument("--width", help="Frame width", type=int, default=960)
    parser.add_argument("--height", help="Frame height", type=int, default=540)
    parser.add_argument(
        "--keypoint_score", type=float, default=0.3, help="Keypoint score threshold"
    )

    args = parser.parse_args()

    return args


def run_inference(model, input_size, image):
    """
    Run inference on image to get keypoints and scores
    """
    image_height, image_width = image.shape[0], image.shape[1]

    # Preprocess image
    input_image = cv.resize(image, dsize=(input_size, input_size))  # Resize
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)  # BGR to RGB
    input_image = input_image.reshape(-1, input_size, input_size, 3)
    input_image = tf.cast(input_image, dtype=tf.int32)  # Cast to int32

    # Run model
    outputs = model(input_image)

    keypoints_with_scores = outputs["output_0"].numpy()
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    # Keypoints, Scores
    keypoints = []
    scores = []

    """ 17 keypoints
    (in the order of: [nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle]
    
    See: https://tfhub.dev/google/movenet/singlepose/lightning/4
    """
    for idx in [
        0,
        1,
        2,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    ]:  # Draw keypoints except ears
        keypoint_x = int(keypoints_with_scores[idx][1] * image_width)
        keypoint_y = int(keypoints_with_scores[idx][0] * image_height)
        score = keypoints_with_scores[idx][2]

        keypoints.append((keypoint_x, keypoint_y))
        scores.append(score)

    return keypoints, scores


def draw_keypoints(image, keypoints, scores, keypoint_score_th, toggle_keypress):
    circle_color_outer = (255, 255, 255)  # White
    circle_color_inner = (0, 0, 255)  # Red
    line_skeleton_color = (0, 255, 0)  # Green
    movement_text_color = (0, 0, 0)  # Blue

    # Draw line right eye to nose (Left eye on frame because of mirror)
    index1, index2 = 1, 0
    if scores[index1] > keypoint_score_th and scores[index2] > keypoint_score_th:
        point1 = keypoints[index1]
        point2 = keypoints[index2]
        cv.line(image, point1, point2, line_skeleton_color, 2)

    # Draw line left eye to nose (Right eye on frame because of mirror)
    index1, index2 = 2, 0
    if scores[index1] > keypoint_score_th and scores[index2] > keypoint_score_th:
        point1 = keypoints[index1]
        point2 = keypoints[index2]
        cv.line(image, point1, point2, line_skeleton_color, 2)

    # Draw line left shoulder to right shoulder (Left shoulder on frame because of mirror)
    index1, index2 = 3, 4
    if scores[index1] > keypoint_score_th and scores[index2] > keypoint_score_th:
        point1 = keypoints[index1]
        point2 = keypoints[index2]
        cv.line(image, point1, point2, line_skeleton_color, 2)

    # Draw line left shoulder to left elbow (Right shoulder on frame because of mirror)
    index1, index2 = 3, 5
    if scores[index1] > keypoint_score_th and scores[index2] > keypoint_score_th:
        point1 = keypoints[index1]
        point2 = keypoints[index2]
        cv.line(image, point1, point2, line_skeleton_color, 2)

    # Draw line left elbow to left wrist (Right elbow on frame because of mirror)
    index1, index2 = 5, 7
    if scores[index1] > keypoint_score_th and scores[index2] > keypoint_score_th:
        point1 = keypoints[index1]
        point2 = keypoints[index2]
        cv.line(image, point1, point2, line_skeleton_color, 2)

    # Draw line right shoulder to right elbow (Left shoulder on frame because of mirror)
    index1, index2 = 4, 6
    if scores[index1] > keypoint_score_th and scores[index2] > keypoint_score_th:
        point1 = keypoints[index1]
        point2 = keypoints[index2]
        cv.line(image, point1, point2, line_skeleton_color, 2)

    # Draw line right elbow to right wrist (Left elbow on frame because of mirror)
    index1, index2 = 6, 8
    if scores[index1] > keypoint_score_th and scores[index2] > keypoint_score_th:
        point1 = keypoints[index1]
        point2 = keypoints[index2]
        cv.line(image, point1, point2, line_skeleton_color, 2)

    # Draw line left shoulder to left hip (Right shoulder on frame because of mirror)
    index1, index2 = 3, 9
    if scores[index1] > keypoint_score_th and scores[index2] > keypoint_score_th:
        point1 = keypoints[index1]
        point2 = keypoints[index2]
        cv.line(image, point1, point2, line_skeleton_color, 2)

    # Draw line right shoulder to right hip (Left shoulder on frame because of mirror)
    index1, index2 = 4, 10
    if scores[index1] > keypoint_score_th and scores[index2] > keypoint_score_th:
        point1 = keypoints[index1]
        point2 = keypoints[index2]
        cv.line(image, point1, point2, line_skeleton_color, 2)

    # Draw line left hip to right hip (Left hip on frame because of mirror)
    index1, index2 = 9, 10
    if scores[index1] > keypoint_score_th and scores[index2] > keypoint_score_th:
        point1 = keypoints[index1]
        point2 = keypoints[index2]
        cv.line(image, point1, point2, line_skeleton_color, 2)

    # Draw line left hip to left knee (Right hip on frame because of mirror)
    index1, index2 = 9, 11
    if scores[index1] > keypoint_score_th and scores[index2] > keypoint_score_th:
        point1 = keypoints[index1]
        point2 = keypoints[index2]
        cv.line(image, point1, point2, line_skeleton_color, 2)

    # Draw line left knee to left ankle (Right knee on frame because of mirror)
    index1, index2 = 11, 13
    if scores[index1] > keypoint_score_th and scores[index2] > keypoint_score_th:
        point1 = keypoints[index1]
        point2 = keypoints[index2]
        cv.line(image, point1, point2, line_skeleton_color, 2)

    # Draw line right hip to right knee (Left hip on frame because of mirror)
    index1, index2 = 10, 12
    if scores[index1] > keypoint_score_th and scores[index2] > keypoint_score_th:
        point1 = keypoints[index1]
        point2 = keypoints[index2]
        cv.line(image, point1, point2, line_skeleton_color, 2)

    # Draw line right knee to right ankle (Left knee on frame because of mirror)
    index1, index2 = 12, 14
    if scores[index1] > keypoint_score_th and scores[index2] > keypoint_score_th:
        point1 = keypoints[index1]
        point2 = keypoints[index2]
        cv.line(image, point1, point2, line_skeleton_color, 2)

    # Draw keypoints (Circle)
    for keypoint, score in zip(keypoints, scores):
        if score > keypoint_score_th:
            cv.circle(image, keypoint, 6, circle_color_outer, -1)  # White outer circle
            cv.circle(image, keypoint, 3, circle_color_inner, -1)  # Red inner circle

    # Right Movement (Right hip the right half of the frame, because it's mirrored)
    if keypoints[10][0] > image.shape[1] / 2:
        movement = "Right"
    elif keypoints[9][0] < image.shape[1] / 2:
        movement = "Left"
    else:
        movement = "Standing"

    # Jump Movement (Left shoulder, right shoulder in the top 2/5 of the frame)
    if (
        keypoints[3][1] < image.shape[0] / 5 * 2
        and keypoints[4][1] < image.shape[0] / 5 * 2
    ):
        movement = "Jump"

    # Crouch Movement (Nose in the bottom 2/5 of the frame)
    if keypoints[0][1] > image.shape[0] / 5 * 2:
        movement = "Crouch"

    # ESC to exit text
    cv.putText(
        image,
        "Press ESC to exit",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    # Space to toggle keypress text
    keypress_state = "Enabled" if toggle_keypress else "Disabled"
    cv.putText(
        image,
        "Press SPACE to toggle keypress [{}]".format(keypress_state),
        (10, 60),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    # Movement Label text (Bottom left)
    cv.putText(
        image,
        "Movement:",
        (10, 90),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    # Movement Value text (Bottom left)
    cv.putText(
        image,
        movement,
        (100, 90),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    # Draw horizontal line (2/5 of the frame)
    cv.line(
        image,
        (0, int(image.shape[0] / 5 * 2)),
        (image.shape[1], int(image.shape[0] / 5 * 2)),
        (255, 255, 255),
        2,
    )

    # Draw vertical line (Center)
    cv.line(
        image,
        (int(image.shape[1] / 2), 0),
        (int(image.shape[1] / 2), image.shape[0]),
        (255, 255, 255),
        2,
    )

    return image, movement


def main():
    """
    Main function
    """
    # Get arguments
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    keypoint_score_th = args.keypoint_score

    # Open camera
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model (MoveNet Lightning), use pre-trained model from TensorFlow Hub
    input_size = 192
    model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    model_load = hub.load(model_url)
    model = model_load.signatures["serving_default"]

    # Enable Keypress
    enable_keypress = True

    # Last movement
    last_movement = "Standing"

    # Exit if camera not opened
    if not cap.isOpened():
        sys.exit("Failed to open camera!")

    # Loop frames
    while True:
        # Read frame
        ret, frame = cap.read()

        # Exit if frame not read
        if not ret:
            sys.exit("Failed to read frame!")

        # Flip frame
        frame = cv.flip(frame, 1)

        # Run inference
        keypoints, scores = run_inference(model, input_size, frame)

        # Copy frame
        frame_copy = copy.deepcopy(frame)

        # Draw frame with keypoints
        draw_frame, frame_movement = draw_keypoints(
            frame_copy, keypoints, scores, keypoint_score_th, enable_keypress
        )

        # Movement keypress
        if frame_movement == "Right":
            if last_movement != "Right":
                if enable_keypress:
                    pyautogui.press(KEY_RIGHT)
                    log_info("Keypress: Right")
                last_movement = "Right"
                log_info("Movement: Right")

        if frame_movement == "Left":
            if last_movement != "Left":
                if enable_keypress:
                    pyautogui.press(KEY_LEFT)
                    log_info("Keypress: Left")
                last_movement = "Left"
                log_info("Movement: Left")

        if frame_movement == "Jump":
            if last_movement != "Jump":
                if enable_keypress:
                    pyautogui.press(KEY_UP)
                    log_info("Keypress: Up")
                last_movement = "Jump"
                log_info("Movement: Jump")

        if frame_movement == "Crouch":
            if last_movement != "Crouch":
                if enable_keypress:
                    pyautogui.press(KEY_DOWN)
                    log_info("Keypress: Down")
                last_movement = "Crouch"
                log_info("Movement: Crouch")

        if frame_movement == "Standing":
            if last_movement != "Standing":
                if last_movement == "Right":
                    if enable_keypress:
                        pyautogui.press("left")
                        log_info("Keypress: Left")

                if last_movement == "Left":
                    if enable_keypress:
                        pyautogui.press("right")
                        log_info("Keypress: Right")

                last_movement = "Standing"
                log_info("Movement: Standing")

        # Opencv KeyPress
        cvkey = cv.waitKey(1)
        if cvkey:
            # ESC to exit
            if cvkey == 27:
                break

            # Space to toggle keypress
            if cvkey == 32:
                if enable_keypress:
                    enable_keypress = False
                    log_info("Keypress: Disabled")
                else:
                    enable_keypress = True
                    log_info("Keypress: Enabled")

        # Display frame
        cv.imshow("frame", draw_frame)

    # Release camera
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()