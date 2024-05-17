import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
import math
from RingBuffer import RingBuffer

start_time = time.time()  # Get the current time
condition_met = False  # Initialize the condition flag
current_time = 0.0

squatTrig = False

squatCount = 0

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pygame.init()
cap = cv2.VideoCapture(0)

bufferSize = 5

angleBuffer = RingBuffer(bufferSize)
leftHsBuffer = RingBuffer(bufferSize)
rightHsBuffer = RingBuffer(bufferSize)

def knee_angle():
    x1 = lm_arr[24].x
    y1 = lm_arr[24].y
    x2 = lm_arr[26].x
    y2 = lm_arr[26].y
    x3 = lm_arr[27].x
    y3 = lm_arr[27].y
    # Calculate the vectors
    A = (x2 - x1, y2 - y1)
    B = (x2 - x3, y2 - y3)

    # Calculate the dot product
    dot_product = A[0] * B[0] + A[1] * B[1]

    # Calculate the magnitudes
    magnitude_A = math.sqrt(A[0] ** 2 + A[1] ** 2)
    magnitude_B = math.sqrt(B[0] ** 2 + B[1] ** 2)

    # Calculate the angle in radians
    theta = math.acos(dot_product / (magnitude_A * magnitude_B))

    # Convert the angle to degrees
    angle_degrees = theta * 180 / math.pi

    return angle_degrees

def chest_angle():
    x1 = lm_arr[12].x
    y1 = lm_arr[12].y
    x2 = lm_arr[24].x
    y2 = lm_arr[24].y
    x3 = lm_arr[26].x
    y3 = lm_arr[26].y
    # Calculate the vectors
    A = (x2 - x1, y2 - y1)
    B = (x2 - x3, y2 - y3)

    # Calculate the dot product
    dot_product = A[0] * B[0] + A[1] * B[1]

    # Calculate the magnitudes
    magnitude_A = math.sqrt(A[0] ** 2 + A[1] ** 2)
    magnitude_B = math.sqrt(B[0] ** 2 + B[1] ** 2)

    # Calculate the angle in radians
    theta = math.acos(dot_product / (magnitude_A * magnitude_B))

    # Convert the angle to degrees
    angle_degrees = theta * 180 / math.pi

    return angle_degrees

def absolute_chest_angle():
    # calculate chest angle relative to screen only

    left_shoulder = np.array([lm_arr[11].x, lm_arr[11].y])
    right_shoulder = np.array([lm_arr[12].x, lm_arr[12].y])
    left_hip = np.array([lm_arr[23].x, lm_arr[23].y])
    right_hip = np.array([lm_arr[24].x, lm_arr[24].y])

    # calculate left body angle
    left_hs = (left_shoulder - left_hip)
    right_hs = (right_shoulder - right_hip)

    return (np.arctan2(left_hs[0], left_hs[1]) + np.arctan2(right_hs[0], right_hs[1]) / 2), left_hs, right_hs

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # Flip the image horizontally for a selfie-view display.
    lm = results.pose_landmarks
    if lm is not None:
        lm_arr = lm.landmark

        temp_angle, temp_left_hs, temp_right_hs = absolute_chest_angle()

        angleBuffer.add(temp_angle)
        leftHsBuffer.add(temp_left_hs)
        rightHsBuffer.add(temp_right_hs)

        angle = np.mean(angleBuffer.get())
        left_hs = np.mean(leftHsBuffer.get(), axis=0)
        right_hs = np.mean(rightHsBuffer.get(), axis=0)

        # draw the vector
        angle = 20
        height, width, _ = image.shape
        zero_vector = (int(width/2), int(height/2))
        draw_vector_coords = (zero_vector[0] - int(200 * math.sin(angle)), zero_vector[1] - int(200*math.cos(angle)))
        image = cv2.line(image, zero_vector, (int(width/2) + int(left_hs[0]*200), int(height/2) + int(left_hs[1]*200)), (255, 0, 0), 10)
        image = cv2.line(image, zero_vector, draw_vector_coords, (0, 255, 0), 5)
        image = cv2.line(image, (zero_vector[0] + 100, zero_vector[1]), (int(width/2)-100, int(height/2)), (0, 0, 255), 5)
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

    if cv2.waitKey(5) & 0xFF == 27:
      break
    
pygame.quit()
cap.release