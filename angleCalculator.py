from RingBuffer import RingBuffer
import time
import math
import numpy as np


class angleCalculator:
    def __init__(self):

        self.start_time = time.time()  # Get the current time
        self.current_time = 0.0
        self.bufferSize = 5

        self.angleBuffer = RingBuffer(self.bufferSize)
        self.leftHsBuffer = RingBuffer(self.bufferSize)
        self.rightHsBuffer = RingBuffer(self.bufferSize)

    def get_angle(self, *, lm_array, which_angle: str, buffer: bool):
        '''
        :param which_angle: What angle do you want? possible values are
        chest_to_ground (average of left and right)
        shin_to_ground (left, right)
        hip --> upper leg to chest(left, right)
        '''
        if buffer:
            return Exception("buffer not implemented, but should calculate this with median")
        else:
            match which_angle:
                case "chest_to_ground":
                    return self.knee_angle(lm_array)
                case "shin_to_ground":
                    return
                case "hip":
                    return

    def knee_angle(self, lm_arr):
        x1, y1 = lm_arr[24].x, lm_arr[24].y
        x2, y2 = lm_arr[26].x, lm_arr[26].y
        x3, y3 = lm_arr[27].x, lm_arr[27].y

        A = (x2 - x1, y2 - y1)
        B = (x2 - x3, y2 - y3)

        dot_product = A[0] * B[0] + A[1] * B[1]
        magnitude_A = math.sqrt(A[0] ** 2 + A[1] ** 2)
        magnitude_B = math.sqrt(B[0] ** 2 + B[1] ** 2)

        theta = math.acos(dot_product / (magnitude_A * magnitude_B))
        angle_degrees = theta * 180 / math.pi

        return angle_degrees

    def chest_angle(self, lm_arr):
        x1, y1 = lm_arr[12].x, lm_arr[12].y
        x2, y2 = lm_arr[24].x, lm_arr[24].y
        x3, y3 = lm_arr[26].x, lm_arr[26].y

        A = (x2 - x1, y2 - y1)
        B = (x2 - x3, y2 - y3)

        dot_product = A[0] * B[0] + A[1] * B[1]
        magnitude_A = math.sqrt(A[0] ** 2 + A[1] ** 2)
        magnitude_B = math.sqrt(B[0] ** 2 + B[1] ** 2)

        theta = math.acos(dot_product / (magnitude_A * magnitude_B))
        angle_degrees = theta * 180 / math.pi

        return angle_degrees

    def two_point_angle(self, lm_arr, pointA, pointB):
        left_shoulder = np.array([lm_arr[11].x, lm_arr[11].y])
        right_shoulder = np.array([lm_arr[12].x, lm_arr[12].y])
        left_hip = np.array([lm_arr[23].x, lm_arr[23].y])
        right_hip = np.array([lm_arr[24].x, lm_arr[24].y])

        left_hs = (left_hip - left_shoulder)
        right_hs = (right_hip - right_shoulder)

        angle = ((np.arctan2(left_hs[1], left_hs[0]) + np.arctan2(right_hs[1], right_hs[0])) / 2)
        return angle, left_hs, right_hs