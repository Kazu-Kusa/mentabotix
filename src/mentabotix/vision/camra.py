import warnings
from time import time
from typing import Tuple, Optional, List

import cv2


class Camera(object):

    def __init__(self, device_id: int = 0):
        # 使用 cv2.VideoCapture(0) 创建视频捕获对象，从默认摄像头捕获视频。
        self._frame_center: Optional[Tuple[int, int]] = None
        self._origin_fps: Optional[int] = None
        self._origin_height: Optional[float] = None
        self._origin_width: Optional[float] = None
        self._camera: Optional[cv2.VideoCapture] = None
        self.open_camera(device_id)

    def open_camera(self, device_id):
        """
        open the cam with self-check
        Args:
            device_id:

        Returns:

        """
        self._camera = cv2.VideoCapture(device_id)
        read_status, _ = self._camera.read()
        if read_status:
            self._origin_width: int = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._origin_height: int = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._origin_fps: int = int(self._camera.get(cv2.CAP_PROP_FPS))
            self._frame_center: Tuple = (int(self._origin_width / 2), int(self._origin_height / 2))
            print(f"CAMERA RESOLUTION：{int(self._origin_width)}x{int(self._origin_height)}\n"
                  f"CAMERA FPS: [{self._origin_fps}]\n"
                  f"CAM CENTER: [{self._frame_center}]")
        else:
            warnings.warn('########CAN\'T GET VIDEO########\n'
                          'please check if the camera is attached!')

    def close_camera(self):
        """
        release the cam
        Returns:

        """
        self._camera.release()
        self._camera = None

    @property
    def origin_width(self):
        """
        the origin width of the cam
        Returns:

        """
        return self._origin_width

    @property
    def origin_height(self):
        """
        the origin height of the cam
        Returns:

        """
        return self._origin_height

    @property
    def origin_fps(self):
        """
        the fps of the cam
        Returns:

        """
        return self._origin_fps

    @property
    def frame_center(self) -> Tuple[int, int]:
        """
        the pixel of the frame center
        Returns:

        """
        return self._frame_center

    def _update_cam_center(self) -> None:
        width = self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._frame_center = (int(width / 2), int(height / 2))

    def set_cam_resolution(self, new_width: Optional[int] = None, new_height: Optional[int] = None,
                           resolution_multiplier: Optional[float] = None) -> None:
        if not self._camera.isOpened():
            warnings.warn("##CAN'T CHANGE the RESOLUTION##\n"
                          "because the camera is NOT opened\n\n")
            return
        assert (new_width is not None and new_height is not None) or (
                resolution_multiplier is not None), 'Please specify the resolution params'
        if resolution_multiplier:
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(resolution_multiplier * self._origin_width))
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(resolution_multiplier * self._origin_height))
        else:
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)
        self._update_cam_center()

    @property
    def camera_device(self) -> cv2.VideoCapture:
        """
        the device instance
        Returns:

        """
        return self._camera

    def test_frame_time(self, test_frames_count: int = 600) -> float:
        """
        test the frame time on the given count and return the average value of it
        :param test_frames_count:
        :return:
        """
        from timeit import repeat
        from numpy import mean, std
        durations: List[float] = repeat(stmt=self._camera.read, number=1, repeat=test_frames_count)
        hall_duration: float = sum(durations)
        average_duration: float = float(mean(hall_duration))
        std_error = std(a=durations, ddof=1)
        print("Frame Time Test Results: \n"
              f"\tRunning on [{test_frames_count}] frame updates\n"
              f"\tTotal Time Cost: [{hall_duration}]\n"
              f"\tAverage Frame time: [{average_duration}]\n"
              f"\tStd Error: [{std_error}]\n")
        return average_duration

    def record_video(self, save_path: str, video_length: float):
        end_time = time() + video_length

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        writer = cv2.VideoWriter(save_path,
                                 fourcc,
                                 self._origin_fps,
                                 (int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        while time() < end_time:
            _, frame = self._camera.read()
            writer.write(frame)
        writer.release()

    def take_a_picture(self, save_path: str, to_gray: bool = True):
        """

        Args:
            save_path:
            to_gray:

        Returns:

        """

        _, frame = self._camera.read()
        if to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(save_path, frame)
