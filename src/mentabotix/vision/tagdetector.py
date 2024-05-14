"""
apriltag detecting app
"""

from dataclasses import dataclass
from threading import Thread
from time import sleep
from typing import List, Literal, Self, Any, Callable, Optional

import cv2
import numpy as np
from cv2 import Mat, cvtColor, COLOR_RGB2GRAY
from numpy import ndarray, dtype, generic
from numpy.linalg import linalg
from pyapriltags import Detector, Detection

from ..modules.logger import _logger


class TagDetector:
    """
    use cam to detect apriltags
    """

    detector = Detector(
        families="tag36h11",
        nthreads=2,
        quad_decimate=1.0,
        refine_edges=False,
        debug=False,
    )

    @dataclass
    class Config:
        single_tag_mode: bool = True
        resolution_multiplier: float = 0.5
        ordering_method: Literal["nearest", "single"] = "nearest"
        halt_check_interval: float = 0.4
        default_tag_id: int = -1
        error_tag_id: int = -10

    def __init__(self, cam_id: int = 0, resolution_multiplier: Optional[float] = None):
        """

        Args:
            cam_id:
            resolution_multiplier:
        """
        self._camera: cv2.VideoCapture = cv2.VideoCapture(cam_id)
        self.set_cam_resolution_mul(resolution_multiplier or self.Config.resolution_multiplier)

        self._tag_id: int = TagDetector.Config.default_tag_id

        self._continue_detection: bool = False
        self._halt_detection: bool = False

    def open_camera(self, device_id: int = 0) -> Self:
        """
        open the cam with self-check
        Args:
            device_id:

        Returns:

        """
        self._camera = cv2.VideoCapture(device_id)
        read_status, _ = self._camera.read()
        if read_status:
            self._update_cam_center()
            _logger.info(
                f"CAMERA RESOLUTION：{self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}\n"
                f"CAMERA FPS: [{self._camera.get(cv2.CAP_PROP_FPS)}]\n"
                f"CAM CENTER: [{self._frame_center}]"
            )
        else:
            _logger.error("Can't open camera!")
        return self

    def release_camera(self) -> Self:
        """
        Releases the camera resource.

        This method releases the camera resource associated with the current object, ensuring that it is properly freed when no longer needed.

        Returns:
            Self: Returns the calling object itself, supporting chaining calls.
        """
        self._camera.release()  # Release the camera resource
        return self  # Support chaining calls

    def apriltag_detect_start(self):
        """
        Initializes the apriltag detection process. This function runs in a separate thread,
        continuously detecting apriltags from the camera feed and processes them according to the configured method.

        The function starts a loop that fetches frames from the camera, performs apriltag detection on each frame.
        Depending on the configuration, it handles detected tags either by "nearest" (closest tag) or "single" (first detected tag).
        If tags are found, the function updates the internal tag ID based on the selected method.
        """

        # Set the tag detection function
        detector_function = TagDetector.detector.detect

        # Retrieve the camera instance and frame center
        cam = self._camera
        frame_center: ndarray = self._frame_center
        check_interval = self.Config.halt_check_interval

        # Choose the tag handling method based on the configuration
        match self.Config.ordering_method:
            case "nearest":
                # Select the nearest tag handling method
                used_method: Callable[[List[Detection]], int] = lambda tags: min(
                    tags, key=lambda tag: linalg.norm(tag.center - frame_center)
                ).tag_id
            case "single":
                # Select the method to handle the first detected tag
                used_method: Callable[[List[Detection]], int] = lambda tags: tags[0].tag_id
            case _:
                # Raise an error if the configuration method is invalid
                raise ValueError(f"Wrong ordering method! Got {self.Config.ordering_method}")

        def __loop():
            # Main detection loop
            while self._continue_detection:
                # Check if detection should be halted
                if self._halt_detection:
                    _logger.debug("Apriltag detect halted!")
                    sleep(check_interval)
                    continue
                # Read a frame from the camera
                success, frame = cam.read()
                if success:
                    # Convert the frame to grayscale and detect tags
                    gray: Mat | ndarray[Any, dtype[generic]] | ndarray = cvtColor(frame, COLOR_RGB2GRAY)
                    tags: list[Detection] = detector_function(gray, self.Config.single_tag_mode)
                    if tags:
                        # Update the tag ID using the selected method
                        self._tag_id = used_method(tags)
                else:
                    # If the camera read fails, log a critical error and set the error tag ID
                    self._tag_id = TagDetector.Config.error_tag_id
                    _logger.critical("Camera not functional!")
                    return
            # Log when the detection loop stops
            _logger.info("AprilTag detect stopped")

        # Create and start the detection thread
        apriltag_detect = Thread(target=__loop, name="apriltag_detect_Process")
        apriltag_detect.daemon = True
        apriltag_detect.start()
        # Log when the detection is activated
        _logger.info("AprilTag detect Activated")

    def apriltag_detect_stop(self) -> Self:
        """
        Stops the AprilTag detection process by setting `_continue_detection` to False and resetting `_tag_id` to the default value.

        Returns:
            Self: The current instance of the `TagDetector` class.
        """
        self._continue_detection = False
        self._tag_id = TagDetector.Config.default_tag_id
        return self

    def halt_detection(self) -> Self:
        """
        Halts the tag detection process by setting the `_halt_detection` flag to `True` and resetting the `_tag_id` to the default value specified in the `TagDetector.Config` class.

        Returns:
            Self: The current instance of the `TagDetector` class.
        """
        self._halt_detection = True
        self._tag_id = TagDetector.Config.default_tag_id
        return self

    def resume_detection(self) -> Self:
        """
        Resumes the detection process.

        Returns:
            Self: The updated instance of the class.
        """
        self._halt_detection = False
        return self

    @property
    def tag_id(self) -> int:
        """
        :return:  current tag id
        """
        return self._tag_id

    def _update_cam_center(self) -> None:
        self._frame_center = np.array(
            [self._camera.get(cv2.CAP_PROP_FRAME_WIDTH) / 2, self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2]
        )

    def set_cam_resolution_mul(
        self,
        resolution_multiplier: float,
    ) -> Self:
        """
        Set the camera resolution by multiplying the current resolution with the given resolution multiplier.

        Args:
            resolution_multiplier (float): The factor by which the camera resolution should be multiplied.

        Returns:
            Self: The updated instance of the class with the new camera resolution.
        """
        return self.set_cam_resolution(
            self._camera.get(cv2.CAP_PROP_FRAME_WIDTH) * resolution_multiplier,
            self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT) * resolution_multiplier,
        )

    def set_cam_resolution(self, new_width: int | float, new_height: int | float) -> Self:
        """
        Set the resolution of the camera.

        Args:
            new_width (int | float): The new width of the camera resolution.
            new_height (int | float): The new height of the camera resolution.

        Returns:
            Self: The updated instance of the class.
        """
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)
        self._update_cam_center()
        return self

    @property
    def camera_device(self) -> cv2.VideoCapture:
        """
        the device instance
        Returns:

        """
        return self._camera


def test_frame_time(camera: cv2.VideoCapture, test_frames_count: int = 600) -> float:
    """
    test the frame time on the given count and return the average value of it
    :param test_frames_count:
    :return:
    """
    from timeit import repeat
    from numpy import mean, std

    durations: List[float] = repeat(stmt=camera.read, number=1, repeat=test_frames_count)
    hall_duration: float = sum(durations)
    average_duration: float = float(mean(hall_duration))
    std_error = std(a=durations, ddof=1)
    print(
        "Frame Time Test Results: \n"
        f"\tRunning on [{test_frames_count}] frame updates\n"
        f"\tTotal Time Cost: [{hall_duration}]\n"
        f"\tAverage Frame time: [{average_duration}]\n"
        f"\tStd Error: [{std_error}]\n"
    )
    return average_duration
