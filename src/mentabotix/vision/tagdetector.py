"""
apriltag detecting app
"""

import warnings
from copy import deepcopy
from threading import Thread
from time import sleep
from typing import Tuple, List, Dict, Optional, Literal

import cv2
from cv2 import Mat, cvtColor, COLOR_RGB2GRAY
from pyapriltags import Detector, Detection

from ..tools.algrithm_tools import calc_p2p_dst, calc_p2p_error

DEFAULT_TAG_TABLE = {2: (None, 0.0), 1: (None, 0.0), 0: (None, 0.0)}

DEFAULT_TAG_ID = -1

NULL_TAG = -999

TABLE_INIT_VALUE = (None, 0.0)

CAMERA_RESOLUTION_MULTIPLIER = 0.4

BLUE_TEAM = "blue"
YELLOW_TEAM = "yellow"


def get_center_tag(frame_center: Tuple[int, int], tags: List):
    """
    get the tag in which is the nearest to the frame center
    Args:
        frame_center:
        tags:

    Returns:

    """
    # 获取离图像中心最近的 AprilTag
    closest_tag = None
    closest_dist = float("inf")
    for tag in tags:
        # 计算当前 AprilTag 中心点与图像中心的距离
        dist = calc_p2p_dst(tag.center, frame_center)
        if dist < closest_dist:
            closest_dist = dist
            closest_tag = tag
    return closest_tag


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
    __tag_detect = detector.detect

    def __init__(
        self,
        cam_id: int,
        team_color: Literal["blue", "yellow"],
        start_detect_tag: bool = True,
        single_tag_mode: bool = True,
        minimal_resolution: bool = True,
    ):
        """

        Args:

            team_color:
            start_detect_tag:
            single_tag_mode:if check only a single tag one time
        """

        self._camera: cv2.VideoCapture = cv2.VideoCapture(cam_id)
        if minimal_resolution:
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1)
        self._frame_center: Tuple[float, float] = (
            self._camera.get(cv2.CAP_PROP_FRAME_WIDTH) / 2,
            self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2,
        )
        self._tags_table: Dict[int, Tuple[Optional[Detection], int | float]] = {}

        self._tag_id: int = DEFAULT_TAG_ID
        self._tag_monitor_switch: bool = True
        self._enemy_tag_id: int = NULL_TAG
        self._ally_tag_id: int = NULL_TAG
        self._neutral_tag_id: int = NULL_TAG

        self.team_color = team_color
        self._init_tags_table()
        self._single_tag_mode: bool = single_tag_mode
        self._apriltag_detect: Optional[Thread] = None
        self._detect_should_continue: bool = True

        self.apriltag_detect_start() if start_detect_tag else None

    def _init_tags_table(self):
        """
        the tag table stores the tag obj and the distance to the camera center
        :return:
        """
        self._tags_table[self._enemy_tag_id] = TABLE_INIT_VALUE
        self._tags_table[self._ally_tag_id] = TABLE_INIT_VALUE
        self._tags_table[self._neutral_tag_id] = TABLE_INIT_VALUE

    @property
    def team_color(self) -> str:
        """

        Returns: team color

        """

        return self._team_color

    @team_color.setter
    def team_color(self, team_color: str = BLUE_TEAM):
        """
        set the ally/enemy tag according the team color
        yellow: ally: 2 | enemy: 1|neutral: 0
        blue: ally: 1 | enemy: 2 | neutral: 0

        :param team_color: blue or yellow
        :return:
        """
        self._team_color = team_color
        self._neutral_tag_id = 0
        if team_color == BLUE_TEAM:
            self._enemy_tag_id = 2
            self._ally_tag_id = 1
        elif team_color == YELLOW_TEAM:
            self._enemy_tag_id = 1
            self._ally_tag_id = 2

    @property
    def detect_should_continue(self) -> bool:
        """

        Returns: the alive-status of the detection process

        """
        return self._detect_should_continue

    @detect_should_continue.setter
    def detect_should_continue(self, should: bool):

        self._detect_should_continue = should

    def apriltag_detect_start(self):
        """
        start the tag-detection thread and set it to daemon
        :return:
        """
        warnings.warn("AprilTag detect Activating")
        self._detect_should_continue: bool = True
        apriltag_detect = Thread(target=self._apriltag_detect_loop, name="apriltag_detect_Process")
        apriltag_detect.daemon = True
        apriltag_detect.start()

        self._apriltag_detect = apriltag_detect

    def _apriltag_detect_loop(self):
        """
        这是一个线程函数，它从摄像头捕获视频帧，处理帧以检测 AprilTags，
        :return:
        """
        frame_updater = self._camera.read

        warnings.warn("Detection Activated")
        while self._detect_should_continue:
            if self._tag_monitor_switch:  # 台上开启 台下关闭 节约性能
                success, frame = frame_updater()  # extract frame from the cam
                if success:
                    self._update_tags(frame)  # extract tags in the detection
                    # extract the correct tag in the detection,
                    # for example, the tag in the center of the frame
                    self._update_tag_id()
                else:
                    break
            else:
                sleep(0.4)
        warnings.warn("\n##########CAMERA CLOSED###########\n" "###ENTERING NO CAMERA MODE###")
        self._tag_id = DEFAULT_TAG_ID

    def _update_tags(self, frame: Mat):
        """
        update tags from the newly sampled frame
        :return:
        """
        # 将帧转换为灰度并存储在 gray 变量中。
        # 使用 AprilTag 检测器对象（self.tag_detector）在灰度帧中检测 AprilTags。检测到的标记存储在 self._tags 变量中。
        # override old tags
        temp_dict = deepcopy(DEFAULT_TAG_TABLE)
        for tag in self.__tag_detect(cvtColor(frame, COLOR_RGB2GRAY)):
            temp_dict[tag.tag_id] = (tag, calc_p2p_error(tag.center, self._frame_center))
        self._tags_table = temp_dict

    def _update_tag_id(self):
        """
        update the tag id from the self._tags_table
        :return:
        """

        def _single_mode():

            self._tag_id = DEFAULT_TAG_ID
            for tag_data in self._tags_table.values():
                if tag_data[0]:
                    self._tag_id = tag_data[0].tag_id
                    break

        def _nearest_mode():
            closest_dist = float("inf")
            closest_tag = None
            for tag_data in self._tags_table.values():
                # check the tag obj is valid and compare with the closest tag
                if tag_data[0] and tag_data[1] < closest_dist:
                    closest_dist = tag_data[1]
                    closest_tag = tag_data[0]
            self._tag_id = closest_tag.tag_id if closest_tag else DEFAULT_TAG_ID

        _single_mode() if self._single_tag_mode else _nearest_mode()

    @property
    def tag_table(self):
        """

        Returns: the tag table that contains all the results

        """
        return self._tags_table

    @property
    def tag_id(self):
        """
        :return:  current tag id
        """
        return self._tag_id

    @property
    def tag_detection_switch(self):
        """

        Returns: the detector status

        """
        return self._tag_monitor_switch

    @tag_detection_switch.setter
    def tag_detection_switch(self, switch: bool):
        """
        setter for the switch
        :param switch:
        :return:
        """
        if switch != self._tag_monitor_switch:
            self._tag_monitor_switch = switch
            self._tag_id = DEFAULT_TAG_ID

    @property
    def ally_tag_id(self) -> int:
        """

        Returns: the tag id of ally

        """
        return self._ally_tag_id

    @property
    def enemy_tag_id(self) -> int:
        """

        Returns: the tag id of the enemy

        """
        return self._enemy_tag_id

    @property
    def neutral_tag_id(self) -> int:
        """

        Returns: the tag id of the neutral

        """
        return self._neutral_tag_id
