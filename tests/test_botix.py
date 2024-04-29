import unittest
from unittest.mock import Mock

from bdmc.modules.controller import CloseLoopController, MotorInfo

from mentabotix import Botix, MovingState, MovingTransition
from mentabotix.modules.exceptions import StructuralError


class TestBotix(unittest.TestCase):

    def setUp(self):
        # 创建一些假的MovingState对象用于测试
        self.start_state = MovingState(-1)
        self.state_a = MovingState(0)
        self.state_b = MovingState(1)
        self.state_c = MovingState(2)
        # 创建一些假的MovingTransition对象用于测试
        self.transition_start_a = MovingTransition(duration=0.1, from_states=self.start_state, to_states=self.state_a)
        self.transition_ab = MovingTransition(duration=1, from_states=[self.state_a], to_states=self.state_b)
        self.transition_bc = MovingTransition(duration=2, from_states=[self.state_b], to_states=self.state_c)

        # 初始化一个Botix实例用于测试
        self.controller_mock = Mock(spec=CloseLoopController)
        self.token_pool = [self.transition_ab, self.transition_bc, self.transition_start_a]
        self.botix_instance = Botix(controller=self.controller_mock, token_pool=self.token_pool)

    def test_get_start(self):
        self.assertEqual(self.start_state, self.botix_instance.acquire_unique_start(self.botix_instance.token_pool))

    def test_initialization(self):
        """测试Botix类的初始化"""
        self.assertIs(self.botix_instance.controller, self.controller_mock)
        self.assertEqual(self.botix_instance.token_pool, self.token_pool)

    def test_ensure_structure_validity(self):
        """测试结构有效性检查"""
        # 正确的情况，每个状态只连接到一个转换
        self.assertIsNone(Botix.ensure_structure_validity(self.token_pool))

        # 错误的情况，创建一个状态连接到两个转换
        state_d = MovingState(41)
        transition_ad = MovingTransition(duration=4, from_states=[self.state_a], to_states=state_d)
        with self.assertRaises(StructuralError):
            Botix.ensure_structure_validity(self.token_pool + [transition_ad])

    def test_ensure_accessibility(self):
        """测试状态可达性检查"""
        # 状态链是连通的
        Botix.ensure_accessibility(self.token_pool, self.state_a, {self.state_c})

        # 添加一个无法到达的状态
        state_d = MovingState(10)
        with self.assertRaises(ValueError):
            Botix.ensure_accessibility(self.token_pool, self.state_c, {state_d})

    def test_acquire_connected_forward_transition(self):
        """测试获取连接的前进转换"""
        # 确保从state_a可以找到通往state_b的转换
        transition = self.botix_instance.acquire_connected_forward_transition(self.state_a)
        self.assertIsInstance(transition, MovingTransition)
        self.assertEqual(list(transition.to_states.values())[0], self.state_b)

        # 尝试从一个没有连接转换的状态获取转换，且期望抛出错误
        with self.assertRaises(ValueError):
            self.botix_instance.acquire_connected_forward_transition(MovingState())

    def test_branchless_chain(self):
        """测试是否存在无分支链"""
        # 存在从A到B到C的无分支链
        self.assertTrue(self.botix_instance.is_branchless_chain(self.state_a, self.state_c))

        # 添加一个额外的转换打破链，应返回False
        state_d = MovingState(0)
        self.transition_bc.to_states[2] = state_d
        self.assertFalse(self.botix_instance.is_branchless_chain(self.state_a, self.state_c))

    def test_acquire_loops(self):
        state_a = MovingState(100)
        state_b = MovingState(200)
        state_c = MovingState(300)
        state_d = MovingState(400)
        state_e = MovingState(500)
        state_f = MovingState(600)

        transition_a_bcd = MovingTransition(
            duration=1, from_states=state_a, to_states={1: state_b, 2: state_c, 3: state_d}
        )

        transition_d_e = MovingTransition(duration=1, from_states=state_d, to_states=state_e)

        transition_e_f = MovingTransition(duration=1, from_states=state_e, to_states=state_f)

        transition_f_d = MovingTransition(duration=1, from_states=state_f, to_states=state_d)

        transition_c_e = MovingTransition(duration=1, from_states=state_c, to_states=state_e)
        controller = CloseLoopController(
            motor_infos=[MotorInfo(0), MotorInfo(1), MotorInfo(2), MotorInfo(3)], port="COM3"
        )
        botix = Botix(
            token_pool=[transition_a_bcd, transition_d_e, transition_e_f, transition_f_d, transition_c_e],
            controller=controller,
        )

        print(botix.acquire_loops())


if __name__ == "__main__":
    unittest.main()
