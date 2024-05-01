import random
import unittest
from typing import List
from unittest.mock import Mock

from bdmc.modules.controller import CloseLoopController

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
        self.token_pool = [self.transition_start_a, self.transition_ab, self.transition_bc]
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

        # test with non-loop check
        self.assertEqual([], self.botix_instance.acquire_loops())

        # test a simple single loop check
        MovingState.__state_id_counter__ = 0
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

        self.botix_instance.token_pool = [
            transition_a_bcd,
            transition_d_e,
            transition_e_f,
            transition_f_d,
            transition_c_e,
        ]

        self.assertEqual(
            str(self.botix_instance.acquire_loops()),
            "[[4-MovingState(500, 500, 500, 500), "
            "5-MovingState(600, 600, 600, 600), "
            "3-MovingState(400, 400, 400, 400)]]",
        )

        # try to deal with the Diamond transition

        self.botix_instance.token_pool.remove(transition_f_d)
        transition_f_cd = MovingTransition(duration=1, from_states=state_f, to_states={1: state_c, 2: state_d})
        self.botix_instance.token_pool.append(transition_f_cd)

        self.assertEqual(
            "[[2-MovingState(300, 300, 300, 300), "
            "4-MovingState(500, 500, 500, 500), "
            "5-MovingState(600, 600, 600, 600)], "
            "[4-MovingState(500, 500, 500, 500), "
            "5-MovingState(600, 600, 600, 600), "
            "3-MovingState(400, 400, 400, 400)]]",
            str(self.botix_instance.acquire_loops()),
        )

    def test_max_branchless_chain_check(self):
        MovingState.__state_id_counter__ = 0
        """测试无分支链检查"""
        # 无分支链
        state_a = MovingState(100)
        state_b = MovingState(200)
        state_c = MovingState(300)
        state_d = MovingState(400)
        state_e = MovingState(500)
        state_f = MovingState(600)

        state_k = MovingState(700)
        state_l = MovingState(800)
        state_z = MovingState(900)

        transition_a_bcd = MovingTransition(
            duration=1, from_states=state_a, to_states={1: state_b, 2: state_c, 3: state_d}
        )

        transition_d_e = MovingTransition(duration=1, from_states=state_d, to_states=state_e)

        transition_e_f = MovingTransition(duration=1, from_states=state_e, to_states=state_f)

        transition_f_d = MovingTransition(duration=1, from_states=state_f, to_states=state_d)

        transition_c_e = MovingTransition(duration=1, from_states=state_c, to_states=state_e)

        transition_b_k = MovingTransition(duration=1, from_states=state_b, to_states=state_k)
        transition_k_lz = MovingTransition(duration=1, from_states=state_k, to_states={1: state_l, 2: state_z})

        self.botix_instance.token_pool = [
            transition_a_bcd,
            transition_d_e,
            transition_e_f,
            transition_f_d,
            transition_c_e,
            transition_b_k,
            transition_k_lz,
        ]

        self.assertEqual(
            str(self.botix_instance.acquire_max_branchless_chain(state_a)), "([0-MovingState(100, 100, 100, 100)], [])"
        )
        self.assertEqual(
            str(self.botix_instance.acquire_max_branchless_chain(state_b)),
            "([1-MovingState(200, 200, 200, 200), 6-MovingState(700, 700, 700, 700)],"
            " [[1-MovingState(200, 200, 200, 200)] => [6-MovingState(700, 700, 700, 700)]])",
        )

    def test_ident(self):
        test_string = "Line 1\nLine 2\nLine 3"
        expected_output = "    Line 1\n    Line 2\n    Line 3"
        assert self.botix_instance._add_indent(test_string, indent="    ", count=1) == expected_output

        test_list = ["Line 1", "Line 2", "Line 3"]
        expected_output = ["    Line 1", "    Line 2", "    Line 3"]
        assert self.botix_instance._add_indent(test_list, indent="    ") == expected_output

        test_list = ["Line 1", 2, "Line 3"]
        with self.assertRaises(TypeError):
            self.botix_instance._add_indent(test_list, indent="    ")

        invalid_input = 123  # 假设一个非字符串非列表的输入
        with self.assertRaises(TypeError):
            self.botix_instance._add_indent(invalid_input, indent="    ")

    def test_asm_and_indent_without_setup(self):
        # Instantiate your class if needed for non-static methods
        test_instance = self.botix_instance

        # Define test data
        cases = {"case1": "result1\n    more_result1", "case2": "result2"}
        match_expression = "some_variable"

        # Test _add_indent with list
        input_lines = ["line1", "line2"]
        expected_output_list = ["    line1", "    line2"]
        self.assertEqual(test_instance._add_indent(input_lines, count=1), expected_output_list)

        # Test _add_indent with string
        input_string = "line1\nline2"
        expected_output_string = "    line1\n    line2"
        self.assertEqual(test_instance._add_indent(input_string, count=1), expected_output_string)

        # Test _assembly_match_cases
        expected_match_cases = [
            "match some_variable:",
            "    case case1:",
            "        result1",
            "            more_result1",
            "    case case2:",
            "        result2",
        ]
        self.assertEqual(test_instance._assembly_match_cases(match_expression, cases), expected_match_cases)

        # Test _add_indent TypeError
        with self.assertRaises(TypeError):
            test_instance._add_indent(123)  # This should raise a TypeError

    def test_compile(self):
        self.botix_instance.compile()
        self.assertEqual(
            self.botix_instance.compile(True)[0],
            [
                "def _func():",
                "    con.set_motors_speed((-1, -1, -1, -1)).delay(0.1).set_motors_speed((0, 0, 0, 0)).delay(1).set_motors_speed((1, 1, 1, 1)).delay(2).set_motors_speed((2, 2, 2, 2))",
            ],
        )

    def test_compile_with_branches(self):
        MovingState.__state_id_counter__ = 0
        MovingTransition.__state_id_counter__ = 0
        state_a = MovingState(100)
        state_b = MovingState(200)
        state_c = MovingState(300)
        state_d = MovingState(400)
        state_e = MovingState(500)
        state_f = MovingState(600)

        def transition_breaker_fac(lst: List[int]):
            def _inner() -> int:
                return random.choice(lst)

            return _inner

        transition_a_bcd = MovingTransition(
            duration=1,
            from_states=state_a,
            to_states={1: state_b, 2: state_c, 3: state_d},
            breaker=transition_breaker_fac([1, 2, 3]),
        )
        transition_d_ef = MovingTransition(
            duration=1,
            from_states=state_d,
            to_states={1: state_e, 2: state_f},
            breaker=transition_breaker_fac([1, 2]),
        )

        self.botix_instance.token_pool = [transition_a_bcd, transition_d_ef]

        compiled = self.botix_instance.compile(True)
        self.assertEqual(
            [
                "def _func():",
                "    match con.set_motors_speed((100, 100, 100, " "100)).delay_b_match(1,transition0_breaker_1,0.01):",
                "        case 1:",
                "            con.set_motors_speed((200, 200, 200, 200))",
                "        case 2:",
                "            con.set_motors_speed((300, 300, 300, 300))",
                "        case 3:",
                "            match con.set_motors_speed((400, 400, 400, "
                "400)).delay_b_match(1,transition1_breaker_1,0.01):",
                "                case 1:",
                "                    con.set_motors_speed((500, 500, 500, 500))",
                "                case 2:",
                "                    con.set_motors_speed((600, 600, 600, 600))",
            ],
            compiled[0],
        )

        obj = self.botix_instance.compile()
        obj()


if __name__ == "__main__":
    unittest.main()