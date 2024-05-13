import random
import unittest

from bdmc import CloseLoopController, MotorInfo

from mentabotix import MovingChainComposer, MovingState, MovingTransition, straight_chain, Botix


# Assuming MovingState, MovingTransition, UnitType, StateTransitionPack are defined elsewhere
# You'll need to replace these with the actual classes and types


class TestComposer(unittest.TestCase):
    def setUp(self):
        self.moving_chain_composer = MovingChainComposer()

    def test_last_state(self):
        # Test when there are no states
        self.assertIsNone(self.moving_chain_composer.last_state)

        # Test with one state
        state1 = MovingState(10)
        self.moving_chain_composer.add(state1)
        self.assertEqual(self.moving_chain_composer.last_state, state1)

        # Test with multiple states
        state2 = MovingState(100)
        with self.assertRaises(ValueError):
            self.moving_chain_composer.add(state2)

        tran1 = MovingTransition(1)
        tran2 = MovingTransition(2)
        self.moving_chain_composer.add(tran1)
        with self.assertRaises(ValueError):
            self.moving_chain_composer.add(tran2)
        # Test with one state
        state3 = MovingState(1052)
        self.moving_chain_composer.add(state3)
        self.assertEqual(self.moving_chain_composer.last_state, state3)

    def test_last_transition(self):
        # Test when there are no transitions
        self.assertIsNone(self.moving_chain_composer.last_transition)
        self.moving_chain_composer.add(MovingState(0))
        # Test with one transition
        tran1 = MovingTransition(10)
        self.moving_chain_composer.add(tran1)
        self.assertEqual(self.moving_chain_composer.last_transition, tran1)

        # Test with multiple transitions
        tran2 = MovingTransition(100)
        with self.assertRaises(ValueError):
            self.moving_chain_composer.add(tran2)

        state1 = MovingState(1)
        state2 = MovingState(2)
        self.moving_chain_composer.add(state1)
        with self.assertRaises(ValueError):
            self.moving_chain_composer.add(state2)
        # Test with one transition
        tran3 = MovingTransition(1052)
        self.moving_chain_composer.add(tran3)
        self.assertEqual(self.moving_chain_composer.last_transition, tran3)

    def test_next_need(self):
        # Test initial next need
        self.assertEqual(self.moving_chain_composer.next_need, MovingState)

        # Test after adding a MovingState
        self.moving_chain_composer.add(MovingState(0))
        self.assertEqual(self.moving_chain_composer.next_need, MovingTransition)

    def test_init_container(self):
        self.moving_chain_composer.add(MovingState(0))
        self.moving_chain_composer.init_container().add(MovingState(0))

    def test_export_structure(self):
        # Test with no units
        self.assertEqual(self.moving_chain_composer.export_structure(), ([], []))

        # Test with states and transitions
        state1, state2 = MovingState(1), MovingState(2)
        transition1, transition2 = MovingTransition(1), MovingTransition(2)
        (self.moving_chain_composer.add(state1).add(transition1).add(state2).add(transition2))

        expected_structure = ([state1, state2], [transition1, transition2])
        self.assertEqual(self.moving_chain_composer.export_structure(), expected_structure)

    def test_add(self):
        # Test adding correct unit types
        state = MovingState(1)
        transition = MovingTransition(1)
        self.moving_chain_composer.add(state)
        self.moving_chain_composer.add(transition)

        # Test adding incorrect unit type
        with self.assertRaises(ValueError):
            self.moving_chain_composer.add(14)

    # 测试没有breaker的情况
    def test_straight_chain_without_breaker(self):
        start_speed = 50
        end_speed = 100
        duration = 5.0
        power_exponent = 1.0
        interval = 0.07

        # 调用待测试函数
        states, transitions = straight_chain(
            start_speed=start_speed,
            end_speed=end_speed,
            duration=duration,
            power_exponent=power_exponent,
            interval=interval,
        )

        # 断言判断结果是否符合预期
        self.assertEqual(len(states), int(duration / interval) + 1)
        self.assertEqual(states[0].unwrap()[0], start_speed)
        self.assertEqual(states[-1].unwrap()[0], end_speed)
        for i in range(len(transitions) - 1):
            self.assertIsInstance(transitions[i], MovingTransition)
            self.assertEqual(transitions[i].duration, interval)

    # 测试有breaker函数的情况
    def test_straight_chain_with_breaker(self):
        start_speed = 50
        end_speed = 100
        duration = 5.0
        power_exponent = 1.0
        interval = 0.1

        def break_function() -> bool:
            return random.random() < 0.1

        # 调用待测试函数
        states, transitions = straight_chain(
            start_speed=start_speed,
            end_speed=end_speed,
            duration=duration,
            power_exponent=power_exponent,
            interval=interval,
            breaker=break_function,
        )

        # 断言判断结果是否符合预期
        self.assertTrue(any(isinstance(t, MovingTransition) and t.breaker for t in transitions))
        self.assertEqual(int(duration / interval), len(transitions))
        self.assertAlmostEqual(duration, sum(t.duration for t in transitions))
        self.assertEqual(int(duration / interval) + 1 + 1, len(states))  # 包含了break状态
        self.assertIn(MovingState(0), states)  # 确认state_on_break被加入到states列表中

    def test_exp(self):
        start_speed = 50
        end_speed = 100
        duration = 5.0
        power_exponent = 2.0
        interval = 0.1

        def break_function() -> bool:
            return random.random() < 0.1

        # 调用待测试函数
        states, transitions = straight_chain(
            start_speed=start_speed,
            end_speed=end_speed,
            duration=duration,
            power_exponent=power_exponent,
            interval=interval,
            breaker=break_function,
        )

    def test_structure(self):
        start_speed = 50
        end_speed = 100
        duration = 5.0
        power_exponent = 2.0
        interval = 0.1

        def break_function() -> bool:
            return random.random() < 0.1

        # 调用待测试函数
        states, transitions = straight_chain(
            start_speed=start_speed,
            end_speed=end_speed,
            duration=duration,
            power_exponent=power_exponent,
            interval=interval,
            breaker=break_function,
        )

        botix = Botix(controller=CloseLoopController([MotorInfo(i) for i in range(4)]), token_pool=transitions)
        botix.export_structure("long_chain.puml")

    # 测试breaker不是None且不是callable的情况
    def test_straight_chain_with_invalid_breaker(self):
        start_speed = 50
        end_speed = 100
        duration = 5.0
        power_exponent = 1.0
        interval = 0.07

        # 使用错误的breaker参数
        with self.assertRaises(ValueError):
            straight_chain(
                start_speed=start_speed,
                end_speed=end_speed,
                duration=duration,
                power_exponent=power_exponent,
                interval=interval,
                breaker="not a callable",
            )

    def test_mk(self):
        from mentabotix import MovingChainComposer, MovingState, MovingTransition

        # init the state-transition composer
        comp = MovingChainComposer()

        # add some states and transitions one by one to the composer, the composer will auto-connect the states and transitions
        (
            comp.add(MovingState(0))
            .add(MovingTransition(0.2))
            .add(MovingState(1000))
            .add(MovingTransition(0.3))
            .add(MovingState(2000))
        )

        # export the structure
        states, transitions = comp.export_structure()

        # let's use botix to make the visualization!
        # first make the botix object
        con = CloseLoopController(motor_infos=[MotorInfo(i) for i in range(4)])
        botix = Botix(controller=con, token_pool=transitions)

        # make the visualization
        botix.export_structure("composed.puml")


if __name__ == "__main__":
    unittest.main()
