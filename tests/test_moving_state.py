import unittest
from unittest.mock import Mock

from bdmc.modules.controller import CloseLoopController, MotorInfo

from mentabotix import MovingState


class TestMovingState(unittest.TestCase):

    def setUp(self):
        MovingState.__state_id_counter__ = 0

    def test_init_speeds(self):
        # Test FullPattern
        state_full = MovingState(50)
        self.assertEqual(state_full.unwrap(), (50, 50, 50, 50))

        # Test LRPattern
        state_lr = MovingState(30, 60)
        self.assertEqual(state_lr.unwrap(), (30, 30, 60, 60))

        # Test IndividualPattern
        state_individual = MovingState(10, 20, 30, 40)
        self.assertEqual(state_individual.unwrap(), (10, 20, 30, 40))

    def test_init_speed_expressions(self):
        used_context_variables = ["var1", "var2"]

        # Test FullExpressionPattern
        state_full_expr = MovingState(speed_expressions="var1+var2", used_context_variables=used_context_variables)
        self.assertEqual(state_full_expr.speed_expressions, ("var1+var2", "var1+var2", "var1+var2", "var1+var2"))

        # Test LRExpressionPattern
        state_lr_expr = MovingState(speed_expressions=("var1", "var2"), used_context_variables=used_context_variables)
        self.assertEqual(state_lr_expr.speed_expressions, ("var1", "var1", "var2", "var2"))

        # Test IndividualExpressionPattern
        state_indiv_expr = MovingState(
            speed_expressions=("var1", "var2", "var1+var2", "2*var1-var2"),
            used_context_variables=used_context_variables,
        )
        self.assertEqual(state_indiv_expr.speed_expressions, ("var1", "var2", "var1+var2", "2*var1-var2"))

    def test_init_invalid_arguments(self):
        with self.assertRaises(ValueError):
            MovingState()

        with self.assertRaises(ValueError):
            MovingState(1, 2, 3, 5, speed_expressions=("expr1", "expr2"))

        with self.assertRaises(ValueError):
            MovingState(speed_expressions=("expr1", "expr2"), used_context_variables=["var1", "var2"])

    def test_class_methods(self):
        # Test halt
        state_halt = MovingState.halt()
        self.assertEqual(state_halt.unwrap(), (0, 0, 0, 0))

        # Test straight
        state_straight_fwd = MovingState.straight(80)
        self.assertEqual(state_straight_fwd.unwrap(), (80, 80, 80, 80))
        state_straight_bwd = MovingState.straight(-80)
        self.assertEqual(state_straight_bwd.unwrap(), (-80, -80, -80, -80))

        # Test differential
        state_diff_l = MovingState.differential("l", radius=20, outer_speed=70)
        self.assertEqual(state_diff_l.unwrap(), (11, 11, 70, 70))
        state_diff_r = MovingState.differential("r", radius=20, outer_speed=70)
        self.assertEqual(state_diff_r.unwrap(), (70, 70, 11, 11))

        # Test turn
        state_turn_l = MovingState.turn("l", 90)
        self.assertEqual(state_turn_l.unwrap(), (-90, -90, 90, 90))
        state_turn_r = MovingState.turn("r", 90)
        self.assertEqual(state_turn_r.unwrap(), (90, 90, -90, -90))

        # Test drift
        state_drift_fl = MovingState.drift("fl", 50)
        self.assertEqual(state_drift_fl.unwrap(), (0, 50, int(76.5), 50))
        state_drift_rl = MovingState.drift("rl", 50)
        self.assertEqual(state_drift_rl.unwrap(), (50, 0, 50, int(76.5)))
        state_drift_rr = MovingState.drift("rr", 50)
        self.assertEqual(state_drift_rr.unwrap(), (int(76.5), 50, 0, 50))
        state_drift_fr = MovingState.drift("fr", 50)
        self.assertEqual(state_drift_fr.unwrap(), (50, int(76.5), 50, 0))

    def test_apply(self):
        state = MovingState(20, 30, 40, 50)
        state_applied = state.apply(1.5)

        self.assertEqual(state_applied.unwrap(), (30, 45, 60, 75))

    def test_clone(self):
        original_state = MovingState(10, 20, 30, 40)
        cloned_state = original_state.clone()
        print()

        print(cloned_state)
        print(original_state)
        print()
        self.assertEqual(cloned_state.unwrap(), (10, 20, 30, 40))
        self.assertIsNot(cloned_state, original_state)  # Ensure a new instance is returned

        original_state = MovingState(speed_expressions="var1", used_context_variables=["var1"])

        cloned_state = original_state.clone()
        print(cloned_state)
        print(original_state)
        print()

        self.assertEqual(cloned_state.speed_expressions, ("var1", "var1", "var1", "var1"))
        self.assertIsNot(cloned_state, original_state)  # Ensure a new instance is returned()
        self.assertNotEqual(id(original_state.after_exiting), id(cloned_state.after_exiting))
        self.assertNotEqual(id(original_state.before_entering), id(cloned_state.before_entering))

    def test_tokenize(self):

        con_mock = Mock(
            spec=CloseLoopController(
                [MotorInfo(1), MotorInfo(2), MotorInfo(3), MotorInfo(4)], port="COM4", context={"var1": 10, "var2": 20}
            )
        )

        state_with_speeds = MovingState(10, 20, 30, 40)
        tokens, context = state_with_speeds.tokenize(con_mock)
        self.assertEqual(tokens, [".set_motors_speed((10, 20, 30, 40))"])
        self.assertEqual(context, {})  # No context needed for states with speeds

        state_with_expressions = MovingState(
            speed_expressions=("var1", "var2", "var1+var2", "2*var1-var2"),
            used_context_variables=["var1", "var2"],
        )
        tokens, context = state_with_expressions.tokenize(con_mock)
        self.assertCountEqual(
            [
                ".set_motors_speed(((state1_context_getter_temp_2:=state1_context_getter_2()), "
                "(state1_context_getter_temp_1:=state1_context_getter_1()), "
                "state1_context_getter_temp_2+state1_context_getter_temp_1, "
                "2*state1_context_getter_temp_2-state1_context_getter_temp_1))"
            ][0],
            tokens[0],
        )

        self.assertIn("state1_context_getter_1", context)
        self.assertIn("state1_context_getter_2", context)

        print(f"Tokens: {tokens}, Context: {context}")
        # Tokens and context assertions depend on the NameGenerator implementation, which is not provided
        # You can add appropriate assertions here once you have the actual generated tokens and context

    def test_hash_and_eq(self):
        state1 = MovingState(10, 20, 30, 40)
        print(f"State1: {state1}")
        state2 = MovingState(10, 20, 30, 40)
        print(f"State2: {state2}")
        state3 = MovingState(11, 20, 30, 40)
        print(f"State3: {state3}")
        self.assertEqual(hash(state1), hash(state2) - 1)
        self.assertNotEqual(hash(state1), hash(state3))

        self.assertEqual(state1, state2)
        self.assertNotEqual(state1, state3)

    def test_str(self):
        state = MovingState(10, 20, 30, 40)
        self.assertEqual(str(state), f"{state.state_id}-MovingState(10, 20, 30, 40)")
        state = MovingState(speed_expressions="var1", used_context_variables=["var1"])
        self.assertEqual(str(state), f"{state.state_id}-MovingState('var1', 'var1', 'var1', 'var1')")

    def test_random_turn(self):
        from mentabotix import MovingTransition, Botix

        con = CloseLoopController(
            [MotorInfo(1), MotorInfo(2), MotorInfo(3), MotorInfo(4)], port="COM10", context={"var1": 10, "var2": 20}
        ).start_msg_sending()
        state = MovingState.rand_turn(con, 500)
        print(state)
        end = MovingState(0)
        trans = MovingTransition(1, from_states=[state], to_states={0: end})

        b = Botix(controller=con, token_pool=[trans])

        fun = b.compile()
        fun()
        con.stop_msg_sending()


if __name__ == "__main__":
    unittest.main()
