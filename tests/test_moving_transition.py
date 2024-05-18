import unittest
from itertools import zip_longest
from unittest.mock import patch

from terminaltables import SingleTable

from mentabotix.modules.botix import MovingTransition, MovingState


# Define a mock MovingState class for testing purposes


class TestMovingTransition(unittest.TestCase):

    def setUp(self):
        self.default_duration = 1.5

        def breker() -> bool:
            return True

        self.default_breaker = breker
        self.default_check_interval = 0.2
        self.default_from_state = MovingState(0)
        self.default_to_state = MovingState(0)

    def test_init_valid_input(self):
        # Test valid input with all parameters provided
        transition = MovingTransition(
            self.default_duration,
            self.default_breaker,
            self.default_check_interval,
            self.default_from_state,
            self.default_to_state,
        )
        self.assertEqual(transition.duration, self.default_duration)
        self.assertEqual(transition.breaker, self.default_breaker)
        self.assertEqual(transition.check_interval, self.default_check_interval)
        self.assertIsInstance(transition.from_states, list)
        self.assertIsInstance(transition.to_states, dict)

    def test_init_duration_not_positive(self):
        with self.assertRaises(ValueError):
            MovingTransition(-1, None, None, None, None)

    def test_init_from_states_valid_types(self):
        # Test valid from_states types
        transition = MovingTransition(self.default_duration, None, None, self.default_from_state, None)
        self.assertEqual(len(transition.from_states), 1)
        self.assertEqual(transition.from_states[0], self.default_from_state)

        from_states_iterable = [MovingState(0), MovingState(0)]
        transition = MovingTransition(self.default_duration, None, None, from_states_iterable, None)
        self.assertEqual(len(transition.from_states), len(from_states_iterable))
        self.assertEqual(transition.from_states, list(from_states_iterable))

    def test_init_from_states_invalid_type(self):
        with self.assertRaises(ValueError):
            MovingTransition(self.default_duration, None, None, "invalid", None)

    def test_init_to_states_valid_types(self):
        # Test valid to_states types
        transition = MovingTransition(self.default_duration, None, None, None, self.default_to_state)
        self.assertEqual(len(transition.to_states), 1)
        self.assertEqual(list(transition.to_states.values())[0], self.default_to_state)

        to_states_dict = {1: MovingState(0), 2: MovingState(1)}
        transition = MovingTransition(self.default_duration, None, None, None, to_states_dict)
        self.assertEqual(len(transition.to_states), len(to_states_dict))
        self.assertEqual(transition.to_states, to_states_dict)

    def test_init_to_states_invalid_type(self):
        with self.assertRaises(ValueError):
            MovingTransition(self.default_duration, None, None, None, "invalid")

    def test_add_from_state(self):
        transition = MovingTransition(self.default_duration, None, None, None, None)
        transition.add_from_state(self.default_from_state)
        self.assertEqual(len(transition.from_states), 1)
        self.assertEqual(transition.from_states[0], self.default_from_state)

    def test_add_to_state(self):
        transition = MovingTransition(self.default_duration, None, None, None, None)
        key = 1
        transition.add_to_state(key, self.default_to_state)
        self.assertEqual(len(transition.to_states), 1)
        self.assertEqual(transition.to_states[key], self.default_to_state)

    @patch("mentabotix.MovingTransition.tokenize")
    def test_tokenize_called(self, mock_tokenize):
        transition = MovingTransition(self.default_duration, None, None, None, None)
        transition.tokenize()
        mock_tokenize.assert_called_once_with()

    @patch("mentabotix.MovingTransition.clone")
    def test_clone_called(self, mock_clone):
        transition = MovingTransition(self.default_duration, None, None, None, None)
        transition.clone()
        mock_clone.assert_called_once_with()

    def test_clone_behavior(self):
        transition1 = MovingTransition(self.default_duration)
        transition2 = transition1.clone()
        self.assertEqual(transition1, transition2)
        self.assertNotEqual(id(transition1), id(transition2))
        self.assertNotEqual(id(transition1.to_states), id(transition2.to_states))
        self.assertNotEqual(id(transition1.from_states), id(transition2.from_states))

    def test_identifier(self):
        transition1 = MovingTransition(self.default_duration, None, None, None, None)
        transition2 = MovingTransition(self.default_duration, None, None, None, None)
        self.assertNotEqual(transition1.identifier, transition2.identifier)

    def test_str(self):
        from_states = [MovingState(100), MovingState(0)]
        to_states = {1: MovingState(4100), 2: MovingState(2000), 3: MovingState(0)}
        transition = MovingTransition(self.default_duration, None, None, from_states, to_states)
        temp = [["From", "To"]]
        for from_state, to_state in zip_longest(transition.from_states, transition.to_states.values()):
            temp.append([str(from_state) if from_state else "", str(to_state) if to_state else ""])

        self.assertEqual(str(transition), SingleTable(temp).table)

    def test_hash(self):
        transition1 = MovingTransition(self.default_duration, None, None, None, None)
        transition2 = MovingTransition(self.default_duration, None, None, None, None)
        self.assertNotEqual(hash(transition1), hash(transition2))


if __name__ == "__main__":
    unittest.main()
