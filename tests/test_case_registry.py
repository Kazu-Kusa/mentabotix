import unittest
from enum import Enum

from mentabotix import CaseRegistry, MovingState


# Mocking MovingState and Enum class for testing purposes


class TestEnum(Enum):
    CASE1 = "case1"
    CASE2 = "case2"
    CASE3 = "case3"


class TestCaseRegistry(unittest.TestCase):
    def setUp(self):
        self.case_registry = CaseRegistry(TestEnum)

    def test_init(self):
        self.assertIsInstance(self.case_registry._case_dict, dict)
        self.assertEqual(self.case_registry._to_cover, TestEnum)

    def test_reassign(self):
        self.case_registry.reassign(TestEnum)
        self.assertEqual(self.case_registry._to_cover, TestEnum)

    def test_init_container(self):
        self.case_registry.init_container()
        self.assertEqual(len(self.case_registry._case_dict), 0)

    def test_export(self):
        self.case_registry.register(TestEnum.CASE1, MovingState(1))
        print(self.case_registry.case_dict)
        with self.assertRaises(ValueError):
            exported_dict = self.case_registry.export()
        self.case_registry.register(TestEnum.CASE2, MovingState(1)).register(TestEnum.CASE3, MovingState(1))
        exported_dict = self.case_registry.export()
        self.assertIn(TestEnum.CASE1.value, exported_dict)
        self.assertEqual(exported_dict[TestEnum.CASE1.value].unwrap()[0], 1)

    def test_table(self):
        self.case_registry.register(TestEnum.CASE1, MovingState(1))
        self.case_registry.register(TestEnum.CASE2, MovingState(1)).register(TestEnum.CASE3, MovingState(1))
        table = self.case_registry.table
        self.assertIn(TestEnum.CASE1.value, table)
        self.assertIn(TestEnum.CASE2.value, table)
        self.assertIn(TestEnum.CASE3.value, table)
        print()
        print(table)

    def test_register(self):
        self.case_registry.register(TestEnum.CASE1, MovingState(1))
        self.assertIn(TestEnum.CASE1.value, self.case_registry._case_dict)
        with self.assertRaises(ValueError):
            self.case_registry.register(TestEnum.CASE1, MovingState(1))

    def test_batch_register(self):
        self.case_registry.batch_register([TestEnum.CASE1, TestEnum.CASE2], MovingState(1))
        self.assertIn(TestEnum.CASE1.value, self.case_registry._case_dict)
        self.assertIn(TestEnum.CASE2.value, self.case_registry._case_dict)
        with self.assertRaises(ValueError):
            self.case_registry.batch_register([TestEnum.CASE1, TestEnum.CASE3], MovingState(1))


if __name__ == "__main__":
    unittest.main()
