import unittest
from typing import List, Callable

from mentabotix import set_log_level
from mentabotix.modules.menta import (
    Menta,
    SamplerType,
    SamplerUsage,
)


# 示例采样器实现
def mock_sequence_sampler() -> List:
    return [1.2, 32, 1.3]


def mock_indexed_sampler(index: int) -> int:
    return 10 * index


def mock_direct_sampler() -> int:
    return 42


class TestMenta(unittest.TestCase):

    def setUp(self):
        set_log_level(10)
        self.menta = Menta(
            [
                mock_sequence_sampler,
                mock_indexed_sampler,
                mock_direct_sampler,
            ]
        )

    def test_init(self):
        self.assertIsInstance(self.menta.samplers, List)
        self.assertEqual(len(self.menta.samplers), 3)
        self.assertIsNotNone(self.menta.sampler_types)

    def test_update_sampler_types(self):
        self.menta.update_sampler_types()
        self.assertEqual(
            self.menta.sampler_types,
            [
                SamplerType.SEQ_SAMPLER,
                SamplerType.IDX_SAMPLER,
                SamplerType.DRC_SAMPLER,
            ],
        )

    def test_construct_updater_single_seq_sampler(self):
        usage = SamplerUsage(used_sampler_index=0, required_data_indexes=[])
        updater = self.menta.construct_updater([usage])
        result = updater()
        self.assertIsInstance(result, List)
        self.assertEqual(len(result), 3)
        for data in result:
            self.assertIsInstance(data, int | float)

    def test_construct_updater_single_idx_sampler(self):
        usage = SamplerUsage(used_sampler_index=1, required_data_indexes=[0])
        updater = self.menta.construct_updater([usage])
        result = updater()
        self.assertIsInstance(result, int)
        self.assertEqual(result, 0)

    def test_construct_updater_single_drc_sampler(self):
        usage = SamplerUsage(used_sampler_index=2, required_data_indexes=[])
        updater = self.menta.construct_updater([usage])
        result = updater()
        self.assertIsInstance(result, int)
        self.assertEqual(result, 42)

    def test_construct_updater_multiple_samplers(self):
        usages = [
            SamplerUsage(used_sampler_index=0, required_data_indexes=[0, 2]),
            SamplerUsage(used_sampler_index=1, required_data_indexes=[5]),
            SamplerUsage(used_sampler_index=2, required_data_indexes=[]),
        ]
        updater = self.menta.construct_updater(usages)
        results = updater()
        print(results)
        self.assertIsInstance(results, tuple)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], (1.2, 1.3))
        self.assertEqual(results[1], 50)
        self.assertEqual(results[2], 42)

    def test_resolve_drc_sampler(self):
        usage = SamplerUsage(used_sampler_index=2, required_data_indexes=[])
        updater = self.menta.resolve_drc_sampler(
            sampler=self.menta.samplers[usage.used_sampler_index], required_data_indexes=usage.required_data_indexes
        )
        result = updater()
        self.assertIsInstance(result, int)
        self.assertEqual(result, 42)

        """
        f'{42:08b}' => '00101010'
        # make this 8-bits to list should get [0,1,0,1,0,1,0,0], it is reversed
        """
        usage = SamplerUsage(used_sampler_index=2, required_data_indexes=[0, 1, 2])
        updater = self.menta.resolve_drc_sampler(
            sampler=self.menta.samplers[usage.used_sampler_index], required_data_indexes=usage.required_data_indexes
        )
        result = updater()
        print(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(0, result[0])
        self.assertEqual(1, result[1])
        self.assertEqual(0, result[2])

    def test_construct_judge(self):
        # Test with multiple usages
        sages = [
            SamplerUsage(used_sampler_index=0, required_data_indexes=[0, 2]),
            SamplerUsage(used_sampler_index=1, required_data_indexes=[5]),
            SamplerUsage(used_sampler_index=2, required_data_indexes=[0, 1, 2]),
        ]
        func = self.menta.construct_inlined_function(
            usages=sages, judging_source="ret=s0 or s1 or s2 or s3 or s4 or s5"
        )
        self.assertIsInstance(func, Callable)
        self.assertEqual(func(), 1.2)
        func = self.menta.construct_inlined_function(usages=sages, judging_source="ret=s0+s1+s2+s3+s4+s5")
        self.assertIsInstance(func, Callable)
        self.assertEqual(func(), 53.5)

        # Test with 1 usage
        func = self.menta.construct_inlined_function(
            usages=[
                SamplerUsage(used_sampler_index=0, required_data_indexes=[0, 2]),
            ],
            judging_source="ret=s0 + s1",
        )
        self.assertIsInstance(func, Callable)
        self.assertEqual(func(), 2.5)

        # Test with  multiline judging source
        func = self.menta.construct_inlined_function(
            usages=sages,
            judging_source=["a=s0+s1", "print(f'this is {a}')", "b=s3+s4+s5", "print(f'this is {b}')", "ret=s2"],
        )
        self.assertIsInstance(func, Callable)
        self.assertEqual(func(), 50)

    def test_indexer_seq(self):
        varname = "hel"
        req = [0, 1, 2, 3]
        data_1 = self.menta._index_for_seq_sampler_data(varname, req)
        print(data_1)
        self.assertEqual(["hel[0]", "hel[1]", "hel[2]", "hel[3]"], data_1)

        data_2 = self.menta._index_for_seq_sampler_data(varname, len(req))
        self.assertEqual(["hel[0]", "hel[1]", "hel[2]", "hel[3]"], data_2)

    def test_indexer_drc(self):
        varname = "hel"
        req = [0, 1, 2]

        data_3 = self.menta._index_for_drc_sampler_data(varname, req)
        self.assertEqual(["((hel>>0)&1)", "((hel>>1)&1)", "((hel>>2)&1)"], data_3)

        data_4 = self.menta._index_for_drc_sampler_data(varname, len(req))
        self.assertEqual(["((hel>>0)&1)", "((hel>>1)&1)", "((hel>>2)&1)"], data_4)

    def test_indexer_idx(self):
        varname = "hel"
        req = [0, 1, 2]
        data_1 = self.menta._index_for_idx_sampler_data(varname, req)
        print(data_1)
        self.assertEqual(["hel(0)", "hel(1)", "hel(2)"], data_1)

    def test_constructed_judge_function_performance(self):
        sages = [
            SamplerUsage(used_sampler_index=0, required_data_indexes=[0, 2]),
            SamplerUsage(used_sampler_index=1, required_data_indexes=[5]),
            SamplerUsage(used_sampler_index=2, required_data_indexes=[0, 1, 2]),
        ]
        func = self.menta.construct_inlined_function(usages=sages, judging_source="ret=s0+s1+s2+s3+s4+s5")

        def _manual_seq_func():
            seq_temp = mock_sequence_sampler()

            return seq_temp[0], seq_temp[2]

        def _manual_drc_func():
            drc_temp = mock_direct_sampler()
            return tuple((drc_temp >> ri) & 1 for ri in [0, 1, 2])

        def _manual_asm_func():
            return *_manual_seq_func(), mock_indexed_sampler(5), *_manual_drc_func()

        def _manual_judge_func():
            return sum(_manual_asm_func())

        self.assertEqual(func(), _manual_judge_func())

        print(_manual_drc_func())
        import timeit

        # 假设这两个函数已经在你的代码中定义好了

        # 设置测试次数（例如：1000次）
        number_of_runs = 1000

        # 对func()进行计时测试
        func_time = timeit.timeit(lambda: func(), number=number_of_runs)

        print(f"func() execution time for {number_of_runs} runs: {func_time:.6f} seconds")

        # 对_manual_judge_func()进行计时测试
        manual_judge_func_time = timeit.timeit(lambda: _manual_judge_func(), number=number_of_runs)

        print(f"_manual_judge_func() execution time for {number_of_runs} runs: {manual_judge_func_time:.6f} seconds")

        print(f"func() is {manual_judge_func_time / func_time:.2f} times faster than _manual_judge_func()")

    def test_construct_ret_raw(self):
        sages = [
            SamplerUsage(used_sampler_index=0, required_data_indexes=[0, 2]),
            SamplerUsage(used_sampler_index=1, required_data_indexes=[5]),
            SamplerUsage(used_sampler_index=2, required_data_indexes=[0, 1, 2]),
        ]
        func, _ = self.menta.construct_inlined_function(
            usages=sages, judging_source="ret=s0,s1,s2,s3,s4+s5", return_raw=True
        )

        self.assertEqual(
            func,
            """def _func():
 func_0_temp,func_2_temp=func_0(),func_2()
 ret=func_0_temp[0],func_0_temp[2],func_1(5),((func_2_temp>>0)&1),((func_2_temp>>1)&1)+((func_2_temp>>2)&1)
 return ret""",
        )

        print(func)
        func = self.menta.construct_inlined_function(
            usages=sages, judging_source="ret=s0,s1,s2,s3,s4+s5", return_raw=False
        )
        print(func())

    def tearDown(self):
        # 清理可能的副作用
        pass


if __name__ == "__main__":
    unittest.main()
