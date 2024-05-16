import os
import sys
import unittest


def discover_tests_in_package(package_path):
    # 获取package路径
    package_dir = os.path.dirname(os.path.abspath(__file__))

    # 使用TestLoader的discover方法来查找所有测试
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=package_dir, pattern="test_*.py", top_level_dir=os.getcwd())

    # 执行测试
    runner = unittest.TextTestRunner(verbosity=2)
    results = runner.run(suite)

    # 检查是否有失败或错误
    if not results.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    # 假设当前位置就是软件包的顶层目录
    discover_tests_in_package(".")
