from enum import Enum
from inspect import signature, Signature
from typing import (
    Callable,
    TypeAlias,
    List,
    Optional,
    Tuple,
    Sequence,
    NamedTuple,
    Union,
    Self,
    SupportsInt,
    SupportsFloat,
    Dict,
    Any,
)

from .exceptions import BadSignatureError, RequirementError
from .logger import _logger

SensorData: TypeAlias = float | int
# basically, no restrictions, support py objects or ctypes._CData variants
SensorDataSequence: TypeAlias = Sequence[SensorData]
UpdaterClosure = Callable[[], SensorDataSequence] | Callable[[], SensorData]

SequenceSampler: TypeAlias = Callable[[], SensorDataSequence]
IndexedSampler: TypeAlias = Callable[[int], SensorData]
DirectSampler: TypeAlias = Callable[[], SensorData]

Sampler: TypeAlias = Union[SequenceSampler, IndexedSampler, DirectSampler]
SourceCode: TypeAlias = str


class SamplerType(Enum):
    """
    采样器类型
    """

    SEQ_SAMPLER: int = 1
    IDX_SAMPLER: int = 2
    DRC_SAMPLER: int = 3


class SamplerUsage(NamedTuple):
    used_sampler_index: int
    required_data_indexes: Sequence[int]


class Menta:
    # reserved to check the return type of the sampler
    __supported__ = (SupportsInt, SupportsFloat)

    def __init__(
        self,
        samplers: Optional[List[Sampler]] = None,
    ):
        self.samplers: List[Sampler] = samplers or []
        self.sampler_types: List[SamplerType] = []
        self.update_sampler_types()

    def update_sampler_types(self) -> Self:
        """
        更新采样器类型列表。遍历self.samplers中的每个采样器，根据其返回类型和参数数量将其分类为序列采样器（SEQ_SAMPLER）、
        索引采样器（IDX_SAMPLER）或直接响应采样器（DRC_SAMPLER），并更新self.sampler_types列表。
        如果采样器没有指定返回类型或指定的返回类型不被支持，则抛出异常。

        返回值:
            Self: 更新后的实例自身。
        """

        self.sampler_types.clear()  # 清空当前的采样器类型列表
        for sampler in self.samplers:  # 遍历所有采样器
            sig = signature(sampler)  # 获取采样器的签名
            try:
                match sig.return_annotation:  # 根据采样器的返回类型进行匹配
                    case seq_sampler_ret if hasattr(seq_sampler_ret, "__getitem__") and len(sig.parameters) == 0:
                        # 如果返回类型是序列且无参数，则认定为序列采样器
                        self.sampler_types.append(SamplerType.SEQ_SAMPLER)
                    case idx_sampler_ret if len(sig.parameters) == 1 and isinstance(
                        idx_sampler_ret, self.__supported__
                    ):
                        # 如果返回类型是索引且只有一个参数，则认定为索引采样器
                        self.sampler_types.append(SamplerType.IDX_SAMPLER)
                    case drc_sampler_ret if len(sig.parameters) == 0 and isinstance(
                        drc_sampler_ret, self.__supported__
                    ):
                        # 如果返回类型是直接响应且无参数，则认定为直接响应采样器
                        self.sampler_types.append(SamplerType.DRC_SAMPLER)
                    case Signature.empty:
                        # 如果采样器没有指定返回类型，则抛出异常
                        raise BadSignatureError(
                            f"Sampler {sampler} must have annotated return type!\nGot {sig.return_annotation}"
                        )
                    case invalid_sampler_ret:
                        # 如果采样器的返回类型注解无效，则抛出异常
                        raise BadSignatureError(
                            f"Sampler {sampler} has invalid return type annotation(s)!\n"
                            f"Must be {SensorDataSequence} or {SensorData} but got {invalid_sampler_ret}"
                        )
            except TypeError:
                raise BadSignatureError(f"Sampler {sampler} has invalid signature!\nGot {sig}")
        return self  # 返回更新后的实例自身

    def construct_updater(self, usages: List[SamplerUsage]) -> Callable[[], Tuple] | Callable[[], SensorData]:
        """
        Constructs an updater function based on the given list of sampler usages.

        Args:
            usages (List[SamplerUsage]): A list of sampler usages.

        Returns:
            Callable[[], Tuple]: The constructed updater function.

        Raises:
            ValueError: If the number of sampler usages does not match the number of samplers.
            RuntimeError: If an unsupported sampler type is encountered.
        """
        self.update_sampler_types()
        if len(self.sampler_types) != len(self.samplers):
            raise ValueError(
                f"Number of sampler usages ({len(usages)}) does not match number of samplers ({len(self.samplers)}), have you used the update_sampler_types() method?"
            )
        if len(usages) == 0:
            raise RequirementError("Can't resolve the empty Usage List")
        update_funcs: List[UpdaterClosure] = []

        for usage in usages:
            _logger.debug(
                f"Sampler_type: {self.sampler_types[usage.used_sampler_index]}|Required: {usage.required_data_indexes}"
            )
            sampler = self.samplers[usage.used_sampler_index]
            sampler_type = self.sampler_types[usage.used_sampler_index]
            match sampler_type:
                case SamplerType.SEQ_SAMPLER:
                    update_funcs.append(self.resolve_seq_sampler(sampler, usage.required_data_indexes))
                case SamplerType.IDX_SAMPLER:
                    update_funcs.append(self.resolve_idx_sampler(sampler, usage.required_data_indexes))
                case SamplerType.DRC_SAMPLER:
                    update_funcs.append(self.resolve_drc_sampler(sampler, usage.required_data_indexes))
                case _:
                    raise RuntimeError(f"Unsupported sampler type: {sampler_type}")

        # TODO allow make flatten ret
        if len(update_funcs) == 1:
            return update_funcs[0]
        eval_kwargs = {f"func_{i}": update_funcs[i] for i in range(len(update_funcs))}
        func_call_strings = [f"{func}()" for func in eval_kwargs]
        eval_string = "lambda:" + "(" + f",".join(func_call_strings) + ")"
        eval_obj = eval(eval_string, eval_kwargs)
        return eval_obj

    def construct_inlined_function(
        self,
        usages: List[SamplerUsage],
        judging_source: List[SourceCode] | SourceCode,
        extra_context: Dict[str, Any] = None,
        return_type_varname: Optional[str] = None,
        return_raw: bool = False,
    ) -> Callable[[], Any] | Tuple[str, Dict[str, Any]]:
        """
        构建一个内联函数，根据给定的采样器使用情况、判断源代码和额外上下文信息。

        Args:
            usages: 一个SamplerUsage对象列表，描述了如何使用采样器。
            judging_source: 判断源代码，可以是单个SourceCode对象或其列表。这些代码将被嵌入到生成的函数中。
            extra_context: 一个可选的字典，包含执行判断源代码时需要的额外上下文变量。
            return_type_varname: 一个可选的字符串，指定生成函数的返回类型变量名。如果提供，必须在extra_context中定义。
            return_raw: 一个布尔值，指示是否返回未执行的函数源码和使用的采样器信息。

        Returns:
            如果return_raw为True，则返回函数源码和使用的采样器信息的元组；
            否则，返回一个可调用对象，该对象在执行时根据判断源代码和提供的采样器数据返回结果。

        Raises:
            RequirementError: 如果返回类型变量名在extra_context中未定义，或者判断源中缺少必要的占位符。
            RuntimeError: 如果遇到了不支持的采样器类型。
        """
        if return_type_varname and return_type_varname not in extra_context:
            raise RequirementError(
                f'The return_type_varname "{return_type_varname}" is not defined in extra_context: {extra_context}.'
            )

        # 将judging_source统一处理为字符串格式
        judging_source: str = judging_source if isinstance(judging_source, str) else "\n ".join(judging_source)

        # 定义返回标识符，检查判断源中是否包含该标识符
        RET_IDENTIFIER: str = "ret"
        if RET_IDENTIFIER not in judging_source:
            raise RequirementError(
                f"Can't find {RET_IDENTIFIER} in judging source, you must define a variable named {RET_IDENTIFIER} in your judging source!\n"
                f"Since it will be used as the return value of the judge function."
            )

        # 匹配使用情况与相应的采样器和采样器类型
        _logger.debug("Match Usage with corresponding samplers and sampler_types")
        compile_context: Dict[str, Any] = {
            f"func_{i}": self.samplers[usage.used_sampler_index] for i, usage in enumerate(usages)
        }
        _logger.debug(f"Matched samplers: {compile_context}")
        self.update_sampler_types()
        used_sampler_types = [self.sampler_types[usage.used_sampler_index] for usage in usages]
        _logger.debug(f"Matched sampler_types: {used_sampler_types}")

        # 根据采样器类型，为采样器数据创建索引表达式
        sampler_temp_var_names_mapping: Dict[str, str] = {}
        indexed_expressions: List[str] = []
        for usage, sampler_type, func_name in zip(usages, used_sampler_types, compile_context):

            match sampler_type:
                case SamplerType.SEQ_SAMPLER:
                    sampler_temp_var_names_mapping[temp_name := f"{func_name}_temp"] = func_name
                    indexed_expressions.extend(self._index_for_seq_sampler_data(temp_name, usage.required_data_indexes))
                case SamplerType.IDX_SAMPLER:
                    indexed_expressions.extend(self._index_for_idx_sampler_data(func_name, usage.required_data_indexes))
                case SamplerType.DRC_SAMPLER:
                    sampler_temp_var_names_mapping[temp_name := f"{func_name}_temp"] = func_name
                    indexed_expressions.extend(self._index_for_drc_sampler_data(temp_name, usage.required_data_indexes))
                case _:
                    raise RuntimeError(f"Unsupported sampler type: {sampler_type}")
        _logger.debug(f"Created {len(indexed_expressions)} indexed expressions.")

        # 创建占位变量并检查是否所有占位符都被包含在判断源中
        placebo_var_names = [f"s{i}" for i in range(len(indexed_expressions))]
        _logger.debug(f"Created {len(placebo_var_names)} placebo variables.")
        _logger.debug("Checking that all placeholders are included in judging_source")
        if not_included := [placebo for placebo in placebo_var_names if placebo not in judging_source]:
            raise RequirementError(
                f"Judging source must have all placeholders: {placebo_var_names} in judging_source\n"
                f"Missing: {not_included}"
            )

        # 替换判断源中的占位符为索引表达式
        for placebo, expr in zip(placebo_var_names, indexed_expressions):
            _logger.debug(f'Replacing "{placebo}" with "{expr}".')
            judging_source = judging_source.replace(placebo, expr)

        # 创建临时变量的源代码，用于在执行环境中存储采样器的返回值
        temp_var_source: str = (
            ",".join(sampler_temp_var_names_mapping.keys())
            + "="
            + ",".join([f"{fname}()" for fname in sampler_temp_var_names_mapping.values()])
        )
        _logger.debug(f"Created temp_var_source: {temp_var_source}")

        # 构建完整的函数源码并编译执行
        function_head: str = f"def _func() -> {return_type_varname}:" if return_type_varname else "def _func():"
        func_source = f"{function_head}\n" f" {temp_var_source}\n" f" {judging_source}\n" f" return {RET_IDENTIFIER}"
        _logger.debug(f"Created func_source: {func_source}")
        _logger.debug("Compiling func_source")

        # 更新执行环境中的采样器和额外上下文信息

        compile_context.update(extra_context) if extra_context else None
        if return_raw:
            return func_source, compile_context
        exec(func_source, compile_context)  # exec the source with the context
        func_obj: Callable[[], bool] = compile_context.get("_func")
        _logger.debug(f"Succeed, compiled func_obj: {func_obj}")
        return func_obj

    @staticmethod
    def _index_for_seq_sampler_data(data_var_name: str, required: List[int] | int) -> List[str]:
        """
        A function that generates a list of indexed expressions based on the given data variable name and a list of required indexes.

        Args:
            data_var_name (str): The name of the data variable.
            required (List[int] | int): Either a single integer representing the required sequence length or a list of required data indexes.

        Returns:
            List[str]: A list of indexed expressions based on the data variable name and required indexes.
        """
        match required:
            case int(required_seq_length):
                final_required_data_indexes = range(required_seq_length)
            case list(required_data_indexes):
                final_required_data_indexes = required_data_indexes
            case _:
                raise RequirementError(
                    f"Unknown Input, arg::required has to be either list[int] or int, got {required}"
                )
        return [f"{data_var_name}[{i}]" for i in final_required_data_indexes]

    @staticmethod
    def _index_for_drc_sampler_data(data_var_name: str, required: List[int] | int) -> List[str]:
        """
        A function that generates a list of indexed expressions based on the given data variable name and a list of required indexes.

        Args:
            data_var_name (str): The name of the data variable.
            required (List[int] | int): Either a single integer representing the required sequence length or a list of required data indexes.

        Returns:
            List[str]: A list of indexed expressions based on the data variable name and required indexes.
        """
        match required:
            case []:
                raise RequirementError("Can't resolve the empty Usage List")
            case int(required_seq_length):
                final_required_data_indexes = range(required_seq_length)
            case list(required_data_indexes):
                final_required_data_indexes = required_data_indexes
            case _:
                raise RequirementError(
                    f"Unknown Input, arg::required has to be either list[int] or int, got {required}"
                )
        return [f"(({data_var_name}>>{i})&1)" for i in final_required_data_indexes]

    @staticmethod
    def _index_for_idx_sampler_data(func_var_name: str, required: List[int]) -> List[str]:
        """
        Generate a list of indexed expressions based on the given function variable name and a list of required indexes.

        Args:
            func_var_name (str): The name of the function variable.
            required (List[int]): A list of required indexes.

        Returns:
            List[str]: A list of indexed expressions based on the function variable name and required indexes.
        """
        return [f"{func_var_name}({i})" for i in required]

    @staticmethod
    def resolve_seq_sampler(sampler: SequenceSampler, required_data_indexes: Sequence[int]) -> UpdaterClosure:
        """
        Resolves the sampler based on the given sequence sampler and required data indexes.

        Args:
            sampler (SequenceSampler): The sequence sampler to resolve.
            required_data_indexes (Sequence[int]): The required data indexes.

        Returns:
            UpdaterClosure: A callable that returns a tuple of SensorData objects
            or a single SensorData object based on the number of required data indexes.

        Raises:
            None

        Examples:
            sampler = SequenceSampler()
            required_data_indexes = []
            resolved_sampler = resolve_seq_sampler(sampler, required_data_indexes)
            data = resolved_sampler()  # Returns a tuple of all SensorData objects

            sampler = SequenceSampler()
            required_data_indexes = [0]
            resolved_sampler = resolve_seq_sampler(sampler, required_data_indexes)
            data = resolved_sampler()  # Returns the SensorData object at index 0

            sampler = SequenceSampler()
            required_data_indexes = [0, 1, 2]
            resolved_sampler = resolve_seq_sampler(sampler, required_data_indexes)
            data = resolved_sampler()  # Returns a tuple of SensorData objects at indexes 0, 1, and 2
        """
        match len(required_data_indexes):
            case 0:
                # 0 means require all data
                return sampler  # Callable[[],SensorDataSequence]
            case 1:
                # 1 means require a specific data
                unique_index = required_data_indexes[0]
                return lambda: sampler()[unique_index]  # Callable[[], SensorData]
            case _:
                # >1 means require multiple data
                required_indexes = required_data_indexes

                def _fun() -> Tuple[SensorData, ...]:
                    data = sampler()
                    return tuple(data[i] for i in required_indexes)

                return _fun  # Callable[[],SensorDataSequence]

    @staticmethod
    def resolve_idx_sampler(sampler: IndexedSampler, required_data_indexes: Sequence[int]) -> UpdaterClosure:
        """
        Resolves the indexed sampler based on the given sampler usage.

        Args:
            sampler (IndexedSampler): The indexed sampler to resolve.
            required_data_indexes (Sequence[int]): The required data indexes.

        Returns:
            UpdaterClosure: A callable that returns a tuple of SensorData objects or a single SensorData object based on the number of required data indexes.

        Raises:
            RequirementError: If no required data indexes are specified.

        Examples:
            sampler = IndexedSampler()
            required_data_indexes = []
            resolved_sampler = resolve_idx_sampler(sampler, required_data_indexes)  # Raises RequirementError

            sampler = IndexedSampler()
            required_data_indexes = [0]
            resolved_sampler = resolve_idx_sampler(sampler, required_data_indexes)
            data = resolved_sampler()  # Returns the SensorData object at index 0

            sampler = IndexedSampler()
            required_data_indexes = [0, 1, 2]
            resolved_sampler = resolve_idx_sampler(sampler, required_data_indexes)
            data = resolved_sampler()  # Returns a tuple of the SensorData objects at indexes 0, 1, and 2
        """

        match len(required_data_indexes):
            case 0:
                raise RequirementError("Must specify at least one required data index")
            case 1:
                # 1 means require a specific data
                unique_index = required_data_indexes[0]
                return lambda: sampler(unique_index)  # Callable[[], SensorData]
            case _:
                # >1 means require multiple data
                required_indexes = required_data_indexes
                return lambda: tuple(sampler(ri) for ri in required_indexes)  # Callable[[], SensorDataSequence]

    @staticmethod
    def resolve_drc_sampler(sampler: DirectSampler, required_data_indexes: Sequence[int]) -> UpdaterClosure:
        """
        Resolves the direct sampler based on the given direct sampler and required data indexes.

        Args:
            sampler (DirectSampler): The direct sampler to resolve.
            required_data_indexes (Sequence[int]): The required data indexes.

        Returns:
            UpdaterClosure: A callable that returns a tuple of SensorData objects or a single SensorData object based on the number of required data indexes.
        """

        match len(required_data_indexes):
            case 0:
                return sampler  # Callable[[], SensorData]
            case 1:
                # 1 means require a specific data
                unique_index = required_data_indexes[0]
                return lambda: (sampler() >> unique_index) & 1
            case _:
                # >1 means require multiple data

                def _fun() -> SensorDataSequence:
                    temp_seq = sampler()
                    return tuple((temp_seq >> ri) & 1 for ri in required_data_indexes)

                return _fun
