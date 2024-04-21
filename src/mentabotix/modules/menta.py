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
    TypeVar,
    SupportsInt,
    SupportsFloat,
)

from .exceptions import BadSignatureError, RequirementError, SamplerTypeError

SensorData = TypeVar("SensorData", float, int)
# basically, no restrictions, support py objects or ctypes._CData variants
SensorDataSequence: TypeAlias = Sequence[SensorData]
UpdaterClosure = Callable[[], SensorDataSequence] | Callable[[], SensorData]

SequenceSampler: TypeAlias = Callable[[], SensorDataSequence]
IndexedSampler: TypeAlias = Callable[[int], SensorData]
DirectSampler: TypeAlias = Callable[[], SensorData]

Sampler: TypeAlias = Union[SequenceSampler, IndexedSampler, DirectSampler]


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

            match sig.return_annotation:  # 根据采样器的返回类型进行匹配
                case seq_sampler_ret if issubclass(seq_sampler_ret, Sequence) and len(sig.parameters) == 0:
                    # 如果返回类型是序列且无参数，则认定为序列采样器
                    self.sampler_types.append(SamplerType.SEQ_SAMPLER)
                case idx_sampler_ret if len(sig.parameters) == 1 and isinstance(idx_sampler_ret, self.__supported__):
                    # 如果返回类型是索引且只有一个参数，则认定为索引采样器
                    self.sampler_types.append(SamplerType.IDX_SAMPLER)
                case drc_sampler_ret if len(sig.parameters) == 0 and isinstance(drc_sampler_ret, self.__supported__):
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
        return self  # 返回更新后的实例自身

    def construct_updater(self, usages: List[SamplerUsage]) -> Callable[[], Tuple]:
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
        if len(usages) != len(self.samplers):
            raise ValueError(
                f"Number of sampler usages ({len(usages)}) does not match number of samplers ({len(self.samplers)}), have you used the update_sampler_types() method?"
            )
        update_funcs: List[UpdaterClosure] = []

        for usage in usages:
            match self.sampler_types[usage.used_sampler_index]:
                case SamplerType.SEQ_SAMPLER:
                    update_funcs.append(self._resolve_seq_sampler(usage))
                case SamplerType.IDX_SAMPLER:
                    update_funcs.append(self._resolve_idx_sampler(usage))
                case SamplerType.DRC_SAMPLER:
                    update_funcs.append(self._resolve_drc_sampler(usage))
                case _:
                    raise RuntimeError(f"Unsupported sampler type: {self.sampler_types[usage.used_sampler_index]}")

        def _updater() -> Tuple:
            return tuple(update_func() for update_func in update_funcs)

        return _updater

    def _resolve_seq_sampler(self, usage: SamplerUsage) -> UpdaterClosure:
        """
        Resolves the sampler based on the given sampler usage.

        Args:
            usage (SamplerUsage): The sampler usage object that contains the information about the used sampler and the required data indexes.

        Returns:
            Callable[[], Tuple[SensorData, ...]] | Callable[[], SensorData]: A callable that returns a tuple of SensorData objects or a single SensorData object based on the number of required data indexes.

        Raises:
            None

        Examples:
            usage = SamplerUsage(used_sampler_index=0, required_data_indexes=[])
            sampler = _resolve_seq_sampler(usage)
            data = sampler()  # Returns a tuple of all SensorData objects

            usage = SamplerUsage(used_sampler_index=0, required_data_indexes=[0])
            sampler = _resolve_seq_sampler(usage)
            data = sampler()  # Returns the SensorData object at index 0

            usage = SamplerUsage(used_sampler_index=0, required_data_indexes=[0, 1, 2])
            sampler = _resolve_seq_sampler(usage)
            data = sampler()  # Returns a tuple of SensorData objects at indexes 0, 1, and 2
        """

        if self.update_sampler_types().sampler_types[usage.used_sampler_index] != SamplerType.SEQ_SAMPLER:
            raise SamplerTypeError(
                f"Sampler at index {usage.used_sampler_index} is not a sequence sampler "
                f"but {self.sampler_types[usage.used_sampler_index]}"
            )
        sampler: SequenceSampler = self.samplers[usage.used_sampler_index]
        match len(usage.required_data_indexes):
            case 0:
                # 0 means require all data
                return sampler  # Callable[[],SensorDataSequence]
            case 1:
                # 1 means require a specific data
                unique_index = usage.required_data_indexes[0]
                return lambda: sampler()[unique_index]  # Callable[[], SensorData]
            case _:
                # >1 means require multiple data
                required_indexes = usage.required_data_indexes
                return lambda: tuple(sampler()[i] for i in required_indexes)  # Callable[[],SensorDataSequence]

    def _resolve_idx_sampler(self, usage: SamplerUsage) -> UpdaterClosure:
        """
        Resolves the indexed sampler based on the given sampler usage.

        Args:
            usage (SamplerUsage): The sampler usage object that contains the information about the used sampler and the required data indexes.

        Returns:
            Callable[[], Tuple[SensorData, ...]] | Callable[[], SensorData]: A callable that returns a tuple of SensorData objects or a single SensorData object based on the number of required data indexes.

        Raises:
            ValueError: If the sampler at the specified index is not an indexed sampler or if the sampler does not require any data.

        Examples:
            usage = SamplerUsage(used_sampler_index=0, required_data_indexes=[])
            sampler = _resolve_idx_sampler(usage)  # Raises ValueError

            usage = SamplerUsage(used_sampler_index=0, required_data_indexes=[0])
            sampler = _resolve_idx_sampler(usage)
            data = sampler()  # Returns the SensorData object at index 0

            usage = SamplerUsage(used_sampler_index=0, required_data_indexes=[0, 1, 2])
            sampler = _resolve_idx_sampler(usage)
            data = sampler()  # Returns a tuple of the SensorData objects at indexes 0, 1, and 2
        """
        if self.update_sampler_types().sampler_types[usage.used_sampler_index] != SamplerType.IDX_SAMPLER:
            raise SamplerTypeError(
                f"Sampler at index {usage.used_sampler_index} is not an indexed sampler "
                f"but {self.sampler_types[usage.used_sampler_index]}"
            )
        sampler: IndexedSampler = self.samplers[usage.used_sampler_index]
        match len(usage.required_data_indexes):
            case 0:
                raise RequirementError("Must specify at least one required data index")
            case 1:
                # 1 means require a specific data
                unique_index = usage.required_data_indexes[0]
                return lambda: sampler(unique_index)  # Callable[[], SensorData]
            case _:
                # >1 means require multiple data
                required_indexes = usage.required_data_indexes
                return lambda: tuple(sampler(ri) for ri in required_indexes)  # Callable[[], SensorDataSequence]

    def _resolve_drc_sampler(self, usage: SamplerUsage) -> UpdaterClosure:
        """
        Resolves the direct sampler based on the given sampler usage.

        Args:
            usage (SamplerUsage): The sampler usage object that contains the information about the used sampler and the required data indexes.

        Returns:
            UpdaterClosure: A callable that returns a SensorData object or a tuple of SensorData objects based on the number of required data indexes.

        Raises:
            SamplerTypeError: If the sampler at the specified index is not a direct sampler.

        Examples:
            usage = SamplerUsage(used_sampler_index=0, required_data_indexes=[])
            sampler = _resolve_drc_sampler(usage)
            data = sampler()  # Returns a SensorData object

            usage = SamplerUsage(used_sampler_index=0, required_data_indexes=[0])
            sampler = _resolve_drc_sampler(usage)
            data = sampler()  # Returns a SensorData object at index 0

            usage = SamplerUsage(used_sampler_index=0, required_data_indexes=[0, 1, 2])
            sampler = _resolve_drc_sampler(usage)
            data = sampler()  # Returns a tuple of SensorData objects at indexes 0, 1, and 2
        """
        if self.update_sampler_types().sampler_types[usage.used_sampler_index] != SamplerType.DRC_SAMPLER:
            raise SamplerTypeError(
                f"Sampler at index {usage.used_sampler_index} is not an indexed sampler "
                f"but {self.sampler_types[usage.used_sampler_index]}"
            )
        sampler: DirectSampler = self.samplers[usage.used_sampler_index]
        match len(usage.required_data_indexes):
            case 0:
                return sampler  # Callable[[], SensorData]
            case 1:
                # 1 means require a specific data
                unique_index = usage.required_data_indexes[0]
                return lambda: (sampler() << unique_index) & 1
            case _:
                # >1 means require multiple data
                required_indexes = usage.required_data_indexes

                def _fun() -> SensorDataSequence:
                    temp_seq = sampler()
                    return tuple(temp_seq << ri for ri in required_indexes)

                return _fun
