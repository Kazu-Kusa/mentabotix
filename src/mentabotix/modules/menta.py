from enum import Enum
from inspect import signature, Signature
from typing import Callable, TypeAlias, SupportsIndex, List, Optional, Tuple, Sequence, NamedTuple, Union, Self,TypeVar
from ctypes import Array

SensorData = TypeVar("SensorData")
# basically, no restrictions, support py objects or ctypes._CData variants
SensorDataSequence: TypeAlias = Sequence[SensorData] | Array


SequenceSampler: TypeAlias = Callable[[], SensorDataSequence]
IndexedSampler: TypeAlias = Callable[[SupportsIndex], SensorData]
DirectSampler: TypeAlias = Callable[[], SensorData]

Sampler: TypeAlias = Union[SequenceSampler, IndexedSampler, DirectSampler]



class SamplerType(Enum):
    SEQ_SAMPLER: int = 1
    IDX_SAMPLER: int = 2
    DRC_SAMPLER: int = 3


class SamplerUsage(NamedTuple):
    used_sampler_index: SupportsIndex
    required_data_indexes: Sequence[SupportsIndex]


class Menta:
    __supported__=("value","__float__","__int__","__bytes__") #reserverd to check the return type of the sampler
    def __init__(
        self,
        samplers: Optional[List[Sampler]] = None,
    ):
        self.samplers: List[Sampler] = samplers or []
        self.sampler_types: List[SamplerType] = []
        self.update_sampler_types()

    def update_sampler_types(self) -> Self:
        """
        Updates the sampler types of the object based on the samplers in the object.

        Returns:
            Self: The updated object.
        """
        self.sampler_types.clear()
        for sampler in self.samplers:
            sig = signature(sampler)

            match sig.return_annotation:
                case seq_sampler_ret if isinstance(seq_sampler_ret, (Sequence,Array)) and len(sig.parameters) == 0:
                    self.sampler_types.append(SamplerType.SEQ_SAMPLER)
                case idx_sampler_ret if  len(sig.parameters) == 1 and (any(hasattr(idx_sampler_ret, ret_type) for ret_type in self.__supported__)):
                    self.sampler_types.append(SamplerType.IDX_SAMPLER)
                case drc_sampler_ret if  len(sig.parameters) == 0 and (any(hasattr(drc_sampler_ret, ret_type) for ret_type in self.__supported__)):
                    self.sampler_types.append(SamplerType.DRC_SAMPLER)
                case Signature.empty:
                    raise ValueError(f"Sampler {sampler} must have annotated return type!\nGot {sig.return_annotation}")
                case invalid_sampler_ret:
                    raise ValueError(
                        f"Sampler {sampler} has invalid return type annotation(s)!\bMust be {SensorDataSequence} or {SensorData} but got {invalid_sampler_ret}"
                    )
        return self

    def construct_updater(self, usages: List[SamplerUsage]) -> Callable[[], Tuple]:
        if len(usages) != len(self.samplers):
            raise ValueError(
                f"Number of sampler usages ({len(usages)}) does not match number of samplers ({len(self.samplers)}), have you used the update_sampler_types() method?"
            )
        used_samplers: List[Tuple[Sampler, SamplerType]] = []
        for i in range(len(self.samplers)):
            used_samplers.append((self.samplers[i], self.sampler_types[i]))

        def updater() -> Tuple:
            return tuple(
                [
                    sampler()
                    for sampler in self.samplers
                    for usage in usages
                    if usage.used_sampler_index == self.samplers.index(sampler)
                ]
            )

        return updater

    def _resolve_seq_sampler(self,usage:SamplerUsage) -> Callable[[], Sequence[SensorData]|SensorData]:

        match len(usage.required_data_indexes):
            case 0:
                sampler=
                return lambda : [data.value for data in self.s]
        def _select_data