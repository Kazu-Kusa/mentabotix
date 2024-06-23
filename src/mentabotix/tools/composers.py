from enum import Enum
from itertools import zip_longest
from typing import List, Tuple, Callable, Optional, Self, Type, TypeVar, Dict, Hashable, Iterable

from numpy.random import random

from ..modules.botix import MovingState, MovingTransition, __PLACE_HOLDER__

StateTransitionPack = Tuple[List[MovingState], List[MovingTransition]]

UnitType = TypeVar("UnitType", Type[MovingState], Type[MovingTransition])
SupportClone = TypeVar("SupportClone", MovingState, MovingTransition)
from terminaltables import SingleTable


class MovingChainComposer:
    """
    A class that manages the composition of moving states and transitions.

    Properties:
    - last_state (MovingState): The last state object in the chain container.
    - last_transition (MovingTransition): The last transition object in the chain container if it exists, otherwise None.
    - next_need (UnitType): The next unit type to be added to the chain container.
    """

    @property
    def last_state(self) -> MovingState:
        """
        Returns the last state object in the chain container.

        Returns:
            MovingState: The last state object in the chain container, or None if the container is empty.
        """
        container = self._chain_container[MovingState]
        return container[-1] if container else None

    @property
    def last_transition(self) -> MovingTransition:
        """
        Returns the last transition object in the chain container.

        Returns:
            MovingTransition: The last transition object in the chain container if it exists, otherwise None.
        """
        container = self._chain_container[MovingTransition]
        return container[-1] if container else None

    @property
    def next_need(self) -> UnitType:
        return self._type_container[0]

    def __init__(self) -> None:
        self._type_container: List[UnitType] = [MovingState, MovingTransition]

        self._chain_container: Dict[Type, List] = {MovingState: [], MovingTransition: []}

    def _flip(self):
        self._type_container.reverse()

    def init_container(self) -> Self:
        """
        Initializes the chain container by clearing all the lists in the `_chain_container` dictionary.
        If the `next_need` is not `MovingState`, the `_type_container` list is reversed.

        Parameters:
            self (MovingChainComposer): The instance of the class.

        Returns:
            Self: The current instance of the class.
        """
        if self.next_need != MovingState:
            self._flip()
        for cont in self._chain_container.values():
            cont.clear()
        return self

    def export_structure(self) -> StateTransitionPack:
        """
        Exports the structure of the chain container.

        Returns:
            StateTransitionPack: A tuple containing two lists, the first list contains all the MovingState objects in the chain container, and the second list contains all the MovingTransition objects in the chain container.

        Side Effects:
            - Initializes the chain container by calling the `init_container` method.
        """
        ret_pack = list(self._chain_container.get(MovingState)), list(self._chain_container.get(MovingTransition))
        self.init_container()
        return ret_pack

    def add(self, unit: MovingState | MovingTransition, register_case: Optional[Hashable] = __PLACE_HOLDER__) -> Self:
        """
        Adds a `MovingState` or `MovingTransition` object to the chain container.

        Args:
            register_case (Optional[Hashable]): The case to register the state or transition with.
            unit (MovingState | MovingTransition): The unit to be added to the chain container.

        Returns:
            Self: The current instance of the class.

        Raises:
            ValueError: If the type of the unit does not match the next need.
            RuntimeError: If the type of the unit is neither `MovingState` nor `MovingTransition`.
        """
        unit_type = type(unit)
        if unit_type != self.next_need:
            raise ValueError(f"Need {self.next_need}, got {type(unit)}, which is {repr(unit)}!")
        elif unit_type == MovingState:
            self._chain_container[MovingState].append(unit)
            if self.last_transition and unit not in self.last_transition.to_states.values():
                self.last_transition.to_states[register_case] = unit
        elif unit_type == MovingTransition:
            unit.from_states.append(self.last_state) if self.last_state not in unit.from_states else None
            self._chain_container[MovingTransition].append(unit)
        else:
            raise RuntimeError("Should never reach here!")
        self._flip()
        return self

    def concat(
        self,
        states: List[MovingState],
        transitions: List[MovingTransition],
        register_case: Optional[Hashable] = __PLACE_HOLDER__,
    ) -> Self:
        """
        Concatenates the given list of states and transitions into the current data structure.

        Args:
            states: A list of MovingState objects representing the states to concatenate.
            transitions: A list of MovingTransition objects representing the transitions to concatenate.
            register_case (Optional[Hashable]): The case to register the state or transition with.
        Returns:
            Returns the concatenated data structure itself.

        Raises:
            ValueError if no states or transitions are provided.
            ValueError if the initial condition of the transition's starting state doesn't match the expectation.
            ValueError if the number of states and transitions don't match by no more than 1.

        This function checks the validity of the input and connects the states and transitions accordingly.
        It handles different scenarios based on whether there's an initial state and the difference between the lengths of states and transitions.
        """
        state_len, trans_len = len(states), len(transitions)
        # Check if at least one state and one transition are provided
        if not (states or transitions):
            raise ValueError("Need at least one state and one transition!")
        elif not abs(state_len - trans_len) <= 1:
            raise ValueError(
                f"The number of states and transitions don't match by no more than 1! "
                f"Got {state_len} states and {trans_len} transitions!"
            )

        # Initialize the head transition and check if it has a starting state
        head_transition = transitions[0]
        has_start_state = bool(head_transition.from_states)
        head_state = states[0]

        # Validate the expected next element type (state or transition) based on the initial condition
        if self.next_need == MovingTransition and has_start_state:
            raise ValueError(f"Need {MovingTransition}, but got {head_state}!")
        elif self.next_need == MovingState and not has_start_state:
            raise ValueError(f"Need {MovingState}, but the head of the seq is {repr(head_transition)}!")
        elif self.next_need == MovingState and head_state not in head_transition.from_states:
            raise ValueError(f"The transition {repr(head_transition)} should only start from {head_state}!")

        # Reset the states and transitions lists for concatenation
        states: List[MovingState] = list(states)
        transitions: List[MovingTransition] = list(transitions)

        if has_start_state:

            start_state = states.pop(0)
            self.add(start_state, register_case)

        for sta, trans in zip_longest(states, transitions):
            self._add_with_connection_check(trans) if trans else None
            self._add_with_connection_check(sta) if sta else None
        return self

    def _add_with_connection_check(self, unit: MovingState | MovingTransition) -> Self:
        """
        Adds a `MovingState` or `MovingTransition` object to the chain container.

        Args:
            unit (MovingState | MovingTransition): The unit to be added to the chain container.

        Returns:
            Self: The current instance of the class.

        Raises:
            ValueError: If the type of the unit does not match the next need.
            RuntimeError: If the type of the unit is neither `MovingState` nor `MovingTransition`.
        """
        unit_type = type(unit)
        if unit_type != self.next_need:
            raise ValueError(f"Need {self.next_need}, got {type(unit)}, which is {repr(unit)}!")
        elif unit_type == MovingState:
            self._chain_container[MovingState].append(unit)
            if self.last_transition and unit not in self.last_transition.to_states.values():
                raise ValueError(
                    f"Connection break! The state {unit} should be in the to_states of the transition {self.last_transition}!"
                )
        elif unit_type == MovingTransition:
            if self.last_state not in unit.from_states:
                raise ValueError(
                    f"Connection break! The state {self.last_state} should be in the from_states of the transition {repr(unit)}!"
                )

            self._chain_container[MovingTransition].append(unit)
        else:
            raise RuntimeError("Should never reach here!")
        self._flip()
        return self


KT = TypeVar("KT", bound=Hashable)


class CaseRegistry:
    """
    Initializes a new instance of the CaseRegistry class.

    Args:
        case_dict (Dict[KT, MovingState]): A dictionary mapping keys of type KT to MovingState objects. Defaults to an empty dictionary if not provided.
        to_cover (Type[Enum]): The enumeration type to cover.


    """

    def __init__(
        self,
        to_cover: Type[Enum],
        case_dict: Optional[Dict[KT, MovingState]] = None,
    ):
        self._case_dict: Dict[KT, MovingState] = case_dict or {}
        self._to_cover: Type[Enum] = to_cover

    @property
    def case_dict(self) -> Dict[KT, MovingState]:
        """

        Returns:
            Dict[KT, MovingState]: A dictionary mapping keys of type KT to MovingState objects.

        """
        return self._case_dict

    @property
    def table(self) -> str:
        """
        Returns a string representation of the case registry table.

        This property retrieves the case registry dictionary and converts it into a list of lists, where each inner list
        contains the case name and its corresponding target state. The case registry title is then used to generate
        the table title.

        Returns:
            str: A string representation of the case registry table.
        """
        data = [["Case", "Target State"]] + list(self._case_dict.items())
        return SingleTable(data, title=f"CaseReg-of-{self._to_cover}").table

    def reassign(self, to_cover: Type[Enum]):
        """
        Reassigns the `_to_cover` attribute of the `CaseRegistry` instance to the specified `to_cover` value.

        Parameters:
            to_cover (Type[Enum]): The new value for the `_to_cover` attribute.

        Returns:
            None
        """
        self._to_cover = to_cover

    def init_container(self) -> Self:
        """
        Initializes the `_case_dict` attribute by clearing all its elements.

        Returns:
            Self: The current instance of the class.
        """
        self._case_dict.clear()
        return self

    def export(self, force: bool = False) -> Dict[KT, MovingState]:
        """
        Export the registered cases and their corresponding target states as a dictionary.

        Returns:
            Dict[KT, MovingState]: A dictionary where the keys are the registered cases and the values are the corresponding target states.

        Raises:
            ValueError: If any of the cases are not registered or if any of the cases are not allowed.
        """
        if not force:
            not_registered = list(filter(lambda x: x.value not in self._case_dict, self._to_cover))

            if not_registered:
                raise ValueError(f"Case not registered: {not_registered}")
        temp = dict(self._case_dict)
        self.init_container()
        return temp

    def register(self, case: Enum, target_state: MovingState) -> Self:
        """
        Register a case with its corresponding target state.

        Args:
            case (KT): The case to be registered.
            target_state (MovingState): The target state associated with the case.

        Raises:
            ValueError: If the case is already registered or if the case is not allowed.

        Returns:
            None
        """
        case_value = case.value
        if case_value in self._case_dict:
            raise ValueError(f"Case already registered: {case}")
        elif not isinstance(case, self._to_cover):
            raise ValueError(f"Case not in {list(self._to_cover)}, got {case_value}.")
        self._case_dict[case_value] = target_state
        return self

    def batch_register(self, cases: List[Enum], target_state: MovingState) -> Self:
        """
        Register multiple cases with the same target state.

        Args:
            cases (List[Enum]): A list of enum types representing the cases to be registered.
            target_state (MovingState): The target state to be associated with the cases.

        Raises:
            ValueError: If any of the cases are already registered.

        Returns:
            None
        """
        already_registered = list(filter(lambda x: x.value in self._case_dict, cases))
        if already_registered:
            raise ValueError(f"Case already registered: {already_registered}")

        for case in cases:
            self._case_dict[case.value] = target_state
        return self

    def unregister(self, case: KT) -> Self:
        """
        Unregister a case.

        Args:
            case (KT): The case to be unregistered.

        Raises:
            ValueError: If the case is not registered.

        Returns:
            None
        """
        if case not in self._case_dict:
            raise ValueError(f"Case not registered: {case}")
        del self._case_dict[case]
        return self

    def unregister_batch(self, cases: List[Enum]) -> Self:
        """
        Unregister multiple cases.

        Args:
            cases (List[Enum]): A list of enum types representing the cases to be unregistered.

        Raises:
            ValueError: If any of the cases are not registered.

        Returns:
            None
        """
        for case in cases:
            self.unregister(case)
        return self


def straight_chain(
    start_speed: int,
    end_speed: int,
    duration: float,
    power_exponent: float = 1.0,
    interval: float = 0.07,
    breaker: Optional[Callable[[], bool]] = None,
    lead_time: float = 0.05,
    state_on_break: Optional[MovingState] = MovingState(0),
) -> StateTransitionPack:
    """
    A function that calculates the states and transitions for a straight chain based on the input parameters.

    Args:
        lead_time (float): The lead time of the chain.
        start_speed (int): The initial speed of the chain.
        end_speed (int): The final speed of the chain.
        duration (float): The total duration of the chain movement.
        power_exponent (float, optional): The power exponent used in the calculation. Defaults to 1.0.
        interval (float, optional): The interval used in the calculation. Defaults to 0.07.
        breaker (Optional[Callable[[], bool]], optional): A function to determine if the transition should be broken.
        Defaults to None.
        state_on_break (Optional[MovingState], optional): The state to transition to if the chain is broken.
        Defaults to MovingState(0).

    Returns:
        StateTransitionPack: A tuple containing the list of states and transitions for the straight chain.
    """
    duration -= lead_time
    comp = MovingChainComposer().init_container().add(MovingState(start_speed))

    # Calculate the deviation in speed for uniform distribution across the duration
    deviation = end_speed - start_speed
    # Generate a sequence of speeds based on the given parameters
    step_len = interval / (duration - lead_time)

    percentages = [step_len] * int(duration // interval) + [1 % step_len]
    progress = 0

    # Handle different scenarios based on whether a breaker function is provided
    match breaker:
        case None:
            # If no breaker function, create transitions without breaking condition
            for percentage in percentages:
                progress += percentage

                (
                    comp.add(MovingTransition(duration=duration * percentage)).add(
                        MovingState(start_speed + round(deviation * progress**power_exponent))
                    )
                )

            return comp.export_structure()

        case break_function if callable(break_function):
            # If a breaker function is provided, create transitions that can be broken
            for percentage in percentages:
                progress += percentage
                (
                    comp.add(
                        MovingTransition(
                            duration=duration * percentage, breaker=break_function, to_states={True: state_on_break}
                        )
                    ).add(MovingState(start_speed + round(deviation * progress**power_exponent)), False)
                )

            return comp.export_structure()
        case _:
            # If breaker is neither None nor callable, raise an error
            raise ValueError("breaker must be callable or None")


def scanning_chain():
    """
    A function that calculates the states and transitions for a scanning chain.
    Returns:

    """
    raise NotImplementedError


def snaking_chain():
    """
    A function that calculates the states and transitions for a snaking chain.
    Returns:

    """
    raise NotImplementedError


def random_lr_turn_branch(
    start_state: MovingState,
    end_state: MovingState,
    start_state_duration: float,
    turn_speed: int,
    turn_duration: float,
    turn_left_prob: 0.5,
) -> Tuple[MovingTransition, MovingTransition]:
    """
    A function that generates random left and right turn states based on probabilities.
    It creates two transition states, a start transition, and a turn transition.

    Parameters:
        start_state (MovingState): The initial state.
        end_state (MovingState): The final state.
        start_state_duration (float): The duration of the start state.
        turn_speed (int): The speed of the turn.
        turn_duration (float): The duration of the turn.
        turn_left_prob (float): The probability of turning left.

    Returns:
        tuple: A tuple containing the start transition and the turn transition.
    """
    if not 0 < turn_left_prob < 1:
        raise ValueError("turn_speed must be between 0 and 1")

    def _die() -> bool:
        return random() > turn_left_prob

    left_turn_state = MovingState.turn("l", turn_speed)
    right_turn_state = MovingState.turn("r", turn_speed)
    start_transition = MovingTransition(
        from_states=start_state,
        to_states={False: right_turn_state, True: left_turn_state},
        duration=start_state_duration,
        breaker=_die,
    )
    turn_transition = MovingTransition(
        from_states=[left_turn_state, right_turn_state],
        to_states={__PLACE_HOLDER__: end_state},
        duration=turn_duration,
    )
    return start_transition, turn_transition


def copy(iters: Iterable[SupportClone]) -> List[SupportClone]:
    """
    Returns a new list containing clones of the elements in the input iterable.

    Args:
        iters (Iterable[SupportClone]): An iterable of objects that support the `clone()` method.

    Returns:
        List[SupportClone]: A new list containing clones of the input objects.
    """
    return [i.clone() for i in iters]
