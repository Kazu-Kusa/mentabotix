from typing import List, Tuple, Callable, Optional, Self, Type, TypeVar, Dict

from numpy import arange
from numpy.random import random

from ..modules.botix import MovingState, MovingTransition, __PLACE_HOLDER__

StateTransitionPack = Tuple[List[MovingState], List[MovingTransition]]

UnitType = TypeVar("UnitType", Type[MovingState], Type[MovingTransition])


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

    def add(self, unit: MovingState | MovingTransition) -> Self:
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
            raise ValueError(f"Need {self.next_need}, got {unit}!")
        elif unit_type == MovingState:
            self._chain_container[MovingState].append(unit)
            if self.last_transition:
                self.last_transition.to_states[__PLACE_HOLDER__] = unit
        elif unit_type == MovingTransition:
            unit.from_states.append(self.last_state)
            self._chain_container[MovingTransition].append(unit)
        else:
            raise RuntimeError("Should never reach here!")
        self._flip()
        return self


def straight_chain(
    start_speed: int,
    end_speed: int,
    duration: float,
    power_exponent: float = 1.0,
    interval: float = 0.07,
    breaker: Optional[Callable[[], bool]] = None,
    state_on_break: Optional[MovingState] = MovingState(0),
) -> StateTransitionPack:
    """
    A function that calculates the states and transitions for a straight chain based on the input parameters.

    Args:
        start_speed (int): The initial speed of the chain.
        end_speed (int): The final speed of the chain.
        duration (float): The total duration of the chain movement.
        power_exponent (float, optional): The power exponent used in the calculation. Defaults to 1.0.
        interval (float, optional): The interval used in the calculation. Defaults to 0.07.
        breaker (Optional[Callable[[], bool]], optional): A function to determine if the transition should be broken. Defaults to None.
        state_on_break (Optional[MovingState], optional): The state to transition to if the chain is broken. Defaults to MovingState(0).

    Returns:
        StateTransitionPack: A tuple containing the list of states and transitions for the straight chain.
    """
    # Initialize the list of states with the starting state
    states: List[MovingState] = [MovingState(start_speed)]
    # Initialize an empty list for transitions
    transitions: List[MovingTransition] = []

    # Calculate the deviation in speed for uniform distribution across the duration
    deviation = end_speed - start_speed
    # Generate a sequence of speeds based on the given parameters
    speed_seq = [
        start_speed + round(deviation * x**power_exponent) for x in arange(0, 1.0, interval / (duration - interval))
    ]

    # Handle different scenarios based on whether a breaker function is provided
    match breaker:
        case None:
            # If no breaker function, create transitions without breaking condition
            for cur_speed in speed_seq:
                last_state = states[-1]
                cur_state = MovingState(cur_speed)
                cur_transition = MovingTransition(from_states=last_state, to_states=cur_state, duration=interval)
                states.append(cur_state)
                transitions.append(cur_transition)
            return states, transitions

        case break_function if callable(break_function):
            # If a breaker function is provided, create transitions that can be broken
            for cur_speed in speed_seq:
                last_state = states[-1]
                cur_state = MovingState(cur_speed)
                cur_transition = MovingTransition(
                    from_states=last_state,
                    to_states={False: cur_state, True: state_on_break},
                    duration=interval,
                    breaker=break_function,
                )
                states.append(cur_state)
                transitions.append(cur_transition)
            # Append the break state to the states list
            states.append(state_on_break)
            return states, transitions
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
