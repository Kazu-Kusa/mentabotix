from dataclasses import dataclass
from inspect import signature, Signature
from typing import (
    Tuple,
    TypeAlias,
    Self,
    Unpack,
    Literal,
    Any,
    Callable,
    Iterable,
    Hashable,
    TypeVar,
    Dict,
    Optional,
    List,
    ClassVar,
    Set,
)

import numpy as np
from bdmc import CloseLoopController

FullPattern: TypeAlias = Tuple[int]
LRPattern: TypeAlias = Tuple[int, int]
IndividualPattern: TypeAlias = Tuple[int, int, int, int]
FullExpressionPattern: TypeAlias = str
LRExpressionPattern: TypeAlias = Tuple[str, str]
IndividualExpressionPattern: TypeAlias = Tuple[str, str, str, str]
KT = TypeVar("KT", bound=Hashable)


__PLACE_HOLDER__ = "Hello World"


class MovingState:
    """
    Describes the movement state of the bot.
    Include:
    - halt: make a stop state,all wheels stop moving
    - straight: make a straight moving state,all wheels move in the same direction,same speed
    - turn: make a turning state,left and right wheels turn in in different direction,same speed
    - differential: make a differential state,all wheels move in the same direction,different speed
    - drift: make a drift state,all wheels except for a specific one drift in the same direction, specific speed

    """

    @dataclass
    class Config:
        """
        Configuration for the MovingState class.
        Args:
            track_width(int):The width of the track(AKA the distance between the wheels with a same axis). dimensionless number
            diagonal_multiplier(float):The multiplier for the diagonal speeds. dimensionless number.All designed for the drift movement.
        """

        track_width: int = 100
        diagonal_multiplier: float = 1.53
        # TODO: remove the dimensionless feature

    __state_id_counter__: ClassVar[int] = 0

    @property
    def state_id(self) -> int:
        """
        Returns the state identifier.

        Returns:
            int: The state identifier.
        """
        return self._identifier

    def __init__(
        self,
        *speeds: Unpack[FullPattern] | Unpack[LRPattern] | Unpack[IndividualPattern],
        speed_expressions: Optional[
            FullExpressionPattern | Unpack[LRExpressionPattern] | Unpack[IndividualExpressionPattern]
        ] = None,
    ) -> None:
        """
        Initialize the MovingState with speeds.

        Args:
            *speeds: A tuple representing the speed pattern.
                It should be one of the following types:
                    - FullPattern: A single integer representing full speed for all directions.
                    - LRPattern: A tuple of two integers representing left and right speeds.
                    - IndividualPattern: A tuple of four integers representing individual speeds for each direction.

        Raises:
            ValueError: If the provided speeds do not match any of the above patterns.
        """
        self._identifier = self.__state_id_counter__
        self.__state_id_counter__ += 1
        match bool(speed_expressions), bool(speeds):
            case True, False:
                self._speeds = None
                self._speed_expressions: FullExpressionPattern | LRExpressionPattern | IndividualExpressionPattern = (
                    speed_expressions
                )
            case False, True:
                self._speed_expressions = None
                match speeds:
                    case (int(full_speed),):
                        self._speeds: np.array = np.full((4,), full_speed)

                    case (int(left_speed), int(right_speed)):
                        self._speeds = np.array([left_speed, left_speed, right_speed, right_speed])
                    case speeds if len(speeds) == 4:
                        self._speeds = np.array(speeds)
                    case _:
                        types = tuple(type(item) for item in speeds)
                        raise ValueError(
                            f"Invalid Speeds. Must be one of [(int,),(int,int),(int,int,int,int)], got {types}"
                        )
            case True, True:
                raise ValueError(
                    f"Cannot provide both speeds and speed_expressions, got {speeds} and {speed_expressions}"
                )

            case False, False:
                raise ValueError(
                    f"Must provide either speeds or speed_expressions, got {speeds} and {speed_expressions}"
                )

    @classmethod
    def halt(cls) -> Self:
        """
        Create a new instance of the class with a speed of 0, effectively halting the movement.

        Returns:
            Self: A new instance of the class with a speed of 0.
        """
        return cls(0)

    @classmethod
    def straight(cls, speed: int) -> Self:
        """
        Create a new instance of the class with the specified speed.
        Lets the bot drive straight with the specified speed.

        Args:
            speed (int): The speed value to be used for the new instance. Positive for forward and negative for backward.

        Returns:
            Self: A new instance of the class with the specified speed.
        """
        return cls(speed)

    @classmethod
    def differential(cls, direction: Literal["l", "r"], radius: float, outer_speed: int) -> Self:
        """
        Create a new instance of the class with the specified differential movement.
        Let the bot make a differential movement with the specified radius and speed.

        Note:
            The outer speed is the speed of the outer wheel.
            The unit of the radius is a dimensionless number, not CM, not MM, etc.
            The inner speed is calculated from the outer speed and the radius and the track_width.
        Args:
            direction (Literal["l", "r"]): The direction of the movement. Must be one of "l" or "r".
            radius (float): The radius of the movement.
            outer_speed (int): The speed of the outer wheel.

        Returns:
            Self: A new instance of the class with the specified differential movement.

        Raises:
            ValueError: If the direction is not one of "l" or "r".

        """
        inner_speed = int(radius / (radius + cls.Config.track_width) * outer_speed)

        match direction:
            case "l":
                return cls(inner_speed, outer_speed)
            case "r":
                return cls(outer_speed, inner_speed)
            case _:
                raise ValueError("Invalid Direction. Must be one of ['l','r']")

    @classmethod
    def turn(cls, direction: Literal["l", "r"], speed: int) -> Self:
        """
        Create a new instance of the class with the specified turn direction and speed.
        Lets the bot make a turn with the specified direction and speed in place.

        Args:
            direction (Literal["l", "r"]): The direction of the turn. Must be one of "l" or "r".
            speed (int): The speed of the turn.

        Returns:
            Self: A new instance of the class with the specified turn direction and speed.

        Raises:
            ValueError: If the direction is not one of "l" or "r".
        """
        match direction:
            case "l":
                return cls(-speed, speed)
            case "r":
                return cls(speed, -speed)
            case _:
                raise ValueError("Invalid Direction. Must be one of ['l','r']")

    @classmethod
    def drift(cls, fixed_axis: Literal["fl", "rl", "rr", "fr"], speed: int) -> Self:
        """
        Create a new instance of the class with the specified drift direction and speed.
        Lets the bot make a drift with the specified direction and speed in place.

        Note:
            This movement is achieved by making a wheel fixed, while the others move with the specified speed.

            The drift movement is affected by the Config.diagonal_multiplier.


        Args:
            fixed_axis (Literal["fl", "rl", "rr", "fr"]): The direction of the drift. Must be one of "fl", "rl", "rr", or "fr".
            speed (int): The speed of the drift.

        Returns:
            Self: A new instance of the class with the specified drift direction and speed.

        Raises:
            ValueError: If the axis is not one of "fl", "rl", "rr", or "fr".
        """
        match fixed_axis:
            case "fl":
                return cls(0, speed, int(speed * cls.Config.diagonal_multiplier), speed)
            case "rl":
                return cls(speed, 0, speed, int(speed * cls.Config.diagonal_multiplier))
            case "rr":
                return cls(speed * cls.Config.diagonal_multiplier, speed, 0, speed)
            case "fr":
                return cls(speed, speed * cls.Config.diagonal_multiplier, speed, 0)
            case _:
                raise ValueError("Invalid Direction. Must be one of ['fl','rl','rr','fr']")

    def apply(self, multiplier: float) -> Self:
        """
        Apply a multiplier to the speeds of the object and return the modified object.

        Args:
            multiplier (float): The multiplier to apply to the speeds.

        Returns:
            Self: The modified object with the updated speeds.
        """
        self._speeds *= multiplier
        return self

    def unwrap(self) -> np.array:
        """
        Return the speeds of the MovingState object.
        """
        return self._speeds

    def clone(self) -> Self:
        """
        Creates a clone of the current `MovingState` object.

        Returns:
            Self: A new `MovingState` object with the same speeds as the current object.
        """
        if self._speeds:

            return MovingState(*tuple(self._speeds))
        else:
            return MovingState(speed_expressions=self._speed_expressions)

    def __hash__(self) -> int:

        return self._identifier

    def __eq__(self, other: Self) -> bool:
        return tuple(self._speeds) == tuple(other._speeds) and self._speed_expressions == other._speed_expressions

    def __str__(self):
        return f"{self._identifier}-MovingState({self._speeds or self._speed_expressions})"


class MovingTransform:
    """
    A class that represents a moving transform.
    A moving transform is a transition between two states in a state machine.
    Features multiple branches and a breaker function to determine if the transition should be broken.
    """

    def __init__(
        self,
        duration: float,
        breaker: Optional[Callable[[], KT] | Callable[[], bool] | Callable[[], Any]] = None,
        from_states: Optional[Iterable[MovingState] | MovingState] = None,
        to_states: Optional[Dict[KT, MovingState] | MovingState] = None,
    ):
        """
        Initializes a new instance of the class.

        Args:
            duration (float): The duration of the moving transform.
            breaker (Optional[Callable[[], KT] | Callable[[], bool] | Callable[[], Any]]): The breaker function to determine if the transition should be broken. Defaults to None.
            from_states (Optional[Iterable[MovingState]|MovingState]): The states the moving transform is transitioning from. Defaults to None.
            to_states (Optional[Dict[KT, MovingState]|MovingState]): The states the moving transform is transitioning to. Defaults to None.

        Raises:
            ValueError: If duration is non-positive or if breaker has an empty annotated return type.

        Returns:
            None
        """
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")
        if breaker is not None and signature(breaker).return_annotation == Signature.empty:
            raise ValueError(f"Breaker {breaker} must have annotated return type!")

        self.duration: float = duration
        self.breaker: Optional[Callable[[], Any]] = breaker
        match from_states:
            case None:
                self.from_states: List[MovingState] = []
            case state if isinstance(state, MovingState):
                self.from_states: List[MovingState] = [from_states]
            case state if isinstance(state, Iterable):
                self.from_states: List[MovingState] = list(from_states)
            case _:
                raise ValueError(f"Invalid from_states, got {from_states}")

        match to_states:
            case None:
                self.to_states: Dict[KT, MovingState] = {}
            case state if isinstance(state, MovingState):
                self.to_states: Dict[KT, MovingState] = {__PLACE_HOLDER__: state}
            case state if isinstance(state, Iterable):
                self.to_states: Dict[KT, MovingState] = {__PLACE_HOLDER__: to_states}
            case _:
                raise ValueError(f"Invalid to_states, got {to_states}")

    def add_from_state(self, state: MovingState) -> Self:
        """
        Adds a `MovingState` object to the `from_state` list.

        Args:
            state (MovingState): The `MovingState` object to be added.

        Returns:
            Self: The current instance of the class.
        """
        self.from_states.append(state)
        return self

    def add_to_state(self, key: KT, state: MovingState) -> Self:
        """
        Adds a state to the `to_states` dictionary with the given key and state.

        Args:
            key (KT): The key to associate with the state.
            state (MovingState): The state to add to the dictionary.

        Returns:
            Self: The current instance of the class.
        """
        self.to_states[key] = state
        return self


TokenPool: TypeAlias = List[MovingTransform]


class Botix:

    def __init__(self, controller: CloseLoopController, token_pool: Optional[List[MovingTransform]] = None):
        self.controller: CloseLoopController = controller
        self.token_pool: TokenPool = token_pool or []

    @staticmethod
    def ensure_unique_start(token_pool: TokenPool) -> MovingState:
        """
        Calculates the indegree of each state in the token pool to ensure unique starting states.

        Args:
            token_pool (TokenPool): A list of MovingTransform objects representing the token pool.

        Raises:
            ValueError: If there is not exactly one state with a zero indegree.

        Returns:
            None
        """
        states_indegree: Dict[MovingState, int] = {}

        for token in token_pool:
            for state in token.to_states.values():
                states_indegree[state] = states_indegree.get(state, 0) + 1

        zero_indegree_states = [state for state, indegree in states_indegree.items() if indegree == 0]
        if zero_indegree_states == 1:
            return zero_indegree_states[0]
        else:
            raise ValueError(f"There must be exactly one state with a zero indegree, got {zero_indegree_states}")

    @staticmethod
    def ensure_end_state(token_pool: TokenPool) -> Set[MovingState]:
        """
        Calculates the end states of the given token pool.

        Args:
            token_pool (TokenPool): A list of MovingTransform objects representing the token pool.

        Returns:
            Set[MovingState]: A set of MovingState objects representing the end states of the token pool.
        """
        states_outdegree: Dict[MovingState, int] = {}
        for token in token_pool:
            for from_state in token.from_states:
                states_outdegree[from_state] = states_outdegree.get(from_state, 0) + 1
        end_states = {state for state, outdegree in states_outdegree.items() if outdegree == 0}

        return end_states

    @staticmethod
    def ensure_accessibility(token_pool: TokenPool, start_state: MovingState, end_states: Set[MovingState]) -> None:
        """
        Ensures that all states in the given token pool are accessible from the start state.

        Args:
            token_pool (TokenPool): A list of MovingTransform objects representing the token pool.
            start_state (MovingState): The starting state from which to check accessibility.
            end_states (Set[MovingState]): A set of MovingState objects representing the end states of the token pool.

        Raises:
            ValueError: If there are states that are not accessible from the start state.

        Returns:
            None
        """
        search_queue: List[MovingState] = [start_state]
        not_accessible_states: Set[MovingState] = end_states
        while search_queue:
            current_state: MovingState = search_queue.pop(0)
            connected_states: List[MovingState] = []
            for token in token_pool:
                if current_state in token.from_states:
                    connected_states.extend(token.to_states.values())
            not_accessible_states -= set(connected_states)
            if not_accessible_states == set():
                break
        if not_accessible_states:
            raise ValueError(f"States {not_accessible_states} are not accessible from {start_state}")

    def _check_met_requirements(self):
        start_state = self.ensure_unique_start(self.token_pool)
        end_states = self.ensure_end_state(self.token_pool)
        self.ensure_accessibility(self.token_pool, start_state, end_states)

    def compile(self) -> Callable:
        """

        Returns:

        """

        # TODO: static pattern
        # TODO: dynamic pattern
        def _func():

            pass

        return _func


if __name__ == "__main__":

    # Good Cases
    full = MovingState(2800)
    lr = MovingState(4000, -4000)
    individual = MovingState(0, 5000, 5000, 5000)

    print(f"full: {full.unwrap()}\nlr: {lr.unwrap()}\nindividual: {individual.unwrap()}")

    stra = MovingState.straight(3000)
    dif = MovingState.differential("l", 60, 5000)
    tur = MovingState.turn("l", 5000)
    dri = MovingState.drift("fl", 5000)

    print(f"stra: {stra.unwrap()}\ndif: {dif.unwrap()}\ntur: {tur.unwrap()}\ndri: {dri.unwrap()}")

    # Bad Cases
    # g=MovingState(10,10,10,10,10)
    # g = MovingState("1")
