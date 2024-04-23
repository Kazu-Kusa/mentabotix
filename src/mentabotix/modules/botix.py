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
)

import numpy as np
from bdmc import CloseLoopController

FullPattern: TypeAlias = Tuple[int]
LRPattern: TypeAlias = Tuple[int, int]
IndividualPattern: TypeAlias = Tuple[int, int, int, int]
KT = TypeVar("KT", bound=Hashable)


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

    def __init__(self, *speeds: Unpack[FullPattern] | Unpack[LRPattern] | Unpack[IndividualPattern]):
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
        match speeds:
            case (int(full_speed),):
                self._speeds: np.array = np.full((4,), full_speed)

            case (int(left_speed), int(right_speed)):
                self._speeds = np.array([left_speed, left_speed, right_speed, right_speed])
            case speeds if len(speeds) == 4:
                self._speeds = np.array(speeds)
            case _:
                types = tuple(type(item) for item in speeds)
                raise ValueError(f"Invalid Speeds. Must be one of [(int,),(int,int),(int,int,int,int)], got {types}")

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
        return MovingState(self._speeds)

    def __hash__(self) -> int:
        return hash(tuple(self._speeds))

    def __eq__(self, other: Self) -> bool:
        return tuple(self._speeds) == tuple(other._speeds)


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
        from_states: Optional[Iterable[MovingState]] = None,
        to_states: Optional[Iterable[MovingState]] = None,
        match_cases: Optional[Dict[KT, MovingState]] = None,
    ):
        """
        Initializes a new instance of the MovingTransform class.

        Args:
            duration (float): The duration of the transition.
            breaker (Optional[Callable[[], KT] | Callable[[], bool] | Callable[[], Any]]): A function that determines if the transition should be broken.
            from_states (Optional[Iterable[MovingState]]): The initial state of the transition.
            to_states (Optional[Iterable[MovingState]]): The final state of the transition.
            match_cases (Optional[Dict[KT, MovingState]]): A dictionary of branch rules for the transition.

        Raises:
            ValueError: If the duration is not positive or if the breaker function does not have an annotated return type.

        Returns:
            None
        """
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")
        if breaker is not None and signature(breaker).return_annotation == Signature.empty:
            raise ValueError(f"Breaker {breaker} must have annotated return type!")

        self.duration: float = duration
        self.breaker: Optional[Callable[[], Any]] = breaker
        self.to_states: List[MovingState] = list(to_states) if to_states is not None else []
        self.from_states: List[MovingState] = list(from_states) if from_states is not None else []
        self.match_cases: Dict[KT, MovingState] = match_cases or {}

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

    def add_to_state(self, state: MovingState) -> Self:
        """
        Adds a `MovingState` object to the `to_state` list.

        Args:
            state (MovingState): The `MovingState` object to be added.

        Returns:
            Self: The current instance of the class.
        """
        self.to_states.append(state)
        return self

    def add_match_case(self, key: KT, state: MovingState) -> Self:
        """
        Adds a rule to the `rules` dictionary with the given `key` and `state`.

        Args:
            key (KT): The key for the rule.
            state (MovingState): The state for the rule.

        Returns:
            Self: The current instance of the class.
        """
        self.match_cases[key] = state
        return self


TokenPool: TypeAlias = List[MovingState | MovingTransform]


class Botix:

    def __init__(
        self, controller: CloseLoopController, token_pool: Optional[List[MovingState | MovingTransform]] = None
    ):
        self.controller: CloseLoopController = controller
        self.token_pool: TokenPool = token_pool or []

    def resolve_confilict(self, token_pool: TokenPool):
        states = []
        transforms = []
        for token in token_pool:
            if isinstance(token, MovingState):
                states.append(token)
            elif isinstance(token, MovingTransform):
                transforms.append(token)
            else:
                raise ValueError("Token must be either a MovingState or MovingTransform")

    def compile(self) -> Callable:
        """

        Returns:

        """

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
