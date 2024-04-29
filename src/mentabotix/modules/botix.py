from collections import Counter
from dataclasses import dataclass
from inspect import signature, Signature
from itertools import zip_longest
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
    Sequence,
)

import numpy as np
from bdmc import CloseLoopController
from terminaltables import SingleTable

from .exceptions import StructuralError, TokenizeError
from ..tools.generators import NameGenerator

Expression: TypeAlias = str | int

FullPattern: TypeAlias = Tuple[int]
LRPattern: TypeAlias = Tuple[int, int]
IndividualPattern: TypeAlias = Tuple[int, int, int, int]
FullExpressionPattern: TypeAlias = str
LRExpressionPattern: TypeAlias = Tuple[Expression, Expression]
IndividualExpressionPattern: TypeAlias = Tuple[Expression, Expression, Expression, Expression]
KT = TypeVar("KT", bound=Hashable)
Context: TypeAlias = Dict[str, Any]

__PLACE_HOLDER__ = "Hello World"
__CONTROLLER_NAME__ = "con"


class MovingState:
    """
    Describes the movement state of the bot.
    Include:
    - halt: make a stop state,all wheels stop moving
    - straight: make a straight moving state,all wheels move in the same direction,same speed
    - turn: make a turning state,left and right wheels turn in different direction,same speed
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

    @property
    def before_entering(self) -> Optional[List[Callable[[], None]]]:
        """
        Returns the list of functions to be called before entering the state.

        :return: An optional list of callables that take no arguments and return None.
        :rtype: Optional[List[Callable[[], None]]]
        """
        return self._before_entering

    @property
    def after_exiting(self) -> Optional[List[Callable[[], None]]]:
        """
        Returns the list of functions to be called after exiting the state.

        :return: An optional list of callables that take no arguments and return None.
        :rtype: Optional[List[Callable[[], None]]]
        """
        return self._after_exiting

    @property
    def used_context_variables(self) -> Set[str]:
        """
        Returns the set of context variable names used in the speed expressions.

        :return: An optional set of strings representing the context variable names.
        :rtype: Optional[Set[str]]
        """
        return self._used_context_variables

    @property
    def speed_expressions(self) -> IndividualExpressionPattern:
        """
        Get the speed expressions of the object.

        :return: The speed expressions of the object.
        :rtype: IndividualExpressionPattern
        """
        return self._speed_expressions

    def __init__(
        self,
        *speeds: Unpack[FullPattern] | Unpack[LRPattern] | Unpack[IndividualPattern],
        speed_expressions: Optional[FullExpressionPattern | LRExpressionPattern | IndividualExpressionPattern] = None,
        used_context_variables: Optional[Set[str]] = None,
        before_entering: Optional[List[Callable[[], None]]] = None,
        after_exiting: Optional[List[Callable[[], None]]] = None,
    ) -> None:
        """
        Initialize the MovingState with speeds.

        Args:
            *speeds: A tuple representing the speed pattern.
                It should be one of the following types:
                    - FullPattern: A single integer representing full speed for all directions.
                    - LRPattern: A tuple of two integers representing left and right speeds.
                    - IndividualPattern: A tuple of four integers representing individual speeds for each direction.

        Keyword Args:
            speed_expressions (Optional[FullExpressionPattern | Unpack[LRExpressionPattern] | Unpack[IndividualExpressionPattern]]): The speed expressions of the wheels.
            before_entering (Optional[List[Callable[[], None]]]): The list of functions to be called before entering the state.
            after_exiting (Optional[List[Callable[[], None]]]): The list of functions to be called after exiting the state.
        Raises:
            ValueError: If the provided speeds do not match any of the above patterns.
        """
        self._speed_expressions: IndividualExpressionPattern
        self._speeds: np.array

        match bool(speed_expressions), bool(speeds):
            case True, False:
                if used_context_variables is None:
                    raise ValueError(
                        "No used_context_variables provided, You must provide a names set that contains all the name of the variables used in the speed_expressions."
                        "If you do not need use context variables, then you should use *speeds argument to create the MovingState."
                    )
                self._speeds = None
                match speed_expressions:
                    case str(full_expression):
                        self._speed_expressions = (full_expression, full_expression, full_expression, full_expression)
                    case (left_expression, right_expression):
                        if all(isinstance(item, int) for item in speed_expressions):
                            raise ValueError(
                                f"All expressions are integers. You should use *speeds argument to create the MovingState, got {speed_expressions}"
                            )

                        self._speed_expressions = (left_expression, left_expression, right_expression, right_expression)
                    case speed_expressions if len(speed_expressions) == 4:
                        if all(isinstance(item, int) for item in speed_expressions):
                            raise ValueError(
                                f"All expressions are integers. You should use *speeds argument to create the MovingState, got {speed_expressions}"
                            )
                        self._speed_expressions = speed_expressions
                    case _:
                        types = tuple(type(item) for item in speed_expressions)
                        raise ValueError(
                            f"Invalid Speed Expressions. Must be one of [{FullExpressionPattern},{LRExpressionPattern},{IndividualExpressionPattern}], got {types}"
                        )

            case False, True:
                self._speed_expressions = None
                match speeds:
                    case (int(full_speed),):
                        self._speeds = np.full((4,), full_speed)

                    case (int(left_speed), int(right_speed)):
                        self._speeds = np.array([left_speed, left_speed, right_speed, right_speed])
                    case speeds if len(speeds) == 4 and all(isinstance(item, int) for item in speeds):
                        self._speeds = np.array(speeds)
                    case _:
                        types = tuple(type(item) for item in speeds)
                        raise ValueError(
                            f"Invalid Speeds. Must be one of [{FullPattern},{LRPattern},{IndividualPattern}], got {types}"
                        )
            case True, True:
                raise ValueError(
                    f"Cannot provide both speeds and speed_expressions, got {speeds} and {speed_expressions}"
                )

            case False, False:
                raise ValueError(
                    f"Must provide either speeds or speed_expressions, got {speeds} and {speed_expressions}"
                )

        if used_context_variables:
            self._validate_used_context_variables_presence(used_context_variables, self._speed_expressions)
        self._before_entering: List[Callable[[], None]] = before_entering
        self._used_context_variables: Set[str] = used_context_variables
        self._after_exiting: List[Callable[[], None]] = after_exiting
        self._identifier: int = MovingState.__state_id_counter__
        MovingState.__state_id_counter__ += 1

    @staticmethod
    def _validate_used_context_variables_presence(
        used_context_variables: Set[str], speed_expressions: IndividualExpressionPattern
    ) -> None:
        for variable in used_context_variables:
            if any(variable in expression for expression in speed_expressions):
                continue
            raise ValueError(f"Variable {variable} not found in {speed_expressions}.")

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
                return cls(int(speed * cls.Config.diagonal_multiplier), speed, 0, speed)
            case "fr":
                return cls(speed, int(speed * cls.Config.diagonal_multiplier), speed, 0)
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
        self._speeds = (self._speeds * multiplier).astype(np.int32)
        return self

    def unwrap(self) -> Tuple[int, ...]:
        """
        Return the speeds of the MovingState object.
        """
        return tuple(self._speeds)

    def clone(self) -> Self:
        """
        Creates a clone of the current `MovingState` object.

        Returns:
            Self: A new `MovingState` object with the same speeds as the current object.
        """

        return MovingState(
            *tuple(self._speeds.tolist()),
            speed_expressions=self._speed_expressions,
            used_context_variables=self._used_context_variables,
            before_entering=self._before_entering,
            after_exiting=self._after_exiting,
        )

    def tokenize(self, con: Optional[CloseLoopController]) -> Tuple[List[str], Context]:
        """
        Converts the current state into a list of tokens and a context dictionary.

        Parameters:
        - con: Optional[CloseLoopController] - The closed-loop controller required if speed expressions exist.

        Returns:
        - Tuple[List[str], Context]: A tuple containing the list of tokens and the context dictionary.

        Raises:
        - TokenizeError: If the state contains both speeds and speed expressions, or neither.
        - RuntimeError: If an internal logic error occurs; this state should理论上 never be reached.
        """

        # Check for simultaneous presence or absence of speeds and speed expressions

        if self._speeds is not None and self._speed_expressions:
            raise TokenizeError(
                f"Cannot tokenize a state with both speed expressions and speeds, got {self._speeds} and {self._speed_expressions}."
            )
        elif self._speeds is None and self._speed_expressions is None:
            raise TokenizeError(f"Cannot tokenize a state with no speed expressions and no speeds.")

        context: Context = {}  # Initialize the context dictionary
        context_updater_func_name_generator: NameGenerator = NameGenerator(f"state{self._identifier}_context_updater_")

        # Generate tokens for actions before entering the state
        before_enter_tokens: List[str] = []
        if self._before_entering:
            for func in self._before_entering:
                context[(func_var_name := context_updater_func_name_generator())] = func
                before_enter_tokens.append(f".wait_exec({func_var_name})")

        # Generate tokens for actions after exiting the state
        after_exiting_tokens: List[str] = []
        if self._after_exiting:
            for func in self._after_exiting:
                context[(func_var_name := context_updater_func_name_generator())] = func
                after_exiting_tokens.append(f".wait_exec({func_var_name})")

        state_tokens: List[str] = []
        # Generate tokens based on speed expressions or speeds
        match self._speed_expressions, self._speeds:
            case expression, None:
                if con is None:
                    raise TokenizeError(
                        f"You must parse a CloseLoopController to tokenize a state with expression pattern"
                    )

                getter_function_name_generator = NameGenerator(f"state{self._identifier}_context_getter_")
                getter_temp_name_generator = NameGenerator(f"state{self._identifier}_context_getter_temp_")
                input_arg_string = str(tuple(expression)).replace("'", "")
                for varname in self._used_context_variables:
                    # Create context retrieval functions using expressions
                    fn: Callable[[], Any] = con.register_context_getter(varname)
                    context[(getter_func_var_name := getter_function_name_generator())] = fn
                    temp_name = getter_temp_name_generator()
                    if input_arg_string.count(varname) == 1:
                        input_arg_string = input_arg_string.replace(varname, f"{getter_func_var_name}()", 1)
                    else:
                        input_arg_string = input_arg_string.replace(
                            varname, f"({temp_name}:={getter_func_var_name}())", 1
                        )
                        input_arg_string = input_arg_string.replace(varname, temp_name)
                state_tokens.append(f".set_motors_speed({input_arg_string})")

            case None, speeds:
                state_tokens.append(f".set_motors_speed({tuple(speeds)})")
            case _:
                raise RuntimeError("should never reach here")

        tokens: List[str] = before_enter_tokens + state_tokens + after_exiting_tokens
        return tokens, context

    def __hash__(self) -> int:
        return self._identifier

    def __eq__(self, other: Self) -> bool:
        return tuple(self._speeds) == tuple(other._speeds) and self._speed_expressions == other._speed_expressions

    def __str__(self):
        return f"{self._identifier}-MovingState{self._speed_expressions or (tuple(self._speeds) if self._speeds is not None else None)}"

    def __repr__(self):
        return str(self)


class MovingTransition:
    """
    A class that represents a moving transition.
    A moving transform is a transition between two states in a state machine.
    Features multiple branches and a breaker function to determine if the transition should be broken.
    """

    __state_id_counter__: ClassVar[int] = 0

    @property
    def identifier(self) -> int:
        return self._transition_id

    def __init__(
        self,
        duration: float,
        breaker: Optional[Callable[[], KT] | Callable[[], bool] | Callable[[], Any]] = None,
        check_interval: Optional[float] = 0.01,
        from_states: Optional[Sequence[MovingState] | MovingState] = None,
        to_states: Optional[Dict[KT, MovingState] | MovingState] = None,
    ):
        """
        Initialize a new instance of the MovingTransition class.

        Args:
            duration (float): The duration of the transition in seconds. Must be positive.
            breaker (Optional[Callable[[], KT] | Callable[[], bool] | Callable[[], Any]]): A function that determines
                whether the transition should be broken. If None, no breaker function is used. Must have an annotated
                return type.
            check_interval (Optional[float]): The interval in seconds at which the breaker function should be checked.
                Defaults to 0.01 seconds.
            from_states (Optional[Iterable[MovingState] | MovingState]): The states that the transition originates from.
                Can be a single state or an iterable of states. Defaults to None.
            to_states (Optional[Dict[KT, MovingState] | MovingState]): The states that the transition leads to. Can be a
                single state or a dictionary mapping keys to states. Defaults to None.

        Raises:
            ValueError: If duration is not positive or if breaker has an empty return type annotation.
            ValueError: If from_states is not None, a MovingState object, or an iterable of MovingState objects.
            ValueError: If to_states is not None, a MovingState object, a dictionary mapping keys to MovingState objects,
                or None.

        Returns:
            None
        """
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")
        if breaker is not None and signature(breaker).return_annotation == Signature.empty:
            raise ValueError(f"Breaker {breaker} must have annotated return type!")

        self.duration: float = duration
        self.breaker: Optional[Callable[[], Any]] = breaker
        self.check_interval: float = check_interval
        match from_states:
            case None:
                self.from_states: List[MovingState] = []
            case state if isinstance(state, MovingState):
                self.from_states: List[MovingState] = [from_states]
            case state if isinstance(state, Sequence) and all(isinstance(s, MovingState) for s in state):
                self.from_states: List[MovingState] = list(from_states)
            case _:
                raise ValueError(f"Invalid from_states, got {from_states}")

        match to_states:
            case None:
                self.to_states: Dict[KT, MovingState] = {}
            case state if isinstance(state, MovingState):
                self.to_states: Dict[KT, MovingState] = {__PLACE_HOLDER__: state}
            case state if isinstance(state, Dict):
                self.to_states: Dict[KT, MovingState] = to_states
            case _:
                raise ValueError(f"Invalid to_states, got {to_states}")

        self._transition_id: int = MovingTransition.__state_id_counter__
        MovingTransition.__state_id_counter__ += 1

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

    def tokenize(self) -> Tuple[List[str], Context]:
        """
        Tokenizes the current object and returns a tuple of tokens and context.

        Returns:
            Tuple[List[str], Context]: A tuple containing a list of tokens and a context dictionary.
        """
        tokens: List[str] = []
        context: Context = {}
        name_generator: NameGenerator = NameGenerator(f"transition{self._transition_id}_breaker_")
        context[(breaker_name := name_generator())] = self.breaker
        match len(self.to_states):
            case 0:
                raise TokenizeError(f"Transition must have at least one to_state, got {self.to_states}.")
            case 1 if not callable(self.breaker):
                tokens.append(f".delay({self.duration})")
            case 1 if callable(self.breaker):
                tokens.append(f".delay_b({self.duration},{breaker_name},{self.check_interval})")
            case length if length > 1 and callable(self.breaker):
                tokens.append(f".delay_b_match({self.duration},{breaker_name},{self.check_interval})")
        return tokens, context

    def clone(self) -> Self:
        """
        Clones the current `MovingTransition` object and returns a new instance with the same values.

        Returns:
            Self: A new `MovingTransition` object with the same values as the current object.
        """
        return MovingTransition(
            self.duration,
            self.breaker,
            self.check_interval,
            self.from_states,
            self.to_states,
        )

    def __str__(self):
        temp = [["From", "To"]]
        for from_state, to_state in zip_longest(self.from_states, self.to_states.values()):
            temp.append([str(from_state) if from_state else "", str(to_state) if to_state else ""])
        return SingleTable(temp).table

    def __repr__(self):
        return f"{self.from_states} => {list(self.to_states.values())}"

    def __hash__(self):
        return self._transition_id


TokenPool: TypeAlias = List[MovingTransition]


class Botix:

    def __init__(self, controller: CloseLoopController, token_pool: Optional[List[MovingTransition]] = None):
        self.controller: CloseLoopController = controller
        self.token_pool: TokenPool = token_pool or []

    @staticmethod
    def acquire_unique_start(token_pool: TokenPool, none_check: bool = True) -> MovingState | None:
        """
        Retrieves a unique starting state from the given token pool.

        Parameters:
        - token_pool: An instance of TokenPool, representing a collection of tokens.
        - none_check: A boolean, defaulting to True. If True, raises a ValueError when no unique starting state is found; otherwise, returns None.

        Returns:
        - Either a MovingState or None. Returns the starting state (with indegree 0) if uniquely identified; based on none_check's value, either returns None or raises an exception.
        """

        # Initialize a dictionary to hold the indegree of each state
        states_indegree: Dict[MovingState, int] = {}

        # Calculates the indegree for each state by examining token transitions
        for token in token_pool:
            for state in token.from_states:
                states_indegree[state] = states_indegree.get(state, 0)
            for state in token.to_states.values():
                # Increments the indegree for each state that a token can reach
                states_indegree[state] = states_indegree.get(state, 0) + 1

        # Identifies states with an indegree of zero
        zero_indegree_states = [state for state, indegree in states_indegree.items() if indegree == 0]

        # Validates that there is exactly one state with an indegree of zero
        if len(zero_indegree_states) == 1:
            return zero_indegree_states[0]
        else:
            if none_check:
                # Raises an error if none_check is enabled and no unique starting state is present
                raise ValueError(f"There must be exactly one state with a zero indegree, got {states_indegree}")
            return None

    @staticmethod
    def acquire_end_state(token_pool: TokenPool) -> Set[MovingState]:
        """
        Calculates the end states of the given token pool.

        This function iterates through each token in the provided token pool,
        counting the number of outgoing transitions from each state.
        States with no outgoing transitions are considered end states.

        Args:
            token_pool (TokenPool): A list of MovingTransform objects representing the token pool.

        Returns:
            Set[MovingState]: A set of MovingState objects representing the end states of the token pool.
        """
        # Initialize a dictionary to keep track of the outdegree (number of outgoing transitions)
        # for each state.
        states_outdegree: Dict[MovingState, int] = {}

        # Count the number of outgoing transitions from each state.
        for token in token_pool:
            for from_state in token.from_states:
                states_outdegree[from_state] = states_outdegree.get(from_state, 0) + 1

        # Identify end states by finding states with an outdegree of 0.
        end_states = {state for state, outdegree in states_outdegree.items() if outdegree == 0}

        return end_states

    @staticmethod
    def ensure_accessibility(token_pool: TokenPool, start_state: MovingState, end_states: Set[MovingState]) -> None:
        """
        Ensures that all states in the given token pool are accessible from the start state.

        This method performs a breadth-first search to check if all specified end states can be reached
        from the given start state by traversing through the token pool. It raises a ValueError if any
        end state is not accessible.

        Args:
            token_pool (TokenPool): A list of MovingTransform objects representing the token pool.
            start_state (MovingState): The starting state from which to check accessibility.
            end_states (Set[MovingState]): A set of MovingState objects representing the end states of the token pool.

        Raises:
            ValueError: If there are states that are not accessible from the start state.

        Returns:
            None

        Note:
            The structure validity of the token pool does not affect the accessibility check,
            So, if the token pool is not structurally valid, this method will still complete the check.
            FIXME: This should be fixed in the future.
        """
        # Initialize a queue for BFS and a set to keep track of states not accessible from the start state
        search_queue: List[MovingState] = [start_state]
        not_accessible_states: Set[MovingState] = end_states
        visited_states: Set[MovingState] = set()
        # Perform breadth-first search to find if all end states are accessible
        while search_queue:
            # Pop the next state from the queue to explore
            current_state: MovingState = search_queue.pop(0)
            connected_states: Set[MovingState] = set()

            # Explore all tokens that can be reached from the current state
            for token in token_pool:
                if current_state in token.from_states:
                    # Add all to_states of the current token to the connected states list
                    connected_states.update(token.to_states.values())
            search_queue.extend((newly_visited := connected_states - visited_states))
            # Update the set of not accessible states by removing the states we just found to be connected
            not_accessible_states -= newly_visited
            visited_states.update(connected_states)
            # If there are no more not accessible states, we are done
            if len(not_accessible_states) == 0:
                return

        # If there are still states marked as not accessible, raise an exception
        if not_accessible_states:
            raise ValueError(f"States {not_accessible_states} are not accessible from {start_state}")

    @staticmethod
    def ensure_structure_validity(pool: TokenPool) -> None:
        """
        Ensures the structural validity of a given token pool by verifying that each state connects to exactly one transition as its input.
        Branching logic is implemented within the MovingTransition class.

        Parameters:
            pool (TokenPool): A list of MovingTransform objects representing the token pool.

        Raises:
            StructuralError: If any state is found connecting to multiple transitions as inputs.
                The error message includes details on the conflicting transitions for each affected state.

        Returns:
            None: If all states connect to a single input transition.
        """
        # Collect all states from the pool
        states: List[MovingState] = []

        for trans in pool:
            states.extend(trans.from_states)
        # Count occurrences of each state
        element_counts = Counter(states)

        # Identify states with multiple occurrences (potential structural issues)
        frequent_elements: List[MovingState] = [element for element, count in element_counts.items() if count >= 2]
        if not frequent_elements:
            return None
        # Construct error message
        std_err = (
            "Each state must connect to EXACTLY one transition as its input, as branching is implemented INSIDE the MovingTransition class.\n"
            "Below are error details:\n"
        )
        for state in frequent_elements:
            # Find transitions conflicting with the current state
            conflict_trans: List[MovingTransition] = list(
                filter(lambda transition: state in transition.from_states, pool)
            )
            std_err += f"For {state}, there are {len(conflict_trans)} conflicting transitions:\n\t" + "\n\t".join(
                [f"{trans}" for trans in conflict_trans]
            )
        # Raise StructuralError with the prepared message
        raise StructuralError(std_err)

    def acquire_loops(self) -> List[List[MovingState]]:
        """
        Retrieves a list of all looping paths, where each path is composed of a series of MovingState instances.

        Returns:
            List[List[MovingState]]: A list containing all looping paths, with each path being a list of MovingState objects.
        """

        # Initialize the starting state, search stack, and list for storing loops.
        start_state: MovingState = self.acquire_unique_start(self.token_pool)
        search_stack: List[MovingState] = [start_state]
        loops: List[List[MovingState]] = []

        # Initialize the chain (current path), dictionary to track branching depths, and a rollback flag.
        chain: List[MovingState] = []
        branch_dict: Dict[MovingState, int] = {}
        rolling_back: bool = False

        # Iterate until the search stack is empty.
        while search_stack:
            # Pop the current state from the stack and find its connected transitions.
            cur_state: MovingState = search_stack.pop()
            chain.append(cur_state)
            connected_transition: MovingTransition = self.acquire_connected_forward_transition(
                cur_state, none_check=False
            )
            match rolling_back, bool(connected_transition):
                case True, False:
                    # once the rollback is activated, there must have a forward transition connected to cur_state,
                    # since we are come from there
                    raise ValueError("Should not arrive here!")
                case True, True:

                    connected_states = list(connected_transition.to_states.values())

                    cur_depth_index: int = branch_dict.get(cur_state)
                    branch_dict[cur_state] = cur_depth_index + 1
                    if cur_depth_index == len(connected_states):
                        chain.pop()
                        search_stack.append(chain[-1])
                    else:
                        next_state = connected_states[cur_depth_index]
                        search_stack.append(next_state)
                        rolling_back = False
                case False, False:
                    rolling_back = True
                    chain.pop()
                    search_stack.append(chain[-1])
                    continue
                case False, True:
                    connected_states = list(connected_transition.to_states.values())

                    cur_depth_index: int = branch_dict.get(cur_state, 0)
                    branch_dict[cur_state] = cur_depth_index + 1
                    if cur_depth_index == len(connected_states):
                        rolling_back = True
                        chain.pop()
                        search_stack.append(chain[-1])
                    else:
                        next_state = connected_states[cur_depth_index]
                        state_hash = hash(next_state)
                        if any(state_hash == hash(state) for state in chain):
                            # Upon detecting a loop, append the loop path to the loops list.
                            loops.append(chain[chain.index(next_state) :])
                            rolling_back = True
                            continue
                        else:
                            # Add the current state to the chain and push the next state onto the search stack.

                            search_stack.append(next_state)

        return loops

    def is_branchless_chain(self, start_state: MovingState, end_state: MovingState) -> bool:
        """
        Check if the given start state and end state form a branchless chain in the token pool.

        This function ensures the structure validity of the token pool and checks the accessibility of the start state and end state.
        It then performs a breadth-first search to check if there is a path from the start state to the end state in the token pool.
        A branchless chain is defined as a path where each state has only one outgoing transition.

        Parameters:
            start_state (MovingState): The starting state of the chain.
            end_state (MovingState): The end state of the chain.

        Returns:
            bool: True if the start state and end state form a branchless chain, False otherwise.
        """
        self.ensure_structure_validity(self.token_pool)
        self.ensure_accessibility(self.token_pool, start_state, {end_state})

        search_queue = [start_state]
        while search_queue:
            cur_state = search_queue.pop(0)
            connected_transitions: MovingTransition = self.acquire_connected_forward_transition(cur_state)

            match len(connected_transitions.to_states):
                case 1:
                    search_queue.append(list(connected_transitions.to_states.values())[0])
                case _:
                    return False
            if search_queue[-1] == end_state:
                return True

    def acquire_connected_forward_transition(
        self, state: MovingState, none_check: bool = True
    ) -> MovingTransition | None:
        """
        Returns the MovingTransition object that is connected to the given MovingState object in the token pool.

        This function takes in a MovingState object and checks if it is connected to any forward transition in the token pool. It filters the token pool based on the from_states attribute of each transition and returns the first matching MovingTransition object. If none_check is True and no matching transition is found, a ValueError is raised. If none_check is False and no matching transition is found, None is returned. If multiple matching transitions are found, a ValueError is raised.

        Parameters:
            state (MovingState): The MovingState object to search for connected forward transitions.
            none_check (bool, optional): Whether to raise a ValueError if no matching transition is found. Defaults to True.

        Returns:
            MovingTransition | None: The MovingTransition object that is connected to the given MovingState object, or None if none_check is False and no matching transition is found.

        Raises:
            ValueError: If no matching transition is found and none_check is True, or if multiple matching transitions are found.
        """
        response = list(filter(lambda trans: state in trans.from_states, self.token_pool))
        match len(response):
            case 0:
                if none_check:
                    raise ValueError(f"the state of {state} is not connected to any forward transition!")
                return None
            case 1:
                return response[0]
            case _:
                raise ValueError(f"A state can *ONLY* connect to ONE transition as its input. Found {response}")

    def acquire_connected_backward_transition(
        self, state: MovingState, none_check: bool = True
    ) -> MovingTransition | List[MovingTransition] | None:
        """
        Returns the MovingTransition object or a list of MovingTransition objects that are connected to the given MovingState object in the token pool.

        This function takes in a MovingState object and checks if it is connected to any backward transition in the token pool. It filters the token pool based on the to_states attribute of each transition and returns the matching MovingTransition object(s). If none_check is True and no matching transition is found, a ValueError is raised. If none_check is False and no matching transition is found, None is returned. If multiple matching transitions are found, a list of the matching MovingTransition objects is returned.

        Parameters:
            state (MovingState): The MovingState object to search for connected backward transitions.
            none_check (bool, optional): Whether to raise a ValueError if no matching transition is found. Defaults to True.

        Returns:
            MovingTransition | List[MovingTransition] | None: The MovingTransition object or a list of MovingTransition objects that are connected to the given MovingState object, or None if none_check is False and no matching transition is found.

        Raises:
            ValueError: If no matching transition is found and none_check is True, or if multiple matching transitions are found.
        """
        response = list(filter(lambda trans: state in trans.to_states.values(), self.token_pool))
        match len(response):
            case 0:
                if none_check:
                    raise ValueError(f"the state of {state} is not connected to any backward transition!")
                return None
            case 1:
                return response[0]
            case _:
                return response

    def _check_met_requirements(self):
        self.ensure_structure_validity(self.token_pool)
        start_state = self.acquire_unique_start(self.token_pool)
        end_states = self.acquire_end_state(self.token_pool)
        self.ensure_accessibility(self.token_pool, start_state, end_states)

    def compile(self) -> Callable:
        raise NotImplementedError


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
