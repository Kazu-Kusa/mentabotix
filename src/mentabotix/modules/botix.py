import inspect
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from itertools import zip_longest
from queue import Queue
from typing import (
    Tuple,
    TypeAlias,
    Self,
    Unpack,
    Literal,
    Any,
    Callable,
    Hashable,
    TypeVar,
    Dict,
    Optional,
    List,
    ClassVar,
    Set,
    Sequence,
    get_type_hints,
)

import numpy as np
from bdmc import CloseLoopController
from numpy.random import random
from terminaltables import SingleTable

from .exceptions import StructuralError, TokenizeError
from ..tools.generators import NameGenerator

T_EXPR = TypeVar("T_EXPR", str, list)

Expression: TypeAlias = str | int

FullPattern: TypeAlias = Tuple[int]
LRPattern: TypeAlias = Tuple[int, int]
IndividualPattern: TypeAlias = Tuple[int, int, int, int]
FullExpressionPattern: TypeAlias = str
LRExpressionPattern: TypeAlias = Tuple[Expression, Expression]
IndividualExpressionPattern: TypeAlias = Tuple[Expression, Expression, Expression, Expression]
KT = TypeVar("KT", bound=Hashable)
Context: TypeAlias = Dict[str, Any]


__PLACE_HOLDER__ = 0
__CONTROLLER_NAME__ = "con"
__MAGIC_SPLIT_CHAR__ = "$$$"


class PatternType(Enum):
    """
    Three types of control cmd
    """

    Full = 1
    LR = 2
    Individual = 4


def get_function_annotations(func: Callable) -> str:
    """
    Get the function annotations of a given function.

    Args:
        func (callable): The function to get the annotations for.

    Returns:
        str: A string representation of the function annotations.
    """
    try:
        # Get the function's signature
        sig = inspect.signature(func)
    except ValueError as e:
        # Handle cases where func is not a valid function
        raise ValueError(f"Invalid function: {func}") from e

    # Get the type annotations
    type_hints = get_type_hints(func)

    # Initialize strings for parameter types and return type
    param_types = []

    for param_name, param in sig.parameters.items():
        # Check if the parameter has an annotation
        if param.annotation != inspect.Parameter.empty:
            hint = type_hints.get(param_name)
            # Convert the type hint to a string representation,
            # handling complex hints like List[int] properly
            param_type_str = convert_type_str(hint)
            param_types.append(param_type_str)
        else:
            # Use 'Any' for parameters without annotations
            param_types.append("Any")

    # Handle the return type
    return_type = convert_type_str(type_hints.get("return", Any))

    # Concatenate the parameter type strings and return type to form the annotation string
    params_str = ", ".join(param_types)
    return f"{func.__name__}({params_str}) -> {return_type}"


def convert_type_str(hint) -> str:
    """
    Convert a type hint to a string representation, handling complex types and generics.

    Args:
        hint: The type hint to convert.

    Returns:
        str: The string representation of the type hint.
    """
    if hint is None:
        return "None"
    elif hint in {int, float, bool, str, list, tuple, set, dict}:
        return hint.__name__
    else:
        # For other types, convert directly to string
        return str(hint).replace("typing.", "")


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
    def pattern_type(self) -> PatternType:
        """
        Returns the pattern type of the state.

        Returns:
            PatternType: The pattern type of the state.
        """
        return self._pattern_type

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
    def used_context_variables(self) -> List[str]:
        """
        Returns the set of context variable names used in the speed expressions.

        :return: An optional set of strings representing the context variable names.
        :rtype: Optional[List[str]]
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
        used_context_variables: Optional[List[str]] = None,
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
            used_context_variables (Optional[List[str]]): The set of context variable names used in the speed expressions.
            before_entering (Optional[List[Callable[[], None]]]): The list of functions to be called before entering the state.
            after_exiting (Optional[List[Callable[[], None]]]): The list of functions to be called after exiting the state.
        Raises:
            ValueError: If the provided speeds do not match any of the above patterns.
        """
        self._speed_expressions: IndividualExpressionPattern
        self._speeds: np.array
        self._pattern_type: PatternType
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
                        self._pattern_type = PatternType.Full
                        self._speed_expressions = (full_expression, full_expression, full_expression, full_expression)
                    case (left_expression, right_expression):
                        if all(isinstance(item, int) for item in speed_expressions):
                            raise ValueError(
                                f"All expressions are integers. You should use *speeds argument to create the MovingState, got {speed_expressions}"
                            )
                        self._pattern_type = PatternType.LR
                        self._speed_expressions = (left_expression, left_expression, right_expression, right_expression)
                    case speed_expressions if len(speed_expressions) == 4:
                        if all(isinstance(item, int) for item in speed_expressions):
                            raise ValueError(
                                f"All expressions are integers. You should use *speeds argument to create the MovingState, got {speed_expressions}"
                            )
                        self._pattern_type = PatternType.Individual
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
                        self._pattern_type = PatternType.Full
                        self._speeds = np.full((4,), full_speed)
                    case (int(left_speed), int(right_speed)):
                        self._pattern_type = PatternType.LR

                        self._speeds = np.array([left_speed, left_speed, right_speed, right_speed])
                    case speeds if len(speeds) == 4 and all(isinstance(item, int) for item in speeds):
                        self._pattern_type = PatternType.Individual

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
            self._validate_used_context_variables(used_context_variables, self._speed_expressions)
        self._used_context_variables: List[str] = used_context_variables or []
        self._before_entering: List[Callable[[], None]] = before_entering or []
        self._after_exiting: List[Callable[[], None]] = after_exiting or []
        self._identifier: int = MovingState.__state_id_counter__
        MovingState.__state_id_counter__ += 1

    @staticmethod
    def _validate_used_context_variables(
        used_context_variables: List[str], speed_expressions: IndividualExpressionPattern
    ) -> None:
        if len(used_context_variables) != len(set(used_context_variables)):
            raise ValueError(f"used_context_variables can't contain duplicated item!")
        for variable in used_context_variables:
            if any(variable in str(expression) for expression in speed_expressions):
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
    def rand_turn(
        cls, con: CloseLoopController, turn_speed: int, used_ctx_varname: str = "direction", turn_left_prob: float = 0.5
    ) -> Self:
        """
        Adds a method for random turning to the CloseLoopController class.

        Parameters:
            cls: Class method convention parameter, referring to the current class.
            con: CloseLoopController object, representing the instance to which the random turning control is applied.
            turn_speed: Turning speed, positive for turning right, negative for turning left.
            used_ctx_varname: Context variable name used to represent the turn direction, defaults to "direction".
            turn_left_prob: Probability of turning left, defaults to 0.5, meaning equal chance of turning left or right.

        Returns:
            None
        """

        def _dir() -> int:
            """
            Internal function to randomly decide the turn direction.

            Returns:
                int: 1 for turning left, -1 for turning right.
            """
            return 1 if random() < turn_left_prob else -1

        # Register a context updater to update the turn direction before entering this behavior.
        _updater = con.register_context_updater(_dir, output_keys=[used_ctx_varname], input_keys=[])

        # Set speed expressions and actions before entering, implementing random turning.
        return cls(
            speed_expressions=(f"{-turn_speed}*{used_ctx_varname}", f"{turn_speed}*{used_ctx_varname}"),
            used_context_variables=[used_ctx_varname],
            before_entering=[_updater],
        )

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
        if self._speeds is not None:

            return MovingState(
                *tuple(self._speeds.tolist()),
                speed_expressions=self._speed_expressions,
                used_context_variables=list(self._used_context_variables),
                before_entering=list(self._before_entering),
                after_exiting=list(self._after_exiting),
            )
        else:
            return MovingState(
                speed_expressions=self._speed_expressions,
                used_context_variables=list(self._used_context_variables),
                before_entering=list(self._before_entering),
                after_exiting=list(self._after_exiting),
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
                expression_final_value_temp = NameGenerator(f"state{self._identifier}_val_tmp")
                input_arg_string: str = str(tuple(expression)).replace("'", "")

                match self._pattern_type:
                    case PatternType.Full:
                        expression: Tuple[str, str, str, str]
                        val_temp_name: str = expression_final_value_temp()
                        full_expression = expression[0]
                        for varname in self._used_context_variables:

                            # Create context retrieval functions using expressions
                            fn: Callable[[], Any] = con.register_context_getter(varname)
                            context[getter_func_var_name := getter_function_name_generator()] = fn
                            full_expression = self._replace_var(
                                full_expression, varname, getter_func_var_name, getter_temp_name_generator()
                            )

                        input_arg_string = (
                            f"({val_temp_name}:=({full_expression}),{val_temp_name},{val_temp_name},{val_temp_name})"
                        )
                    case PatternType.LR:
                        expression: IndividualExpressionPattern
                        l_val_temp_name: str = expression_final_value_temp()
                        r_val_temp_name: str = expression_final_value_temp()
                        lr_expression: str = f"{expression[0]}{__MAGIC_SPLIT_CHAR__}{expression[-1]}"
                        for varname in self._used_context_variables:

                            # Create context retrieval functions using expressions
                            fn: Callable[[], Any] = con.register_context_getter(varname)
                            context[getter_func_var_name := getter_function_name_generator()] = fn
                            lr_expression = self._replace_var(
                                lr_expression, varname, getter_func_var_name, getter_temp_name_generator()
                            )
                        left_expression, right_expression = lr_expression.split(__MAGIC_SPLIT_CHAR__)

                        match isinstance(expression[0], int), isinstance(expression[-1], int):
                            case True, True:
                                raise TokenizeError(f"Should never be here!")
                            case False, False:

                                input_arg_string = f"({l_val_temp_name}:=({left_expression}),{l_val_temp_name},{r_val_temp_name}:=({right_expression}),{r_val_temp_name})"
                            case False, True:
                                input_arg_string = f"({l_val_temp_name}:=({left_expression}),{l_val_temp_name},{right_expression},{right_expression})"
                            case True, False:
                                input_arg_string = f"({left_expression},{left_expression},{r_val_temp_name}:=({right_expression}),{r_val_temp_name})"

                    case PatternType.Individual:
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
                    case _:
                        raise TokenizeError(f"Unknown expression type, got {self._pattern_type}")
                state_tokens.append(f".set_motors_speed({input_arg_string})")

            case None, speeds:
                state_tokens.append(f".set_motors_speed({tuple(speeds)})")
            case _:
                raise TokenizeError("should never reach here")

        tokens: List[str] = before_enter_tokens + state_tokens + after_exiting_tokens
        return tokens, context

    @staticmethod
    def _replace_var(source: str, var_name: str, func_name: str, temp_name: str) -> str:
        if source.count(var_name) == 1:
            return source.replace(var_name, f"{func_name}()", 1)
        else:
            return source.replace(var_name, f"({temp_name}:={func_name}())", 1).replace(var_name, temp_name)

    def __hash__(self) -> int:
        return self._identifier

    def __eq__(self, other: Self) -> bool:
        if self._speeds is None or other._speeds is None:
            return self._speed_expressions == other._speed_expressions
        elif self._speeds is None:
            return False
        elif other._speeds is None:
            return False
        else:

            return all(np.equal(self._speeds, other._speeds)) and self._speed_expressions == other._speed_expressions

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
        Initialize a MovingTransition object.

        Args:
            duration: The transition duration, must be a non-negative float.
            breaker: An optional callback function that can return a key (of type KT), a boolean, or any other value, used to interrupt the current state transition.
            check_interval: The frequency at which to check for state transition, i.e., how often in seconds to check.
            from_states: The starting states for the transition, can be a MovingState instance or a sequence of them.
            to_states: The destination states mapped to corresponding MovingState instances, or directly a MovingState instance.

        Raises:
            ValueError: If duration is negative, or from_states, to_states parameters are incorrectly formatted.
        """

        # Validate the duration
        if duration < 0:
            raise ValueError(f"Duration can't be negative, got {duration}")

        # Initialize attributes
        self.duration: float = duration
        self.breaker: Optional[Callable[[], Any]] = breaker
        self.check_interval: float = check_interval

        # Process the initial states parameter
        match from_states:
            case None:
                self.from_states: List[MovingState] = []
            case state if isinstance(state, MovingState):
                self.from_states: List[MovingState] = [from_states]
            case state if isinstance(state, Sequence) and all(isinstance(s, MovingState) for s in state):
                self.from_states: List[MovingState] = list(from_states)
            case _:
                raise ValueError(f"Invalid from_states, got {from_states}")

        # Process the target states parameter
        match to_states:
            case None:
                self.to_states: Dict[KT, MovingState] = {}
            case state if isinstance(state, MovingState):
                self.to_states: Dict[KT, MovingState] = {__PLACE_HOLDER__: state}
            case state if isinstance(state, Dict):
                self.to_states: Dict[KT, MovingState] = to_states
            case _:
                raise ValueError(f"Invalid to_states, got {to_states}")

        # Assign a unique transition ID
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
        match len(self.to_states):
            case 0:
                raise TokenizeError(f"Transition must have at least one to_state, got {self.to_states}.")
            case 1 if not callable(self.breaker):
                tokens.append(f".delay({self.duration})")
            case 1 if callable(self.breaker):
                context[(breaker_name := name_generator())] = self.breaker
                tokens.append(f".delay_b({self.duration},{breaker_name},{self.check_interval})")
            case length if length > 1 and callable(self.breaker):
                context[(breaker_name := name_generator())] = self.breaker
                tokens.append(f".delay_b_match({self.duration},{breaker_name},{self.check_interval})")
            case length if length > 1 and not callable(self.breaker):
                raise TokenizeError(
                    f"got branching states {self.to_states}, but not give correct breaker, {self.breaker} is not a callable."
                )
            case _:
                raise TokenizeError(f"Undefined Error, got {self.to_states} and {self.breaker}.")
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
            list(self.from_states),
            dict(self.to_states),
        )

    def __eq__(self, other):
        return (
            self.duration == other.duration
            and self.breaker == other.breaker
            and self.check_interval == other.check_interval
            and self.from_states == other.from_states
            and self.to_states == other.to_states
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
    """
    Args:
        *controller* : A `CloseLoopController` object that represents the bot's controller.
        *token_pool* : A `TokenPool` object that represents the bot's token pool.
    """

    def __init__(self, controller: CloseLoopController, token_pool: Optional[TokenPool] = None):
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
        # Identifies states with an indegree of zero
        zero_indegree_states = list(states_indegree := Botix.acquire_start_states(token_pool))

        # Validates that there is exactly one state with an indegree of zero
        if len(zero_indegree_states) == 1:
            return zero_indegree_states[0]
        else:
            if none_check:
                # Raises an error if none_check is enabled and no unique starting state is present
                raise ValueError(f"There must be exactly one state with a zero indegree, got {states_indegree}")
            return None

    @staticmethod
    def acquire_start_states(token_pool: TokenPool) -> Set[MovingState]:
        """
        Calculates the starting states in the given token pool.

        Parameters:
            token_pool (TokenPool): A list of MovingTransition objects representing the token pool.

        Returns:
            Set[MovingState]: A set of MovingState objects representing the starting states in the token pool.
                The starting states are the states with an indegree of zero.

        Algorithm:
            1. Initialize a dictionary to hold the indegree of each state.
            2. Calculate the indegree for each state by examining token transitions.
            3. Identify states with an indegree of zero.
            4. Return a set of MovingState objects representing the starting states.

        Note:
            - The indegree of a state is the number of tokens that can reach that state.
            - A state is considered a starting state if it has an indegree of zero.
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
        zero_indegree_states = {state for state, indegree in states_indegree.items() if indegree == 0}
        return zero_indegree_states

    @staticmethod
    def acquire_end_states(token_pool: TokenPool) -> Set[MovingState]:
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
            for to_state in token.to_states.values():
                states_outdegree[to_state] = states_outdegree.get(to_state, 0)

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

        This function explores all possible paths from a unique starting state, identifying loops in the process.
        A loop is defined as a series of MovingState objects that eventually return to a previously visited state.

        Returns:
            List[List[MovingState]]: A list containing all looping paths, with each path being a list of MovingState objects.
        """

        # Initialize necessary variables for path exploration and loop detection.
        start_state: MovingState = self.acquire_unique_start(self.token_pool)  # Starting state for path exploration
        search_queue: Queue[MovingState] = Queue(maxsize=1)  # queue to keep track of states yet to explore
        search_queue.put(start_state)
        loops: List[List[MovingState]] = []  # List to store loops found during exploration

        # Variables for managing path tracking, branch depths, and rollback operations within the search algorithm.
        chain: List[MovingState] = []  # Current path being explored
        branch_dict: Dict[MovingState, int] = {}  # Dictionary to track the depth of exploration for each state
        rolling_back: bool = False  # Flag indicating if the exploration is rolling back due to a loop detection
        rollback_to_start: bool = False  # Flag indicating if the exploration is rolling back to the start state
        # Main loop to explore all possible paths and identify loops
        while not rollback_to_start:
            # Pop the current state from the stack for exploration
            cur_state: MovingState = search_queue.get(False)

            # Attempt to find a connected forward transition from the current state
            connected_transition: MovingTransition = self.acquire_connected_forward_transition(
                cur_state, none_check=False
            )

            # Decision making based on the presence of a connected transition and the current exploration status
            match rolling_back, bool(connected_transition):
                case True, False:
                    # This case should theoretically never be reached, indicating a logic error
                    raise ValueError("Should not arrive here!")
                case True, True:

                    # Handling rollback and progression through a loop
                    connected_states = list(connected_transition.to_states.values())
                    cur_depth_index: int = branch_dict.get(cur_state)
                    if cur_depth_index == len(connected_states):
                        if hash(cur_state) == hash(start_state):
                            rollback_to_start = True
                            continue
                        # Loop completion, backtracking to continue exploration
                        search_queue.put(chain.pop())
                    else:
                        branch_dict[cur_state] = cur_depth_index + 1

                        # Progressing to the next state within the loop
                        next_state = connected_states[cur_depth_index]
                        chain.append(cur_state)
                        search_queue.put(next_state)
                        rolling_back = False
                case False, False:
                    # Backtracking to continue exploration from a previous state
                    rolling_back = True
                    search_queue.put(chain.pop())
                    continue
                case False, True:
                    # Progressing to a new state and potentially identifying a loop
                    connected_states = list(connected_transition.to_states.values())

                    cur_depth_index: int = branch_dict.get(cur_state, 0)

                    if cur_depth_index == len(connected_states):
                        # Loop completion, preparing for backtracking
                        rolling_back = True
                        search_queue.put(chain.pop())
                    else:
                        chain.append(cur_state)  # Add current state to the current path
                        branch_dict[cur_state] = cur_depth_index + 1
                        # Checking for and handling loop detection
                        next_state = connected_states[cur_depth_index]
                        state_hash = hash(next_state)
                        if any(state_hash == hash(state) for state in chain):
                            # Loop detected, appending the loop path to the loops list
                            loops.append(chain[chain.index(next_state) :])
                            rolling_back = True
                            search_queue.put(chain.pop())
                            continue

                        # Adding the new state to the chain and continuing exploration
                        search_queue.put(next_state)

        return loops  # Returning the list of identified loops

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
                err_out = "\n".join(str(x) for x in response)
                raise ValueError(
                    f"A state can *ONLY* connect to ONE transition as its input. Found conflicting transitions:\n {err_out}"
                )

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
        end_states = self.acquire_end_states(self.token_pool)
        self.ensure_accessibility(self.token_pool, start_state, end_states)
        if self.acquire_loops():
            raise ValueError("Loops detected! All State-Transition should be implemented without loops.")

    def _assembly_match_cases(self, match_expression: str | List[str], cases: Dict[KT, str | List[str]]) -> List[str]:
        """
        Assembles a list of strings representing match cases based on the given match expression and cases dictionary.

        Parameters:
            match_expression (str): The match expression to be used.
            cases (Dict[str, str]): A dictionary containing the cases and their corresponding values.

        Returns:
            List[str]: A list of strings representing the match cases.
        """
        match_expression = "".join(match_expression) if isinstance(match_expression, list) else match_expression

        lines: List[str] = [f"match {match_expression}:"]
        for key, value in cases.items():
            case_expr: str = f"'{key}'" if isinstance(key, str) else f"{key}"
            lines.append(self._add_indent(f"case {case_expr}:", count=1))
            lines.extend(self._add_indent(value.split("\n") if isinstance(value, str) else value, count=2))
        lines.append(self._add_indent(f"case undefined:", count=1))
        lines.append(
            self._add_indent(
                f"raise ValueError(f'No matching case found, got {{undefined}}, not in {list(cases.keys())}')", count=2
            )
        )
        return lines

    @staticmethod
    def _add_indent(lines: T_EXPR, indent: str = "    ", count: int = 1) -> T_EXPR:
        """
        Adds an indent to each line in the given list of lines or string.

        Parameters:
            lines (T_EXPR): The list of lines or string to add an indent to.
            indent (str, optional): The string to use for the indent. Defaults to " ".

        Returns:
            T_EXPR: The list of lines with the indent added, or the string with the indent added.
        """
        final_indent = indent * count
        match lines:
            case line_seq if isinstance(line_seq, list) and all((isinstance(line, str) for line in line_seq)):
                return [f"{final_indent}{line}" for line in line_seq]
            case lines_string if isinstance(lines_string, str):
                lines_string: str
                lines_list = lines_string.split("\n")
                return "\n".join(f"{final_indent}{line}" for line in lines_list)
            case _:
                raise TypeError(f"Expected list or string, but got {type(lines)}")

    def acquire_max_branchless_chain(self, start: MovingState) -> Tuple[List[MovingState], List[MovingTransition]]:
        """
        Retrieves the longest branchless chain of states starting from the given initial state.

        Parameters:
        - start: MovingState, the starting state for the search.

        Returns:
        - List[MovingState], the longest branchless chain of states from the start to some terminal state.
        """

        # Initialize the search queue and the longest branchless chain
        search_queue: Queue[MovingState] = Queue(maxsize=1)
        search_queue.put(start)
        max_chain_states: List[MovingState] = []
        max_chain_transitions: List[MovingTransition] = []
        # Perform a breadth-first search
        while not search_queue.empty():
            cur_state = search_queue.get()  # Get the current state

            # Attempt to acquire the next state connected to the current one
            connected_transition: MovingTransition | None = self.acquire_connected_forward_transition(
                cur_state, none_check=False
            )

            max_chain_states.append(cur_state)  # Add the current state to the max chain

            match connected_transition:
                case None:
                    # If no next state, end the search path
                    pass
                case branchless_transition if len(branchless_transition.to_states) == 1:
                    # If the next state is branchless, enqueue it for further search
                    max_chain_transitions.append(branchless_transition)
                    search_queue.put(list(branchless_transition.to_states.values())[0])
                case _:
                    # If the next state has branches, ignore and continue with other paths
                    pass

        return max_chain_states, max_chain_transitions

    @staticmethod
    def _compile_branchless_chain(
        states: List[MovingState],
        transitions: List[MovingTransition],
        controller: Optional[CloseLoopController] = None,
    ) -> Tuple[List[str], Context]:
        """
        Retrieves information from states and transitions to compile a branchless chain.

        Parameters:
            - self: The current instance of the class.
            - states: A list of MovingState objects representing the states in the chain.
            - transitions: A list of MovingTransition objects representing the transitions between states.
            - controller: An optional CloseLoopController object, defaulting to None.

        Returns:
            - Tuple[List[str], Context]: A tuple containing a list of strings representing compiled branchless chain tokens and a context dictionary.

        Note:
            This function compiles a branchless chain by analyzing the states and transitions provided.
        """
        states = list(states)
        transitions = list(transitions)
        context: Context = {}
        state_temp_data = states.pop(0).tokenize(controller)
        ret: List[str] = []
        ret.extend(state_temp_data[0])
        context.update(state_temp_data[1])

        for state, transition in zip(states, transitions):
            state: MovingState
            transition: MovingTransition
            match state.tokenize(controller), transition.tokenize():
                case (state_tokens, state_context), (transition_tokens, transition_context):
                    ret.extend(transition_tokens)
                    context.update(transition_context)
                    ret.extend(state_tokens)
                    context.update(state_context)
        return ret, context

    def _recursive_compile_tokens(self, start: MovingState, controller_name: str, context: Context) -> List[str]:
        """
        Compiles the tokens for a recursive chain of states starting from the given `start` state.

        Args:
            start (MovingState): The starting state of the chain.
            context (Context): The context dictionary to update with the compiled tokens.

        Returns:
            List[str]: A list of strings representing the compiled tokens.

        This function recursively compiles the tokens for a chain of states starting from the given `start` state. It first
        retrieves the maximum branchless chain from the `start` state using the `acquire_max_branchless_chain` method.
        Then, it compiles the tokens for the branchless chain using the `_compile_branchless_chain` method and updates
        the `context` dictionary with the compiled tokens. If the maximum branchless chain has a connected forward
        transition, the function recursively compiles the tokens for the forward states and appends them to the
        `compiled_lines` list. Finally, the function returns the `compiled_lines` list.
        """
        # Initialize an empty list to hold the compiled lines of tokens
        compiled_lines: List[str] = []

        # Retrieve the maximum branchless chain from the starting state
        max_branchless_chain = self.acquire_max_branchless_chain(start=start)

        # Compile the tokens for the acquired branchless chain and update context
        compiled_tokens, compiled_context = self._compile_branchless_chain(
            *max_branchless_chain, controller=self.controller
        )
        context.update(compiled_context)

        line = f"{controller_name}" + "".join(compiled_tokens)

        # Check if the last state in the branchless chain has a connected forward transition
        match max_branchless_chain:
            case ([*_, last_state], _) if connected_forward_transition := self.acquire_connected_forward_transition(
                last_state, none_check=False
            ):

                branch_transition, branch_context = connected_forward_transition.tokenize()
                context.update(branch_context)
                match_expr = line + "".join(branch_transition)
                match_branch: Dict[KT, List[str]] = {}
                for case, value in connected_forward_transition.to_states.items():
                    match_branch[case] = self._recursive_compile_tokens(
                        start=value, context=context, controller_name=controller_name
                    )
                lines = self._assembly_match_cases(match_expression=match_expr, cases=match_branch)
                return lines
            case _:
                compiled_lines.append(line)
                # If no forward transition is present, return the compiled lines
                return compiled_lines

    @classmethod
    def export_structure(cls, save_path: str, transitions: TokenPool) -> Self:
        """
        Export the structure to a UML file based on the provided transitions.

        Args:
            save_path (str): The path to save the UML file.
            transitions (Optional[List[MovingTransition]]): The list of transitions to represent in the UML.

        Returns:
            Self: The current instance.
        """
        start_string = "@startuml\n"
        end_string = "@enduml\n"

        used_state: Dict[MovingState, str] = {}

        lines: List[str] = []
        name_gen: NameGenerator = NameGenerator(basename="state_")
        break_gen: NameGenerator = NameGenerator(basename="break_")
        for transition in transitions:
            for from_state in transition.from_states:

                if from_state in used_state:
                    from_state_alias = used_state.get(from_state)
                else:
                    from_state_alias: str = name_gen()
                    used_state[from_state] = from_state_alias
                    lines.insert(0, f'state "{from_state}" as {from_state_alias}\n')

                if len(transition.to_states) == 1:
                    to_state = list(transition.to_states.values())[0]
                    if to_state not in used_state:
                        to_state_alias: str = name_gen()
                        used_state[to_state] = to_state_alias
                        lines.insert(0, f'state "{to_state}" as {to_state_alias}\n')
                    else:
                        to_state_alias = used_state.get(to_state)
                    lines.append(f"{from_state_alias} --> {to_state_alias}\n")
                else:
                    break_node_name: str = break_gen()
                    lines.insert(0, f"state {break_node_name} <<choice>>\n")
                    lines.insert(
                        1, f"note right of {break_node_name}: {get_function_annotations(transition.breaker)}\n"
                    )
                    lines.append(f"{from_state_alias} --> {break_node_name}\n")
                    for case_name, to_state in transition.to_states.items():

                        if to_state not in used_state:
                            to_state_alias: str = name_gen()
                            used_state[to_state] = to_state_alias
                            lines.insert(0, f'state "{to_state}" as {to_state_alias}\n')
                        else:
                            to_state_alias = used_state.get(to_state)

                        lines.append(f"{break_node_name} --> {to_state_alias}: {case_name}\n")

        start_states: Set[MovingState] = Botix.acquire_start_states(token_pool=transitions)
        end_states: Set[MovingState] = Botix.acquire_end_states(token_pool=transitions)

        start_heads: List[str] = [f"[*] --> {used_state.get(sta)}\n" for sta in start_states]
        end_heads: List[str] = [f"{used_state.get(sta)} --> [*]\n" for sta in end_states]

        with open(save_path, "w") as f:

            f.writelines([start_string, *lines, "\n", *start_heads, "\n", *end_heads, "\n", end_string])
        return cls

    def compile(self, return_median: bool = False) -> Callable[[], None] | Tuple[List[str], Context]:
        """
        Compiles the bot's code and returns a callable function or a tuple of compiled lines and context.

        Args:
            return_median (bool, optional): Whether to return the compiled lines and context instead of a callable function. Defaults to False.

        Returns:
            Callable[[], None] | Tuple[List[str], Context]: The compiled function or a tuple of compiled lines and context.

        Raises:
            None

        Description:
            This function compiles the bot's code and returns a callable function that can be executed. It first checks the requirements of the bot's code using the `_check_met_requirements` method. Then, it creates a function name, function head, and controller name. It initializes an empty context dictionary with the controller name as the key and the bot's controller as the value.

            It retrieves the unique start state from the token pool using the `acquire_unique_start` method.

            Next, it compiles the tokens for the recursive chain of states using the `_recursive_compile_tokens` method. It adds the function head to the beginning of the compiled lines and adds the necessary indentation.

            If `return_median` is True, it returns the compiled lines and context as a tuple. Otherwise, it executes the compiled lines using the `exec` function and retrieves the compiled function from the context dictionary. The compiled function is then returned.
        """

        self._check_met_requirements()
        function_name = "_func"
        function_head = f"def {function_name}():"
        controller_name = "con"
        context: Context = {controller_name: self.controller}

        start_state: MovingState = self.acquire_unique_start(self.token_pool)

        function_lines = self._add_indent(
            self._recursive_compile_tokens(start=start_state, context=context, controller_name=controller_name)
        )
        function_lines.insert(0, function_head)

        if return_median:
            return function_lines, context
        exec("\n".join(function_lines), context)
        compiled_obj: Callable[[], None] = context.get(function_name)
        return compiled_obj


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
