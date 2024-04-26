from collections import Counter
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

from ..tools.generators import NameGenerator
from .exceptions import StructuralError, TokenizeError

Expression: TypeAlias = str | int

FullPattern: TypeAlias = Tuple[int]
LRPattern: TypeAlias = Tuple[int, int]
IndividualPattern: TypeAlias = Tuple[int, int, int, int]
FullExpressionPattern: TypeAlias =  str
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
    def __init__(
        self,
        *speeds: Unpack[FullPattern] | Unpack[LRPattern] | Unpack[IndividualPattern],
        speed_expressions: Optional[
            FullExpressionPattern| LRExpressionPattern | IndividualExpressionPattern
        ] = None,
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
        self._speed_expressions:IndividualExpressionPattern
        self._speeds: np.array

        match bool(speed_expressions), bool(speeds):
            case True, False:
                if used_context_variables is None:
                    raise ValueError("No used_context_variables provided, You must provide a names set that contains all the name of the variables used in the speed_expressions."
                                     "If you do not need use context variables, then you should use *speeds argument to create the MovingState.")
                self._speeds = None
                match speed_expressions:
                    case str(full_expression):
                        self._speed_expressions = (full_expression, full_expression, full_expression, full_expression)
                    case (left_expression, right_expression):
                        if all(isinstance(item, int) for item in speed_expressions):
                            raise ValueError(f"All expressions are integers. You should use *speeds argument to create the MovingState, got {speed_expressions}")

                        self._speed_expressions = (left_expression,left_expression, right_expression,right_expression)
                    case speed_expressions if len(speed_expressions) == 4:
                        if all(isinstance(item, int) for item in speed_expressions):
                            raise ValueError(f"All expressions are integers. You should use *speeds argument to create the MovingState, got {speed_expressions}")
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
        self._before_entering:List[Callable[[], None]] = before_entering
        self._used_context_variables: Set[str] = used_context_variables
        self._after_exiting:List[Callable[[], None]] = after_exiting
        self._identifier:int = self.__state_id_counter__
        self.__state_id_counter__ += 1
    @staticmethod
    def _validate_used_context_variables_presence(used_context_variables: Set[str], speed_expressions: IndividualExpressionPattern) -> None:
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

    def unwrap(self) -> Tuple[int,...]:
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
            *tuple(self._speeds),
            speed_expressions=self._speed_expressions,
            used_context_variables=self._used_context_variables,
            before_entering=self._before_entering,
            after_exiting=self._after_exiting,
        )


    def tokenize(self,con:Optional[CloseLoopController])->Tuple[List[str],Context]:
        if not (bool(self._speeds) ^ bool(self._speed_expressions)):
            if self._speeds:
                raise TokenizeError(f"Cannot tokenize a state with both speed expressions and speeds, got {self._speeds} and {self._speed_expressions}.")
            else:
                raise TokenizeError(f"Cannot tokenize a state with no speed expressions and no speeds, got {self._speeds} and {self._speed_expressions}.")

        context: Context = {}
        context_updater_func_name_generator:NameGenerator=NameGenerator(f'state{self._identifier}_context_updater_')

        before_enter_tokens:List[str]=[]
        if self._before_entering:
            for func in self._before_entering:
                context[(func_var_name:=context_updater_func_name_generator())]=func
                before_enter_tokens.append(f'.wait_exec({func_var_name})')

        after_exiting_tokens:List[str]=[]
        if self._after_exiting:
            for func in self._after_exiting:
                context[(func_var_name:=context_updater_func_name_generator())]=func
                after_exiting_tokens.append(f'.wait_exec({func_var_name})')

        state_tokens:List[str]=[]
        match self._speed_expressions,self._speeds:
            case expression,None:
                if con is None:
                    raise  TokenizeError(f'You must parse a CloseLoopController to tokenize a state with expression pattern')
                # use expression to construct context getters
                context_getter_func_seq=[con.register_context_getter(varname) for varname in self._used_context_variables]
                getter_function_name_generator=NameGenerator(f'state{self._identifier}_context_getter_')
                getter_temp_name_generator=NameGenerator(f'state{self._identifier}_context_getter_temp_')
                input_arg_string=str(tuple(expression)).replace("'","")
                for varname,fun in zip(self._used_context_variables,context_getter_func_seq):
                    context[(func_var_name:=getter_function_name_generator())]=fun
                    temp_name=getter_temp_name_generator()
                    input_arg_string=input_arg_string.replace(varname,f'({temp_name}:={func_var_name}())',1)
                    input_arg_string=input_arg_string.replace(varname,temp_name)
                state_tokens.append(f'.set_motors_speed({input_arg_string})')


            case None,speeds:
                state_tokens.append(f'.set_motors_speed({tuple(speeds)})')
            case _:
                raise RuntimeError("should never reach here")
        tokens: List[str] = before_enter_tokens + state_tokens + after_exiting_tokens
        return tokens,context


    def __hash__(self) -> int:
        return self._identifier

    def __eq__(self, other: Self) -> bool:
        return tuple(self._speeds) == tuple(other._speeds) and self._speed_expressions == other._speed_expressions

    def __str__(self):
        return f"{self._identifier}-MovingState({self._speeds or self._speed_expressions})"


class MovingTransition:
    """
    A class that represents a moving transition.
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
            case state if isinstance(state, Dict):
                self.to_states: Dict[KT, MovingState] = to_states
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

    def make(self)->str:
        #TODO

    def __str__(self):
        return f"{self.from_states} -> {self.to_states}"


TokenPool: TypeAlias = List[MovingTransition]


class Botix:

    def __init__(self, controller: CloseLoopController, token_pool: Optional[List[MovingTransition]] = None):
        self.controller: CloseLoopController = controller
        self.token_pool: TokenPool = token_pool or []

    @staticmethod
    def ensure_unique_start(token_pool: TokenPool) -> MovingState:
        """
        Calculates the indegree of each state in the token pool to ensure unique starting states.

        This method iterates through all tokens in the token pool and counts the indegree (the number of incoming edges) for each state.
        It ensures that there is exactly one state with zero indegree, which represents the unique starting state.

        Args:
            token_pool (TokenPool): A list of MovingTransform objects representing the token pool.

        Raises:
            ValueError: If there is not exactly one state with a zero indegree.

        Returns:
            MovingState: The unique starting state if exactly one state has zero indegree.
        """
        # Initialize a dictionary to store the indegree of each state
        states_indegree: Dict[MovingState, int] = {}

        # Calculate the indegree of each state
        for token in token_pool:
            for state in token.to_states.values():
                # Increment the indegree of each state reached by the token
                states_indegree[state] = states_indegree.get(state, 0) + 1

        # Identify states with zero indegree
        zero_indegree_states = [state for state, indegree in states_indegree.items() if indegree == 0]

        # Ensure there is exactly one state with zero indegree
        if zero_indegree_states == 1:
            return zero_indegree_states[0]
        else:
            raise ValueError(f"There must be exactly one state with a zero indegree, got {zero_indegree_states}")

    @staticmethod
    def ensure_end_state(token_pool: TokenPool) -> Set[MovingState]:
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
        """
        # Initialize a queue for BFS and a set to keep track of states not accessible from the start state
        search_queue: List[MovingState] = [start_state]
        not_accessible_states: Set[MovingState] = end_states
        visited_states: Set[MovingState] = set()
        # Perform breadth-first search to find if all end states are accessible
        while search_queue:
            # Pop the next state from the queue to explore
            current_state: MovingState = search_queue.pop(0)
            connected_states: Set[MovingState] = []

            # Explore all tokens that can be reached from the current state
            for token in token_pool:
                if current_state in token.from_states:
                    # Add all to_states of the current token to the connected states list
                    connected_states.update(token.to_states.values())

            # Update the set of not accessible states by removing the states we just found to be connected
            not_accessible_states -= (connected_states-visited_states)
            visited_states+=connected_states
            # If there are no more not accessible states, we are done
            if len(not_accessible_states):
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

    def _check_met_requirements(self):
        self.ensure_structure_validity(self.token_pool)
        start_state = self.ensure_unique_start(self.token_pool)
        end_states = self.ensure_end_state(self.token_pool)
        self.ensure_accessibility(self.token_pool, start_state, end_states)

    def compile(self) -> Callable:
        """

        Returns:

        """
        function_context: Dict[str, Any] = {"con": self.controller}

        self._check_met_requirements()
        function_string: str = f"def _func():\n" f" con"

        return

    def _make_chain_expression(
        self, start_state: MovingState, end_state: MovingState, controller_name: str = "con"
    ) -> str:

        if start_state == end_state:
            #TODO fill the one state case
        self.ensure_accessibility(self.token_pool, start_state, {end_state})
        chain_elements: List[MovingState | MovingTransition] = [start_state]
        while True:
            current_state = chain_elements[-1]
            connected_transitions: List[MovingTransition] = list(
                filter(lambda trans: current_state in trans.from_states, self.token_pool)
            )
            if len(connected_transitions) > 1:
                raise StructuralError(
                    f"{current_state} is the input of more than one transition, which is not allowed.\n"
                    f"you can check the structure of the token pool using self.ensure_structure_validity method."
                )
            if len(connected_transitions) == 0:
                raise StructuralError(f"{current_state} is a terminate")


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
