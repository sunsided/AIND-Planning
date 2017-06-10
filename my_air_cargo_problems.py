from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """
        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """

        # Create concrete Action objects based on the domain action schema for: Load, Unload, and Fly
        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        def load_actions():
            """Create all concrete Load actions and return a list

            :return: list of Action objects
            """
            loads = []
            for c in self.cargos:
                for p in self.planes:
                    for a in self.airports:
                        precond_pos = [expr("At({}, {})".format(c, a)),
                                       expr("At({}, {})".format(p, a)),
                                       # expr("Cargo({})".format(c)),
                                       # expr("Plane({})".format(p)),
                                       # expr("Airport({})".format(a))
                                       ]
                        precond_neg = []
                        effect_add = [expr("In({}, {})".format(c, p))]
                        effect_rem = [expr("At({}, {})".format(c, a))]
                        load = Action(expr("Load({}, {}, {})".format(c, p, a)),
                                      [precond_pos, precond_neg],
                                      [effect_add, effect_rem])
                        loads.append(load)
            return loads

        def unload_actions():
            """Create all concrete Unload actions and return a list

            :return: list of Action objects
            """
            unloads = []
            for c in self.cargos:
                for p in self.planes:
                    for a in self.airports:
                        precond_pos = [expr("In({}, {})".format(c, p)),
                                       expr("At({}, {})".format(p, a)),
                                       # expr("Cargo({})".format(c)),
                                       # expr("Plane({})".format(p)),
                                       # expr("Airport({})".format(a))
                                       ]
                        precond_neg = []
                        effect_add = [expr("At({}, {})".format(c, a))]
                        effect_rem = [expr("In({}, {})".format(c, p))]
                        unload = Action(expr("Unload({}, {}, {})".format(c, p, a)),
                                        [precond_pos, precond_neg],
                                        [effect_add, effect_rem])
                        unloads.append(unload)
            return unloads

        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           # expr("Plane({})".format(p)),
                                           # expr("Airport({})".format(fr)),
                                           # expr("Airport({})".format(to)),
                                           ]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def pos_states(self, state):
        """ Return the states that are known to be true.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'ß
        :return: list of state objects
        """
        return [pair[0] for pair in zip(self.state_map, state) if pair[1] == 'T']

    def neg_states(self, state):
        """ Return the states that are known to be false.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of state objects
        """
        # Checking for false explicitly due to open world assumption.
        return [pair[0] for pair in zip(self.state_map, state) if pair[1] == 'F']

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        # TODO: It should be possible to use Action.check_precond() here, but the kb and args params are unclear.
        possible_actions = [a for a in self.actions_list
                            if all(p in self.pos_states(state) for p in a.precond_pos)
                            and all(p in self.neg_states(state) for p in a.precond_neg)]

        return possible_actions

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        pos_states = self.pos_states(state)
        neg_states = self.neg_states(state)

        for s in action.effect_rem:
            assert s not in neg_states
            assert s in pos_states
            pos_states.remove(s)
            neg_states.append(s)

        for s in action.effect_add:
            assert s in neg_states
            assert s not in pos_states
            neg_states.remove(s)
            pos_states.append(s)

        new_state = FluentState(pos_states, neg_states)
        return encode_state(new_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        # TODO implement (see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2)

        pos_states = self.pos_states(node.state)
        all_goals = set(self.goal)

        count = 0
        open_goals = set(g for g in all_goals if g not in pos_states)
        for a in self.actions_list:  # type: Action
            pos_effects = set(a.effect_add).intersection(open_goals)
            neg_effects = set(a.effect_rem).intersection(all_goals)
            if len(pos_effects) > 0:
                count += 1
                open_goals.difference_update(pos_effects)
                open_goals.update(neg_effects)

        return count


def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def build_positives(predicate, rules):
    return [expr('{}({}, {})'.format(predicate, a0, a1))
            for (a0, a1) in rules]


def build_negatives(predicate, arg0, arg1, exceptions=()):
    if exceptions is None:
        exceptions = []
    return [expr('{}({}, {})'.format(predicate, a0, a1))
            for a0 in arg0
            for a1 in arg1
            if (a0, a1) not in exceptions]


def air_cargo_p2() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3']
    planes = ['P1', 'P2', 'P3']
    airports = ['JFK', 'SFO', 'ATL']

    cargo_at = [('C1', 'SFO'), ('C2', 'JFK'), ('C3', 'ATL')]
    plane_at = [('P1', 'SFO'), ('P2', 'JFK'), ('P3', 'ATL')]

    pos = build_positives('At', cargo_at) + build_positives('At', plane_at)

    neg = build_negatives('At', cargos, airports, cargo_at) \
        + build_negatives('In', cargos, planes) \
        + build_negatives('At', planes, airports, plane_at)

    init = FluentState(pos, neg)
    goal = build_positives('At', [('C1', 'JFK'), ('C2', 'SFO'), ('C3', 'SFO')])

    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3', 'C4']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO', 'ATL', 'ORD']

    cargo_at = [('C1', 'SFO'), ('C2', 'JFK'), ('C3', 'ATL'), ('C4', 'ORD')]
    plane_at = [('P1', 'SFO'), ('P2', 'JFK')]

    pos = build_positives('At', cargo_at) + build_positives('At', plane_at)

    neg = build_negatives('At', cargos, airports, cargo_at) \
        + build_negatives('In', cargos, planes) \
        + build_negatives('At', planes, airports, plane_at)

    init = FluentState(pos, neg)
    goal = build_positives('At', [('C1', 'JFK'), ('C2', 'SFO'), ('C3', 'JFK'), ('C4', 'SFO')])

    return AirCargoProblem(cargos, planes, airports, init, goal)
