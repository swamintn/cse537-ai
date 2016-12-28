#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementaion of A* and IDA*

Definitions:

State:
This will be a 2-D list defining the state of the puzzle. It will contain
the elements of the puzzles as a tuple of tuples. A gap will be represented 
by the python None
Eg:
1,2,4        ((1,2   ,4),
3, ,5     =>  (3,None,5),
6,8,7         (6,8   ,7))

Heuristic functions:
All heuristic functions should accept a state and puzzle length(n for nxn) 
as input and return an int value as output. It is assumed that the state is 
a valid one (But it can be any nxn state). Currently supported heuristics,
1) Misplaced Tiles Heuristic
2) Manhattan Distance Heuristic (Default)

Algorithms:
1) A_Star
2) IDA_Star
Both algorithms are classes that must be initialized with a start state and a 
heuristic function. Calling solve() on the initialized object will generate the
solution. The solution will be a list of moves, saved in the 'solution' attribute

Execution:
    puzzleSolver.py <#Algorithm> <N> <INPUT_FILE_PATH> <OUTPUT_FILE_PATH>
    where,
        A_Star is used if #Algorithm is 1, IDA_Star if #Algorithm is 2
        N is the puzzle span(eg.: 3 for 3x3)

Extra helpers:
    As an added benefit, once you've called solve, you can call print_steps_to_solution to
    print a nice visual representation of how the goal is reached from the start.
    This line has been added at the end of this program.

@author: Swaminathan Sivaraman
"""
from   queue import PriorityQueue as DefPriorityQueue
import math
import sys

# Constants

LEFT, RIGHT, UP, DOWN = 'L', 'R', 'U', 'D'
ACTIONS = [LEFT, RIGHT, UP, DOWN]
GAP = None
MOVE_COST = 1

INVERSE_ACTIONS = {
    'L': 'R',
    'R': 'L',
    'U': 'D',
    'D': 'U'
}

# Helpers

def pretty_print(state):
    """
    Prints a nice visual representation of a state
    """
    max_size = len(str(len(state)**2))
    for row in state:
        print(' '.join((str(tile) if tile else ' ').rjust(max_size) for tile in row))


class PriorityQueue:
    """
    A Priority Queue with an added feature of breaking ties by choosing 
    the element that was added most recently.

    (This will improve the time complexity here since for two nodes with equal f(n) values, it
     makes sense to choose a node with higher g(n) value(i.e. later element) first as
     the search space(or h(n)) will be lower for it than for the first element.)
    """
    def __init__(self, *args, **kwargs):
        self.pq = DefPriorityQueue()
        self.count = 0
    def put(self, priority, item):
        self.count += -1
        self.pq.put((priority, self.count, item))
    def get(self):
        priority, _, item = self.pq.get()
        return (priority, item)
    def qsize(self):
        return self.pq.qsize()


# Heuristics

def misplaced_tiles_heuristic(state, puzzle_size):
    """
    This heuristic calculates the distance to goal
    state based on number of misplaced tiles
    """
    heuristic = 0
    for i, row in enumerate(state):
        for j, tile in enumerate(row):
            # Don't calculate heuristic for the gap tile
            if tile == GAP:
                continue
            # Find the expected tile at this position. None(gap) if last position
            if i == j == (puzzle_size - 1):
                expected_tile = GAP
            else:
                expected_tile = (i*puzzle_size) + j + 1 # +1 to account for 0-indexing
            # Calculate heuristic
            if tile != expected_tile:
                heuristic = heuristic + 1

    return heuristic


def manhattan_distance_heuristic(state, puzzle_size):
    """
    This heuristic calculates the distance to goal
    state based on the Manhattan Distance for each tile
    """
    heuristic = 0
    for i, row in enumerate(state):
        for j, tile in enumerate(row):
            # Find the i, j values of where this tile should actually be.
            # Don't do this for the gap tile
            if tile == GAP:
                continue
            goal_i, goal_j = int((tile-1)//puzzle_size), (tile-1)%puzzle_size  # -1 to account for 0-indexing
            # Add difference to heuristic
            heuristic += abs(goal_i - i) + abs(goal_j -j)

    return heuristic


# The normal A* algorithm

class A_Star:

    def __init__(self, start_state, heuristic):
        
        self.start_state = self.tupleize(start_state)
        self.heuristic   = heuristic
        self.puzzle_size = len(start_state)
        self.total_size  = self.puzzle_size**2

        self.generate_goal_state()
        self.solution = None

    # Goal functions

    def generate_goal_state(self):
        """Generates the goal state for input puzzle size"""
        p = self.puzzle_size
        self.goal_state = [list(range(s, s+p)) for s in range(1, p**2, p)]
        self.goal_state[-1][-1] = None

        self.goal_state = self.tupleize(self.goal_state)

    def is_goal_state(self, state):
        """Check if given state is a goal state"""
        return state == self.goal_state

    # Action functions

    def get_allowed_actions(self, state, disallowed_action=None):
        """Get allowed list of actions for given state"""
        my_actions = ACTIONS[:]

        i, j = self.find_element(state, GAP)
        # At top extreme, can't move up
        if i == 0:
            my_actions.remove(UP)
        # At bottom extreme, can't move down
        elif i == self.puzzle_size - 1:
            my_actions.remove(DOWN)
        # At left extreme, can't move left
        if j == 0:
            my_actions.remove(LEFT)
        # At right extreme, can't move right
        elif j== self.puzzle_size - 1:
            my_actions.remove(RIGHT)

        try: my_actions.remove(disallowed_action)
        except ValueError: pass

        return my_actions

    def get_new_state(self, state, action):
        """
        Gets the new state by executing the given action on the given state
        """
        new_state = list(list(r) for r in state)
        
        i, j = self.find_element(state, GAP)
        if action == LEFT:
            new_state[i][j], new_state[i][j-1] = new_state[i][j-1], new_state[i][j]
        elif action == RIGHT:
            new_state[i][j], new_state[i][j+1] = new_state[i][j+1], new_state[i][j]
        elif action == UP:
            new_state[i][j], new_state[i-1][j] = new_state[i-1][j], new_state[i][j]
        elif action == DOWN:
            new_state[i][j], new_state[i+1][j] = new_state[i+1][j], new_state[i][j]

        return self.tupleize(new_state)

    # Actual Algorithm

    def solve(self):

        self.explored = {}
        self.frontier = PriorityQueue()

        # Construct the start_node and add it to the frontier
        self.start_node = self.make_node(self.start_state, prev_state=None, action=None)
        self.frontier.put(self.start_node['TOTAL_POSSIBLE_COST'], self.start_node)

        # Start it off...
        while self.frontier.qsize() != 0:
            _, node = self.frontier.get()
            current_state = node['CURRENT_STATE']
            if self.is_goal_state(current_state):
                break
            if current_state not in self.explored:
                self.explored[current_state] = node
                disallowed_action = INVERSE_ACTIONS.get(node['ACTION_USED'])
                for action in self.get_allowed_actions(current_state, disallowed_action):
                    new_state = self.get_new_state(current_state, action)
                    new_node  = self.make_node(new_state, current_state, action)
                    self.frontier.put(new_node['TOTAL_POSSIBLE_COST'], new_node)
            
        # Build solution
        goal_node = node
        solution = []
        while node['PREVIOUS_STATE'] is not None:
            solution.append(node['ACTION_USED'])
            node = self.explored[node['PREVIOUS_STATE']]
        
        self.solution = list(reversed(solution))

        return self.solution

    # Node maker

    def make_node(self, state, prev_state, action):
        node = {}

        # Special case for start node
        if prev_state is None:
            node['ACTUAL_COST'] = 0
            node['DEPTH']       = 0
        # Now handle the normal case
        else:
            node['ACTUAL_COST'] = self.explored[prev_state]['ACTUAL_COST'] + MOVE_COST # g(n)
            node['DEPTH']       = self.explored[prev_state]['DEPTH'] + 1

        # Calculate the rest
        node['POSSIBLE_COST_TO_GOAL'] = self.heuristic(state, self.puzzle_size) # h(n)
        node['TOTAL_POSSIBLE_COST']   = node['ACTUAL_COST'] + node['POSSIBLE_COST_TO_GOAL'] # f(n)
        node['CURRENT_STATE']  = state
        node['PREVIOUS_STATE'] = prev_state
        node['ACTION_USED']    = action

        return node 


    # Helper to see progression of events from start to solution
    # This must be called only if the puzzle is solved

    def print_steps_to_solution(self):
        print('\nStarting off as...')
        pretty_print(self.start_state)
        state = self.start_state
        for action in self.solution:
            print('\nApplying action %s' % action)
            state = self.get_new_state(state, action)
            pretty_print(state)

    # Static helpers

    @staticmethod
    def find_element(state, element):
        """
        Returns the (i, j) tuple of the indices of the element in the given 2D state
        """
        a, b = None, None
        for i, row in enumerate(state):
            try:
                a, b = i, row.index(element)
                break
            except ValueError:
                pass
        return (a, b)

    @staticmethod
    def parse(state_string):
        """
        A helper method to parse a state from a string and convert
        it into the expected state format (tuple of tuples)
        eg:
        '1,2,3\n4,,5\n6,7,8' => ((1, 2, 3), (4, None, 5), (6, 7, 8))
        """
        state = []
        for row in state_string.rstrip('\n').split('\n'):
            tiles = row.split(',')
            tiles = [int(t) if t else None for t in tiles]
            state.append(tiles)
        return A_Star.tupleize(state)

    @staticmethod
    def tupleize(state):
        return tuple(tuple(row) for row in state)


# Memory-bounded A*

class IDA_Star(A_Star):
    """
    The Iterative-Deepening version of the A* algorithm

    (Though an extension of A*, note that explored and frontier are not used here)
    """
    def solve(self):
        """
        IDA* solver
        """
        # The initial threshold is the estimated cost at the root
        threshold = self.heuristic(self.start_state, self.puzzle_size)

        while self.solution is None:
            self.solution, new_threshold = self.depth_first_search(self.start_state, 0, threshold, None)
            threshold = new_threshold

        return self.solution

    def depth_first_search(self, state, cost, threshold, action_used):
        """
        Searches for the goal state in a depth-first manner within the
        given threshold

        This method should always return a 2-element tuple - 

        The first element will hold the solution, which should be a list,
        eg: ['L', 'D']. 'None' will signify that no solution was found.
        (Note that [] will signify that the goal has been found and that the
         given state itself is a goal state)

        The second element should be the new suggested threshold. This can be
        ignored by the caller if a solution is found
        """
        # f(n) at this node
        f_n = cost + self.heuristic(state, self.puzzle_size)

        # This condition means that this node is a leaf node for this threshold
        # (This check must be done before the goal node check since this node must
        #  not be included in the solution if it exceeds the threshold on this pass)
        if f_n > threshold:
            return None, f_n

        # Check for goal state
        if state == self.goal_state:
            return [], f_n

        # Go through the possible actions for this state
        new_suggested_threshold = math.inf
        disallowed_action = INVERSE_ACTIONS.get(action_used)
        for action in self.get_allowed_actions(state, disallowed_action):
            new_state = self.get_new_state(state, action)
            solution, new_threshold = self.depth_first_search(new_state, cost + MOVE_COST, threshold, action)
            if solution is not None:
                return [action] + solution, threshold
            new_suggested_threshold = min(new_suggested_threshold, new_threshold)

        # If we reach here, we didn't find any solutions for any of the actions
        # Return None and the new smallest higher threshold
        return None, new_suggested_threshold


def main():
    # Process command-line options
    # puzzleSolver.py <#Algorithm> <N> <INPUT_FILE_PATH> <OUTPUT_FILE_PATH>
    try:
        algo, puzzle_len, in_file, out_file = sys.argv[1:]
    except ValueError:
        print("Run using puzzleSolver.py <#Algorithm> <N> <INPUT_FILE_PATH> <OUTPUT_FILE_PATH>")
        print(__doc__)
        sys.exit(1)

    # Read the input file and convert it into a format we expect
    in_data = open(in_file).read()
    start_state = A_Star.parse(in_data)

    # Algorithm to use
    A_Star_Algo = A_Star if algo == '1' else IDA_Star

    # Initialize Algo and solve
    puzzle = A_Star_Algo(start_state, heuristic=manhattan_distance_heuristic)
    puzzle.solve()

    # Print result to output file
    out_fd = open(out_file, 'w')
    out_fd.write(','.join(puzzle.solution) + '\n')
    out_fd.close()

    print('Solution is ', ','.join(puzzle.solution) + '\n')
    # Print steps to solution
    print('Printing solution steps...')
    puzzle.print_steps_to_solution()

if __name__ == '__main__':
    main()

