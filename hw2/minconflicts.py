#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of CSP Solver using,
    Min-Conflict Local Search Algorithm with Random Restarts

Algorithms:
1) MinConflicts
This is a class that must be initialized with an input
graph. Calling solve() on the initialized object will generate the
solution. The solution will be a dict with assignments to all variables.
It will be None if no solution could be reached in a reasonable time.

Execution:
    minconflicts.py <INPUT_FILE_PATH> <OUTPUT_FILE_PATH>

@author: Swaminathan Sivaraman
"""
import random
import sys
import time

MAX_SECONDS = 60

class MinConflicts:
    """
    CSP Solver using the Min-Conflict Local Search algorithm
    """
    def __init__(self, graph):
        self.graph    = graph
        self.colors   = list(self.graph.values())[0]['COLORS'][:]
        self.solution = None
        
        # Performance details
        self.time_taken  = 0.0
        self.steps_taken = 0
        self.timed_out   = False

        # Configurations
        min_max_tries      = 10000
        self.max_tries     = max(min_max_tries, len(self.graph) * 10)
        self.restart_tries = 100

    def generate_random_state(self):
        """
        Generates a random solution
        """
        return {k: random.choice(self.colors) for (k, v) in self.graph.items()}

    def get_conflicts(self, solution):
        """
        Returns the number of conflicts and conflicting variables
        for the solution
        """
        conflicts, conflicting_vars = 0, set()
        for node in solution:
            for dep_node in set(self.graph[node]['NODES']):
                if solution[node] == solution[dep_node]:
                    conflicting_vars.add(node)
                    conflicting_vars.add(dep_node)
                    conflicts += 1
        return conflicts, list(conflicting_vars)

    def get_best_color(self, var, var_colors, solution):
        """
        Chooses and returns the color from var_colors
        that least conflict with the var's dependent node colors
        """
        color_wts = [0] * len(var_colors)
        for i, color in enumerate(var_colors):
            for dep_node in set(self.graph[var]['NODES']):
                if color == solution[dep_node]:
                    color_wts[i] += 1
        return [x for (y, x) in sorted(zip(color_wts, var_colors))][0]

    def solve(self):
        start_time = time.time()
        for i in range(self.restart_tries):
            # Generate a random solution
            possible_solution = self.generate_random_state()
            # Do hill-climbing for the conflicted variables
            for j in range(self.max_tries):
                self.time_taken   = time.time() - start_time
                self.steps_taken += 1
                conflicts, conflicting_vars = self.get_conflicts(possible_solution)
                if conflicts == 0:
                    self.solution = possible_solution
                    return self.solution
                if self.time_taken > MAX_SECONDS:
                    self.timed_out = True
                    self.solution  = None
                    return self.solution
                var = random.choice(conflicting_vars)
                var_colors = self.colors[:]
                var_colors.remove(possible_solution[var])
                var_color = self.get_best_color(var, var_colors, possible_solution)
                possible_solution[var] = var_color
            # Seems like there is no solution in this area of the state-space.
            # Give up and do a restart to go somewhere else in the state-space

        # No solution found
        return None


# Input file Parser

def get_graph(input_file):
    """
    Reads an input file of this format,
    NO_OF_VARIABLES NO_OF_RELATIONS NO_OF_COLORS
    variable1 variable2
    ...

    And returns a graph (a python dict). The keys will be
    the no_of_variables(n) from 0 to n-1. The values will be 
    again a dict with two keys,
        'NODES' : list of dependent nodes for this variable
        'COLORS': list of possible colors for this variable 
                  (Set to [0:no_of_colors] for all variables)
    """
    lines = open(input_file).read().splitlines()
    elements, edges, colors = map(int, lines[0].split())
    graph = {e: {'NODES' : [], 
                 'COLORS': list(range(colors))} for e in range(elements)}
    for edge in lines[1:]:
        start, end = map(int, edge.split())
        graph[start]['NODES'].append(end)
        graph[end]['NODES'].append(start)
    return graph

# Helper to verify if a solution is valid

def verify_solution(solution, graph):
    if solution == None:
        return True
    for node in solution:
        for dependent_node in graph[node]['NODES']:
            if solution[node] == solution[dependent_node]:
                return False
    return True

# Main method

def main():
    # Read command-line arguments
    if len(sys.argv) == 2 and sys.argv[1] == '--help':
        print(__doc__)
        sys.exit()
    try:
        input_file, output_file = sys.argv[1:]
    except ValueError:
        print('Wrong number of arguments.')
        print(__doc__)
        sys.exit()

    # Instantiate problem and solve
    csp = MinConflicts(get_graph(input_file))
    csp.solve()

    # Check solution
    if not verify_solution(csp.solution, csp.graph):
        raise ValueError('Solution %s is wrong' % csp.solution)

    # Print result to output file
    solution_string = '\n'.join(str(csp.solution[k]) for k in sorted(csp.graph)) \
                                                    if csp.solution else 'No answer\n'
    out_fd = open(output_file, 'w')
    out_fd.write(solution_string)
    out_fd.close()


if __name__ == '__main__':
    main()
    
