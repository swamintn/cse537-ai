#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementations of CSP Solvers using
    1) DFSB
    2) DFSB with Variable+Value Ordering and Arc Consistency

Algorithms:
1) Plain_DFSB
2) Powerful_DFSB
Both algorithms are classes that must be initialized with an input
graph. Calling solve() on the initialized object will generate the
solution. The solution will be a dict with assignments to all variables.
It will be None if no solution is possible.

Execution:
    dfsb.py <INPUT_FILE_PATH> <OUTPUT_FILE_PATH> <#Algorithm>
    where,
        Plain_DFSB is used if #Algorithm is 0, Powerful_DFSB if #Algorithm is 1

@author: Swaminathan Sivaraman
"""
from   copy      import deepcopy
import itertools
import sys
import time

MAX_SECONDS = 60

class Plain_DFSB:
    """
    CSP Solver using Plain Depth-First Search with Backtracking
    """
    def __init__(self, graph):
        self.graph     = graph
        self.variables = list(self.graph.keys())
        self.solution  = None
        
        # Performance details
        self.time_taken  = 0.0
        self.steps_taken = 0
        self.timed_out   = False

    def solve(self):
        self.start_time = time.time()
        self.solution = self.do_recursive_search(0, {})
        return self.solution

    def do_recursive_search(self, level, cur_solution):
        """
        Depth-First search of given graph. It assigns possible colors
        for the variable on this level based on the values already assigned
        on previous levels(obtained using cur_solution), adds that assignment
        to cur_solution and sends it downwards to the next level for all
        assignments.

        On the last level, if solution is found, returns cur_solution. The first
        occurrence of a valid solution will be propogated back as the solution
        as soon as it is found. If no solution is found on the last level, None is
        returned
        """
        if level == len(self.variables):
            return cur_solution

        self.time_taken = time.time() - self.start_time
        self.steps_taken += 1
        if self.time_taken > MAX_SECONDS:
            self.timed_out = True
            return None

        node      = self.variables[level]
        node_data = self.graph[node]
        
        # Prune color list based on assignments so far
        allowed_colors = node_data['COLORS'][:]
        for dependent_node in node_data['NODES']:
            try:    allowed_colors.remove(cur_solution[dependent_node])
            except: pass

        # Assign all possible colors and send it downwards
        for color in allowed_colors:
            possible_solution = {**cur_solution, **{node: color}}
            full_solution = self.do_recursive_search(level+1, possible_solution)
            if full_solution:
                return full_solution
            if self.timed_out:
                return None

        # No result found
        return None


class Powerful_DFSB:
    """
    CSP Solver using Depth-First Search with Backtracking and 
    some extra optimizations -
    1) Variable and Value Ordering
    2) Arc Consistency
    """
    def __init__(self, graph):
        self.graph     = graph
        self.variables = list(self.graph.keys())
        self.solution  = None
        
        # Performance details
        self.time_taken  = 0.0
        self.steps_taken = 0
        self.timed_out   = False

    def solve(self):
        self.start_time = time.time()
        self.solution = self.do_recursive_search(0, {})
        return self.solution    

    def get_best_variable(self, observed_vars=[]):
        """
        Variable-Ordering Heuristic

        From the list of unobserved vars, it chooses the variable with the least amount
        of values. To break ties, it chooses variables with more connected nodes.
        """
        unobserved_vars = [(k, v) for (k, v) in self.graph.items() if k not in observed_vars]
        return sorted(unobserved_vars, key=lambda s: (len(s[1]['COLORS']), 
                                                     -len(s[1]['NODES'])))[0][0]

    def get_best_color_order(self, node, colors, dep_nodes, observed_vars=[]):
        """
        Value-Ordering Heuristic

        For the given colors and node, it sorts the colors in the order of colors that 
        least restrict the node's dependent node colors.
        """
        color_wts = [0] * len(colors)
        for i, color in enumerate(colors):
            for dep_node in set(dep_nodes):
                if dep_node not in observed_vars and color in self.graph[dep_node]['COLORS']:
                    color_wts[i] += 1
        return [x for (y, x) in sorted(zip(color_wts, colors))]
        
    def do_forward_checking(self, node, color, observed_vars=[]):
        """
        Forward Checking (Not used)

        Given a node and color, removes that color from the dependent 
        nodes' domains. If any dependent node domain reaches 0, this 
        method returns False, else True.
        """
        for dep_node in set(self.graph[node]['NODES']):
            if dep_node not in observed_vars:
                try: self.graph[dep_node]['COLORS'].remove(color)
                except ValueError: pass
                if len(self.graph[dep_node]['COLORS']) == 0:
                    return False
        return True 

    def build_arcs(self, observed_vars=[]):
        """
        Return a dict of valid arcs for each node-pair in the graph,
        with arcs lists keyed by heads of the arcs.

        No arc is built if the tail of an arc is in observed_vars
        """
        arcs_by_head = {}
        for tail, tail_data in self.graph.items():
            for head in set(tail_data['NODES']):
                if tail not in observed_vars:
                    arcs_by_head.setdefault(head, []).append((tail, head))
        return arcs_by_head

    def do_arc_consistency(self, observed_vars=[]):
        """
        Arc Consistency - Uses the standard AC3 algorithm

        Returns False if any of the domain's colors become zero,
        else True
        """
        # Get all arcs
        arcs_by_head = self.build_arcs(observed_vars)
        arcs = set(itertools.chain(*arcs_by_head.values()))

        # Keep doing consistency checks until we run out of arcs
        while len(arcs) != 0:
            arc = arcs.pop()
            tail, head = arc
            head_colors = set(self.graph[head]['COLORS'])
            old_tail_colors = self.graph[tail]['COLORS']
            new_tail_colors = []
            for color in old_tail_colors:
                if len(head_colors - {color}) != 0:
                    new_tail_colors.append(color)
            if len(new_tail_colors) == 0:
                return False
            if new_tail_colors != old_tail_colors:
                self.graph[tail]['COLORS'] = new_tail_colors
                arcs = arcs.union(arcs_by_head.get(tail, []))
        return True


    def do_recursive_search(self, level, cur_solution):
        """
        Depth-First search of given graph. It assigns possible colors
        for the variable on this level based on the values already assigned
        on previous levels(obtained using cur_solution), adds that assignment
        to cur_solution and sends it downwards to the next level for all
        assignments.

        On the last level, if solution is found, returns cur_solution. The first
        occurrence of a valid solution will be propogated back of the solution
        as soon as it is found. If no solution is found on the last level, None is
        returned
        """
        if level == len(self.variables):
            return cur_solution

        self.time_taken = time.time() - self.start_time
        self.steps_taken += 1
        if self.time_taken > MAX_SECONDS:
            self.timed_out = True
            return None

        # Variable Ordering Heuristic
        node      = self.get_best_variable(cur_solution.keys())
        node_data = self.graph[node]

        # Value Ordering Heuristic
        orig_colors    = node_data['COLORS'][:]
        dep_nodes      = node_data['NODES']
        allowed_colors = self.get_best_color_order(node, orig_colors, dep_nodes, cur_solution.keys())

        # Assign all possible colors and send it downwards
        for color in allowed_colors:
            # Save the current state of the graph to restore it on failure
            old_graph_state = deepcopy(self.graph)

            # Do Arc Consistency for this color assignment
            self.graph[node]['COLORS'] = [color]
            is_valid_graph = self.do_arc_consistency(list(cur_solution.keys()) + [node])
            self.steps_taken += 1
            
            if is_valid_graph:
                possible_solution = {**cur_solution, **{node: color}}
                full_solution = self.do_recursive_search(level+1, possible_solution)
                if full_solution != None:
                    return full_solution

            # Restore state for next value assignment
            self.graph = old_graph_state

            if self.timed_out:
                return None

        # No result found
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
        input_file, output_file, mode = sys.argv[1:]
    except ValueError:
        print('Wrong number of arguments.')
        print(__doc__)

    # Instantiate problem and solve
    CSP_ALGO = Plain_DFSB if mode == '0' else Powerful_DFSB
    csp = CSP_ALGO(get_graph(input_file))
    csp.solve()

    # Check solution
    if not verify_solution(csp.solution, csp.graph):
        raise ValueError('Solution %s is wrong, %s conflicts' % (csp.solution, conflicts))

    # Print result to output file
    solution_string = '\n'.join(str(csp.solution[k]) for k in sorted(csp.graph)) \
                                                    if csp.solution else 'No answer\n'
    out_fd = open(output_file, 'w')
    out_fd.write(solution_string)
    out_fd.close()


if __name__ == '__main__':
    main()
