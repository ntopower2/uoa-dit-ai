"""CSP (Constraint Satisfaction Problems) problems and solvers. (Chapter 6)."""

import itertools
import random
import time
from collections import defaultdict

import search
from utils import argmin_random_tie, count, first


class CSP(search.Problem):
    """This class describes finite-domain Constraint Satisfaction Problems.
    A CSP is specified by the following inputs:
        variables   A list of variables; each is atomic (e.g. int or string).
        domains     A dict of {var:[possible_value, ...]} entries.
        neighbors   A dict of {var:[var,...]} that for each variable lists
                    the other variables that participate in constraints.
        constraints A function f(A, a, B, b) that returns true if neighbors
                    A, B satisfy the constraint when they have values A=a, B=b

    In the textbook and in most mathematical definitions, the
    constraints are specified as explicit pairs of allowable values,
    but the formulation here is easier to express and more compact for
    most cases. (For example, the n-Queens problem can be represented
    in O(n) space using this notation, instead of O(N^4) for the
    explicit representation.) In terms of describing the CSP as a
    problem, that's all there is.

    However, the class also supports data structures and methods that help you
    solve CSPs by calling a search function on the CSP. Methods and slots are
    as follows, where the argument 'a' represents an assignment, which is a
    dict of {var:val} entries:
        assign(var, val, a)     Assign a[var] = val; do other bookkeeping
        unassign(var, a)        Do del a[var], plus other bookkeeping
        nconflicts(var, val, a) Return the number of other variables that
                                conflict with var=val
        curr_domains[var]       Slot: remaining consistent values for var
                                Used by constraint propagation routines.
    The following methods are used only by graph_search and tree_search:
        actions(state)          Return a list of actions
        result(state, action)   Return a successor of state
        goal_test(state)        Return true if all constraints satisfied
    The following are just for debugging purposes:
        nassigns                Slot: tracks the number of assignments made
        display(a)              Print a human-readable representation
    """

    def __init__(self, variables, domains, neighbors, constraints):
        """Construct a CSP problem. If variables is empty, it becomes domains.keys()."""
        variables = variables or list(domains.keys())

        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.initial = ()
        self.curr_domains = None
        self.nassigns = 0

    def assign(self, var, val, assignment):
        """Add {var: val} to assignment; Discard the old value if any."""
        assignment[var] = val
        self.nassigns += 1

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            del assignment[var]

    def nconflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables."""

        # Subclasses may implement this more efficiently
        def conflict(var2):
            return (var2 in assignment and
                    not self.constraints(var, val, var2, assignment[var2]))

        return count(conflict(v) for v in self.neighbors[var])

    def display(self, assignment):
        """Show a human-readable representation of the CSP."""
        # Subclasses can print in a prettier way, or display with a GUI
        print('CSP:', self, 'with assignment:', assignment)

    # These methods are for the tree and graph-search interface:

    def actions(self, state):
        """Return a list of applicable actions: nonconflicting
        assignments to an unassigned variable."""
        if len(state) == len(self.variables):
            return []
        else:
            assignment = dict(state)
            var = first([v for v in self.variables if v not in assignment])
            return [(var, val) for val in self.domains[var]
                    if self.nconflicts(var, val, assignment) == 0]

    def result(self, state, action):
        """Perform an action and return the new state."""
        (var, val) = action
        return state + ((var, val),)

    def goal_test(self, state):
        """The goal is to assign all variables, with all constraints satisfied."""
        assignment = dict(state)
        return (len(assignment) == len(self.variables)
                and all(self.nconflicts(variables, assignment[variables], assignment) == 0
                        for variables in self.variables))

    # These are for constraint propagation

    def support_pruning(self):
        """Make sure we can prune values from domains. (We want to pay
        for this only if we use it.)"""
        if self.curr_domains is None:
            # changed behavior for auxiliary variables
            # if domain for variable V is a tuple then V is auxiliary
            # ergo V's domain is the cartesian product of its subdomains
            self.curr_domains = {}
            for v in self.variables:
                if not isinstance(self.domains[v], tuple):
                    self.curr_domains[v] = list(self.domains[v])
                else:
                    alldomains = list([list(domain) for domain in self.domains[v]])
                    self.curr_domains[v] = list(itertools.product(*alldomains))

    def suppose(self, var, value):
        """Start accumulating inferences from assuming var=value."""
        self.support_pruning()
        removals = [(var, a) for a in self.curr_domains[var] if a != value]
        self.curr_domains[var] = [value]
        return removals

    def prune(self, var, value, removals):
        """Rule out var=value."""
        # Do not prune if var is an auxiliary variable
        if isinstance(var, str) and var.startswith('h'): return None
        self.curr_domains[var].remove(value)
        if removals is not None:
            removals.append((var, value))

    def choices(self, var):
        """Return all values for var that aren't currently ruled out."""
        return (self.curr_domains or self.domains)[var]

    def infer_assignment(self):
        """Return the partial assignment implied by the current inferences."""
        self.support_pruning()
        return {v: self.curr_domains[v][0]
                for v in self.variables if 1 == len(self.curr_domains[v])}

    def restore(self, removals):
        """Undo a supposition and all inferences from it."""
        for B, b in removals:
            self.curr_domains[B].append(b)

    # This is for min_conflicts search

    def conflicted_vars(self, current):
        """Return a list of variables in current assignment that are in conflict"""
        return [var for var in self.variables
                if self.nconflicts(var, current[var], current) > 0]


# ______________________________________________________________________________
# Constraint Propagation with AC-3


def AC3(csp, queue=None, removals=None):
    """[Figure 6.3]"""
    if queue is None:
        queue = [(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]]
    csp.support_pruning()
    while queue:
        (Xi, Xj) = queue.pop()
        if revise(csp, Xi, Xj, removals):
            if not csp.curr_domains[Xi]:
                return False
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.append((Xk, Xi))
    return True


def revise(csp, Xi, Xj, removals):
    """Return true if we remove a value."""
    revised = False
    for x in csp.curr_domains[Xi][:]:
        # If Xi=x conflicts with Xj=y for every possible y, eliminate Xi=x
        if all(not csp.constraints(Xi, x, Xj, y) for y in csp.curr_domains[Xj]):
            csp.prune(Xi, x, removals)
            revised = True
    return revised


# ______________________________________________________________________________
# CSP Backtracking Search

# Variable ordering


def first_unassigned_variable(assignment, csp):
    """The default variable order."""
    return first([var for var in csp.variables if var not in assignment])


def mrv(assignment, csp):
    """Minimum-remaining-values heuristic."""
    return argmin_random_tie(
        [v for v in csp.variables if v not in assignment],
        key=lambda var: num_legal_values(csp, var, assignment))


def num_legal_values(csp, var, assignment):
    if csp.curr_domains:
        return len(csp.curr_domains[var])
    else:
        return count(csp.nconflicts(var, val, assignment) == 0
                     for val in csp.domains[var])


# Value ordering


def unordered_domain_values(var, assignment, csp):
    """The default value order."""
    return csp.choices(var)


def lcv(var, assignment, csp):
    """Least-constraining-values heuristic."""
    return sorted(csp.choices(var),
                  key=lambda val: csp.nconflicts(var, val, assignment))


# Inference


def no_inference(csp, var, value, assignment, removals):
    return True


def forward_checking(csp, var, value, assignment, removals):
    """Prune neighbor values inconsistent with var=value."""
    csp.support_pruning()
    for B in csp.neighbors[var]:
        if B not in assignment:
            for b in csp.curr_domains[B][:]:
                if not csp.constraints(var, value, B, b):
                    csp.prune(B, b, removals)
            if not csp.curr_domains[B]:
                return False
    return True


def mac(csp, var, value, assignment, removals):
    """Maintain arc consistency."""
    return AC3(csp, [(X, var) for X in csp.neighbors[var]], removals)


# The search, proper


def backtracking_search(csp,
                        select_unassigned_variable=first_unassigned_variable,
                        order_domain_values=unordered_domain_values,
                        inference=no_inference,
                        timeout=40):
    """[Figure 6.5]"""
    tic = time.time()

    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            return assignment
        # return None if time exceeds the timeout limit
        if time.time() - tic > timeout: return None
        var = select_unassigned_variable(assignment, csp)
        for value in order_domain_values(var, assignment, csp):
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                removals = csp.suppose(var, value)
                if inference(csp, var, value, assignment, removals):
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                csp.restore(removals)
        csp.unassign(var, assignment)
        return None

    result = backtrack({})
    assert result is None or csp.goal_test(result)
    return result


# ______________________________________________________________________________
# Min-conflicts hillclimbing search for CSPs


def min_conflicts(csp, max_steps=100000):
    """Solve a CSP by stochastic hillclimbing on the number of conflicts."""
    # Generate a complete assignment for all variables (probably with conflicts)
    csp.current = current = {}
    for var in csp.variables:
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    # Now repeatedly choose a random conflicted variable and change it
    for i in range(max_steps):
        conflicted = csp.conflicted_vars(current)
        if not conflicted:
            return current
        var = random.choice(conflicted)
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    return None


def min_conflicts_value(csp, var, current):
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose at random."""
    return argmin_random_tie(csp.domains[var],
                             key=lambda val: csp.nconflicts(var, val, current))


# ______________________________________________________________________________


def tree_csp_solver(csp):
    """[Figure 6.11]"""
    assignment = {}
    root = csp.variables[0]
    X, parent = topological_sort(csp, root)

    csp.support_pruning()
    for Xj in reversed(X[1:]):
        if not make_arc_consistent(parent[Xj], Xj, csp):
            return None

    assignment[root] = csp.curr_domains[root][0]
    for Xi in X[1:]:
        assignment[Xi] = assign_value(parent[Xi], Xi, csp, assignment)
        if not assignment[Xi]:
            return None
    return assignment


def topological_sort(X, root):
    """Returns the topological sort of X starting from the root.

    Input:
    X is a list with the nodes of the graph
    N is the dictionary with the neighbors of each node
    root denotes the root of the graph.

    Output:
    stack is a list with the nodes topologically sorted
    parents is a dictionary pointing to each node's parent

    Other:
    visited shows the state (visited - not visited) of nodes

    """
    neighbors = X.neighbors

    visited = defaultdict(lambda: False)

    stack = []
    parents = {}

    build_topological(root, None, neighbors, visited, stack, parents)
    return stack, parents


def build_topological(node, parent, neighbors, visited, stack, parents):
    """Build the topological sort and the parents of each node in the graph."""
    visited[node] = True

    for n in neighbors[node]:
        if (not visited[n]):
            build_topological(n, node, neighbors, visited, stack, parents)

    parents[node] = parent
    stack.insert(0, node)


def make_arc_consistent(Xj, Xk, csp):
    """Make arc between parent (Xj) and child (Xk) consistent under the csp's constraints,
    by removing the possible values of Xj that cause inconsistencies."""
    # csp.curr_domains[Xj] = []
    for val1 in csp.domains[Xj]:
        keep = False  # Keep or remove val1
        for val2 in csp.domains[Xk]:
            if csp.constraints(Xj, val1, Xk, val2):
                # Found a consistent assignment for val1, keep it
                keep = True
                break

        if not keep:
            # Remove val1
            csp.prune(Xj, val1, None)

    return csp.curr_domains[Xj]


def assign_value(Xj, Xk, csp, assignment):
    """Assign a value to Xk given Xj's (Xk's parent) assignment.
    Return the first value that satisfies the constraints."""
    parent_assignment = assignment[Xj]
    for val in csp.curr_domains[Xk]:
        if csp.constraints(Xj, parent_assignment, Xk, val):
            return val

    # No consistent assignment available
    return None


# ______________________________________________________________________________
# Map-Coloring Problems


class UniversalDict:
    """A universal dict maps any key to the same value. We use it here
    as the domains dict for CSPs in which all variables have the same domain.
    >>> d = UniversalDict(42)
    >>> d['life']
    42
    """

    def __init__(self, value): self.value = value

    def __getitem__(self, key): return self.value

    def __repr__(self): return '{{Any: {0!r}}}'.format(self.value)


def different_values_constraint(A, a, B, b):
    """A constraint saying two neighboring variables must differ in value."""
    return a != b


def MapColoringCSP(colors, neighbors):
    """Make a CSP for the problem of coloring a map with different colors
    for any two adjacent regions. Arguments are a list of colors, and a
    dict of {region: [neighbor,...]} entries. This dict may also be
    specified as a string of the form defined by parse_neighbors."""
    if isinstance(neighbors, str):
        neighbors = parse_neighbors(neighbors)
    return CSP(list(neighbors.keys()), UniversalDict(colors), neighbors,
               different_values_constraint)


def parse_neighbors(neighbors, variables=None):
    """Convert a string of the form 'X: Y Z; Y: Z' into a dict mapping
    regions to neighbors. The syntax is a region name followed by a ':'
    followed by zero or more region names, followed by ';', repeated for
    each region name. If you say 'X: Y' you don't need 'Y: X'.
    >>> parse_neighbors('X: Y Z; Y: Z') == {'Y': ['X', 'Z'], 'X': ['Y', 'Z'], 'Z': ['X', 'Y']}
    True
    """
    dic = defaultdict(list)
    specs = [spec.split(':') for spec in neighbors.split(';')]
    for (A, Aneighbors) in specs:
        A = A.strip()
        for B in Aneighbors.split():
            dic[A].append(B)
            dic[B].append(A)
    return dic


# ______________________________________________________________________________
# The Kenken Puzzle

easy0 = (3, ['D 00 01 2', '02 3', 'S 10 20 2', 'S 11 12 1', 'M 21 22 6'])
easy1 = (4, ['A 00 01 4', 'A 02 12 22 6', 'S 03 13 1', 'S 10 20 3', 'D 11 21 2', 'S 30 31 1', 'M 23 33 32 8'])
given = (6, ['A 00 10 11', 'D 01 02 2', 'M 03 13 20', 'M 04 05 15 25 6', 'S 11 12 3', 'D 14 24 3', 'M 20 21 30 31 240',
             'M 22 23 6', 'M 32 42 6', 'A 33 43 44 7', 'M 34 35 30', 'M 40 41 6', 'A 45 55 9', 'A 50 51 52 8',
             'D 53 54 2'])
hard0 = (7, ['A 00 10 5', 'M 01 02 03 60', 'S 04 14 2', 'A 05 06 16 18', 'S 11 21 4', 'D 12 13 2', '15 6',
             'M 20 30 31 120', 'S 22 32 1', 'S 23 33 6', 'S 24 25 1', '26 3', 'A 34 44 43 11', 'D 35 36 3', 'D 40 41 2',
             'S 42 52 2', 'S 45 46 2', 'M 50 51 60 61 630', 'S 53 54 1', 'A 55 56 5', '62 4', 'S 63 64 1', 'D 65 66 2'])
hard1 = (7, ['M 00 10 12', 'D 01 02 3', 'A 03 04 13 14 18', 'S 05 06 4', 'M 11 12 7', 'D 15 16 2', 'S 20 30 2',
             'D 21 31 3', 'A 22 23 7', 'A 24 25 34 35 44 24', 'M 26 36 46 28', 'A 32 42 9', 'D 33 43 3', 'D 40 50 2',
             'S 41 51 3', 'S 45 55 2', 'M 52 62 63 35', 'M 53 54 4', 'A 56 66 8', 'S 60 61 2', 'S 64 65 4'])
fourH1 = (4, ['D 00 10 2', 'M 01 02 12', 'S 03 13 1', 'A 11 12 5', 'A 20 21 5', 'M 22 23 4', 'S 30 31 1', 'D 32 33 2'])


class KenKen(CSP):
    """A class for KenKen games. Input = (N, constraints)
    N = dimension of the Kenken grid [integer]
    constraints = vector [c1, c2, c3...] where ci is a constraint string
    "{A/S/M/D} num1 num2 num3 ... numN result" or "num1 result"
    """

    def __init__(self, game):
        """Build a KenKen game of size n and additional cell constraints"""
        n, gconstraints = game[0], game[1]
        self._RN = list(range(n))
        self._GRID = ['%d%d' % (i, j) for i in self._RN for j in self._RN]
        self.variables = self._GRID
        self.domains = {var: ''.join(str(i) for i in range(1, n + 1)) for var in self.variables}
        self.funcdata, self.inpcnstr = {}, ""
        self.checks = 0
        self.neighbors = """"""
        for var1 in self.variables:
            self.neighbors += var1 + ': '
            for var2 in self.variables:
                if var1 != var2 and (var1[0] == var2[0] or var1[1] == var2[1]):
                    self.neighbors += var2 + ' '
            self.neighbors += ';'
        # the number of auxiliary variables
        auxvar = 0

        # transform constraints into funcdata elements
        # in order to be used from kenken constraint function
        for constraint in gconstraints:
            splitted = constraint.split(' ')
            nums = splitted[1:-1]
            result = splitted[-1]
            operation = splitted[0]
            self.inpcnstr += str("{0:2d}".format(gconstraints.index(constraint) + 1)) + ". |"
            # if constraint starts with digits, it's a unary constraint
            if operation.isdigit():
                # change variable's domain into result
                self.domains[operation[0] + operation[1]] = result
                # display configuration
                self.inpcnstr += 'x' + chr(8320 + int(operation[0])) + chr(8320 + int(operation[1]))
            # if two numbers in constraint, then it's binary
            elif len(nums) == 2:
                # add expected result and corresponding operation
                # with each variable as key to funcdata dictionary
                self.funcdata[nums[0]] = (operation, result, nums[1])
                self.funcdata[nums[1]] = (operation, result, nums[0])
                self.neighbors += nums[0] + ": " + nums[1] + ";"
                # display configuration
                self.inpcnstr += 'x' + chr(8320 + int(nums[0][0])) + chr(8320 + int(nums[0][1]))
                if operation == 'A':
                    self.inpcnstr += " + "
                elif operation == 'S':
                    self.inpcnstr += " - "
                elif operation == 'M':
                    self.inpcnstr += " * "
                elif operation == 'D':
                    self.inpcnstr += " / "
                else:
                    print('Unknown symbol ', operation)
                self.inpcnstr += 'x' + chr(8320 + int(nums[1][0])) + chr(8320 + int(nums[1][1]))
            # conversion to binary constraint is needed
            else:
                newvar = 'h' + str(auxvar)
                auxvar += 1
                self.variables.append(newvar)
                self.domains[newvar] = tuple([self.domains[num] for num in nums[:-1]])
                self.neighbors += newvar + ": " + " ".join(nums) + ";"
                self.funcdata[newvar] = tuple([num for num in nums[:-1]] + [operation, result])
                # display configuration
                if operation == 'A':
                    symbol = " + "
                elif operation == 'M':
                    symbol = " * "
                else:
                    print('Unknown symbol ', operation)
                for num in nums:
                    self.inpcnstr += 'x' + chr(8320 + int(num[0])) + chr(8320 + int(num[1]))
                    self.inpcnstr += symbol
                self.inpcnstr = self.inpcnstr[:-2]

            self.inpcnstr += ' = ' + result + '\n'

        def kenken_constraint(A, a, B, b):
            self.checks += 1
            # A and B are non auxiliary variables with values a and b respectively
            if A.isdigit() and B.isdigit():
                num1, num2 = int(a), int(b)
                constraint = self.funcdata.get(A)
                # constraint for variable A does not include variable B
                if constraint is None or constraint[2] != B:
                    # check if there are equal values in same row or column
                    if num1 == num2 and (A[0] == B[0] or A[1] == B[1]):
                        return False
                    else:
                        return True
                operation, strresult, num = constraint
                result = int(strresult)
                # check if constraint operation gives proper result
                if operation == 'A':
                    return (num1 + num2 == result)
                elif operation == 'M':
                    return (num1 * num2 == result)
                elif operation == 'S':
                    return (abs(num1 - num2) == result)
                elif operation == 'D':
                    return (max(num1 / num2, num2 / num1) == result)
                else:
                    print('ERROR ', self.funcdata[A], self.funcdata[B])
            # only A is an auxiliary variable
            elif not A.isdigit():
                num = int(b)
                auxdata = self.funcdata[A]
                result = int(auxdata[-1])
                auxvars = auxdata[:-2]
                if B in auxvars:
                    return (a[auxvars.index(B)] == b)
                elif auxdata[-2] == 'M':
                    product = 1
                    for val in a:
                        product *= int(val)
                    return (product * num == result)
                elif auxdata[-2] == 'A':
                    summation = 0
                    for val in a:
                        summation += int(val)
                    return (summation + num == result)
                else:
                    print('ERROR ', self.funcdata[A], self.funcdata[B])
            # only B is an auxiliary variable
            elif not B.isdigit():
                num = int(a)
                auxdata = self.funcdata[B]
                result = int(auxdata[-1])
                auxvars = auxdata[:-2]
                if A in auxvars:
                    return (b[auxvars.index(A)] == a)
                elif auxdata[-2] == 'M':
                    product = 1
                    for val in b:
                        product *= int(val)
                    return (product * num == result)
                elif auxdata[-2] == 'A':
                    summation = 0
                    for val in b:
                        summation += int(val)
                    return (summation + num == result)
                else:
                    print('ERROR ', self.funcdata[A], self.funcdata[B])
            else:
                print('Unknown case ', self.funcdata[A], self.funcdata[B])

        neighbors = parse_neighbors(self.neighbors[:-1])
        CSP.__init__(self, self.variables, self.domains, neighbors, kenken_constraint)

    def display(self, assignment):
        n = len(self._RN)

        def show_grid():
            line = '\n    == KenKen CSP %dx%d ==\n\n         ' % (n, n)
            line += '|'.join(str(i) for i in range(n))
            line += '\n'
            for row in range(n):
                temp = str(row) + '|'
                for col in range(n):
                    element = assignment.get(str(row) + str(col), ' ')
                    if len(element) > 1 or element == " ":
                        temp += '|.'
                    else:
                        temp += "|" + element
                line += '      ' + temp + '\n'
            # line += ((8+2*(n-1))*'-')+'\n'
            print(line)

        show_grid()
        print("     CONSTRAINTS \n" + self.inpcnstr)
