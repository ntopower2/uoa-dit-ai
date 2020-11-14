# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    current = problem.getStartState()
    fringe, explored, path = util.Stack(), set(), {current: [None, None]}
    fringe.push(current)
    while not fringe.isEmpty():
        current = fringe.pop()
        if problem.isGoalState(current):
            state = current
            # get current's parent and action required from dict
            actions, parent = [path[state][0]], path[state][1]
            # parent is None only at the starting state
            while parent is not None and path[parent][0] is not None:
                # append each action required and get an upper parent
                actions.append(path[parent][0])
                parent = path[parent][1]
            # went backwards so actions should be parsed in reverse
            actions.reverse()
            return actions
        if current not in explored:
            explored.add(current)
            nearStates = problem.getSuccessors(current)
            for state in nearStates:
                child, direction, cost = state
                fringenodes = [node[0] for node in fringe.list]
                if child not in explored and child not in fringenodes:
                    fringe.push(child)
                    path[child] = [direction, current]
    util.raiseNotDefined()
    return False


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    current = problem.getStartState()
    fringe, explored, path = util.Queue(), set(), []
    if problem.isGoalState(current): return path
    fringe.push((current, path))
    while not fringe.isEmpty():
        (current, path) = fringe.pop()
        if current not in explored:
            explored.add(current)
            nearStates = problem.getSuccessors(current)
            for state in nearStates:
                child, direction, cost = state
                fringenodes = [node[0] for node in fringe.list]
                if child not in explored and child not in fringenodes:
                    if problem.isGoalState(child): return path + [direction]
                    fringe.push((child, path + [direction]))
    util.raiseNotDefined()
    return False


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    current, cost = problem.getStartState(), 0
    costfunction = lambda state: path[state][2] + cost
    fringe, explored, path = util.PriorityQueueWithFunction(costfunction), set(), {current: [None, None, 0]}
    fringe.push(current)
    while not fringe.isEmpty():
        current = fringe.pop()
        # check if current is goal
        if problem.isGoalState(current):
            state = current
            # get current's parent and action required from dict
            actions, parent = [path[state][0]], path[state][1]
            # parent is None only at the starting state
            while parent is not None and path[parent][0] is not None:
                # append each action required and get an upper parent
                actions.append(path[parent][0])
                parent = path[parent][1]
            # went backwards so actions should be parsed in reverse
            actions.reverse()
            return actions
        if current not in explored:
            explored.add(current)
            nearStates = problem.getSuccessors(current)
            for state in nearStates:
                child, direction, cost = state
                fringenodes = [node[2] for node in fringe.heap]
                if child not in explored and child not in fringenodes:
                    # add child to dict with action required, parent and its path cost
                    path[child] = [direction, current, costfunction(current)]
                    # add child in priority queue with priority its path cost
                    fringe.push(child)
                elif child in fringenodes:
                    # child exists in fringe, another path is found with cost newcost
                    # newcost is equal to the parent's cost plus the cost of the new action
                    newcost = costfunction(current)
                    # checks if child's cost in dict is greater than the new
                    if path[child][2] > newcost:
                        # replace the child's cost with the better newcost in dict and fringe
                        path[child] = [direction, current, newcost]
                        fringe.update(child, newcost)
    util.raiseNotDefined()
    return False


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    current, cost = problem.getStartState(), 0
    costfunction = lambda state: path[state][2] + cost
    priorityfunction = lambda state: path[state][2] + heuristic(state, problem)
    fringe, explored, path = util.PriorityQueueWithFunction(priorityfunction), set(), {current: [None, None, 0]}
    fringe.push(current)
    while not fringe.isEmpty():
        current = fringe.pop()
        # check if current is goal
        if problem.isGoalState(current):
            state = current
            # get current's parent and action required from dict
            actions, parent = [path[state][0]], path[state][1]
            # parent is None only at the starting state
            while parent is not None and path[parent][0] is not None:
                # append each action required and get an upper parent
                actions.append(path[parent][0])
                parent = path[parent][1]
            # went backwards so actions should be parsed in reverse
            actions.reverse()
            return actions
        if current not in explored:
            explored.add(current)
            nearStates = problem.getSuccessors(current)
            for state in nearStates:
                child, direction, cost = state
                fringenodes = [node[2] for node in fringe.heap]
                if child not in explored and child not in fringenodes:
                    # add child to dict with action required, parent and its path cost
                    path[child] = [direction, current, costfunction(current)]
                    # add child in priority queue with priority its path cost plus heuristic value for child
                    fringe.push(child)
                elif child in fringenodes:
                    # child exists in fringe, another path is found with cost newcost
                    # newcost is equal to the parent's cost plus the cost of the new action
                    # plus the value of heuristic for child
                    newcost = costfunction(current) + heuristic(child, problem)
                    # checks if child's current priority is greater than new priority
                    if priorityfunction(child) > newcost:
                        # replace child's path cost with new path cost in dict
                        path[child] = [direction, current, costfunction(current)]
                        # replace child's priority with new priority
                        fringe.update(child, newcost)
    util.raiseNotDefined()
    return False


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
