# multiAgents.py
# --------------
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


import random

import util
from game import Agent
from util import manhattanDistance


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        # print([(score,action) for score,action in zip(scores,legalMoves)])
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # TODO utilize scared times more
        # TODO create script for meta-optimization using coeff dict
        food = newFood.asList()
        # gets all ghost pos for the successor
        newGhostPos = successorGameState.getGhostPositions()
        # calculates manhattan for every ghost and every food point
        distancesFood = [manhattanDistance(newPos, point) for point in food]
        distancesGhosts = [manhattanDistance(newPos, ghostPos) for ghostPos in newGhostPos]
        # coefficient dictionary where cf: closestFood, cg: closestGhost, lenf: len(food)
        coeff = {'cf': -2, 'cfPunish': 0, 'cg': -6, 'cgPunish': -700, 'lenf': -50}
        # chooses the minimum dist and calculates the appropriate value for ghost and food (if found)
        # sets 0 if food list is empty and punishes if a ghost catches pacman with a coefficient
        closestFood = coeff['cf'] * min(distancesFood) if len(distancesFood) != 0 else coeff['cfPunish']

        if len(distancesGhosts) != 0 and min(distancesGhosts) != 0:
            closestGhost = coeff['cg'] / min(distancesGhosts)
        else:
            closestGhost = coeff['cgPunish']
        # deduces if ghost(s) is(are) scared and sets ghost parameter 0
        if newScaredTimes.count(0) == 0:
            scared = min([times for times in newScaredTimes if times > 0])
            closestGhost = 0
        # returns the appropriate values from above and punishes with another coefficient
        # depending on the number of food points that remain unvisited
        return closestFood + closestGhost + coeff['lenf'] * len(food)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        agentsNum = gameState.getNumAgents()

        def minimax(state, agentIndex, depth):
            # every node (all agents) gets its minimax value with this function
            if depth < self.depth:
                # if minimax is not in terminal depth
                value = float('inf') if agentIndex > 0 else -float('inf')
                # initialize value depending on agentIndex
                # agent0-MAX gets -infinity
                # agentX-MIN gets +infinity
                for action in state.getLegalActions(agentIndex):
                    # examine every feasible action based on agentIndex
                    if agentIndex == agentsNum - 1:
                        # if current agent is the last ghost
                        value = min(value, minimax(state.generateSuccessor(agentIndex, action), 0, depth + 1))
                        # start a deeper ply with agent0
                    elif agentIndex > 0:
                        # if current agent is another typical ghost
                        value = min(value, minimax(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth))
                        # calculate minimax value based on the next ghost in the same depth
                    else:
                        # current agent is agent0
                        value = max(value, minimax(state.generateSuccessor(agentIndex, action), 1, depth))
                        # calculate minimax for max agent based on the agent1 (first ghost) in the same depth
                if abs(value) == float('inf'):
                    # if no move is possible, then replace current minimax value with score
                    return self.evaluationFunction(state)
                else:
                    # minimax value is not inf, so current value is correct minimax
                    return value
            else:
                # depth == target depth so minimax value is the score of the current state
                return self.evaluationFunction(state)

        best = [float('-inf'), 'LOUIS FRIEND']
        # placeholder values for comparison with possible minimax values for agent0
        for action in gameState.getLegalActions(0):
            # examine every possible action for agent0 in gameState
            best = max(best, [minimax(gameState.generateSuccessor(0, action), 1, 0), action])
            # get the maximum value of minimax values produced by agent1 in depth 0
        # best matrix contains the maximum minimax value for agent0 and the corresponding action
        # so it returns the second element of the matrix
        return best[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        agentsNum = gameState.getNumAgents()

        def minimaxAB(state, agentIndex, depth, a=-float('inf'), b=float('inf')):
            value = float('inf') if agentIndex > 0 else -float('inf')
            # initialize value depending on agentIndex
            # agent0-MAX gets -infinity
            # agentX-MIN gets +infinity
            best = [value, "LOUIS FRIEND"]
            # placeholder values for comparison with possible minimax values for agent0
            for action in state.getLegalActions(agentIndex):
                # examine every feasible action based on agentIndex
                if agentIndex == agentsNum - 1:
                    # if current agent is the last ghost
                    agentIndex = -1
                    depth += 1
                    # increase depth and set agent -1 to work with agentIndex+1
                if depth == self.depth:
                    # max depth reached, evaluate state with score
                    value = self.evaluationFunction(state.generateSuccessor(agentIndex, action))
                else:
                    # run minimaxAB for the next agent in proper depth (increased only if agent is last)
                    value = minimaxAB(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth, a, b)
                # when value is a list, parse only the first value
                if isinstance(value, list):
                    value = value[0]
                if agentIndex == 0:
                    # max agent behaviour
                    best = max(best, [value, action])
                    if value > b:
                        # actual pruning if applicable
                        return [value, action]
                    # update a value
                    a = max(a, value)
                else:
                    # min agent behaviour
                    best = min(best, [value, action])
                    if value < a:
                        # actual pruning if applicable
                        return [value, action]
                    # update b value
                    b = min(b, value)
            if abs(best[0]) == float('inf'):
                # if no move is possible, then return score
                return self.evaluationFunction(state)
            # return the pruned minimax value
            return best

        # calculate the pair value, action for agent0 in depth 0 while being in gameState
        best = minimaxAB(gameState, 0, 0)
        return best[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        agentsNum = gameState.getNumAgents()

        def expectimax(state, agentIndex, depth):
            # every node (all agents) gets its minimax value with this function
            if depth < self.depth:
                # if minimax is not in terminal depth
                value = 0 if agentIndex > 0 else -float('inf')
                # initialize value depending on agentIndex
                # agent0-MAX gets -infinity
                # agentX-MIN gets 0 - neutral element for addition
                actions = state.getLegalActions(agentIndex)
                for action in actions:
                    # examine every feasible action based on agentIndex
                    if agentIndex == agentsNum - 1:
                        # if current agent is the last ghost
                        value += expectimax(state.generateSuccessor(agentIndex, action), 0, depth + 1)
                        # start a deeper ply with agent0 and add resulting values to value
                    elif agentIndex > 0:
                        # if current agent is another typical ghost
                        value += expectimax(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth)
                        # calculate each possible value based on the next ghost in the same depth and add to value
                    else:
                        # current agent is agent0
                        value = max(value, expectimax(state.generateSuccessor(agentIndex, action), 1, depth))
                        # calculate minimax for max agent based on the agent1 (first ghost) in the same depth
                if len(actions) == 0:
                    # if no move is possible, then replace current minimax value with score
                    return self.evaluationFunction(state)
                elif agentIndex > 0:
                    # current agent is ghost so take the average of expectimax values
                    return value / len(actions)
                else:
                    # current agent is pacman so get the max agent value
                    return value
            else:
                # depth == target depth so minimax value is the score of the current state
                return self.evaluationFunction(state)

        best = [float('-inf'), 'LOUIS FRIEND']
        # placeholder values for comparison with possible minimax values for agent0
        for action in gameState.getLegalActions(0):
            # examine every possible action for agent0 in gameState
            best = max(best, [expectimax(gameState.generateSuccessor(0, action), 1, 0), action])
            # get the maximum value of minimax values produced by agent1 in depth 0
        # best matrix contains the maximum minimax value for agent0 and the corresponding action
        # so it returns the second element of the matrix
        return best[1]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: I used a similar coefficient-based evaluation model from q1
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    value = currentGameState.getScore()
    food = newFood.asList()
    # gets all ghost pos for the successor
    newGhostPos = currentGameState.getGhostPositions()
    # calculates manhattan for every ghost and every food point
    distancesFood = [manhattanDistance(newPos, point) for point in food]
    distancesGhosts = [manhattanDistance(newPos, ghostPos) for ghostPos in newGhostPos]
    # coefficient dictionary where cg: closestGhost, lenf: len(food)
    coeff = {'cfReward': 10e5, 'cg': 1, 'cgPunish': -900, 'lenf': -150}
    # calculates the appropriate value for ghost and food (if found)
    # rewards if food list is empty and punishes if a ghost catches pacman with a coefficient
    closestFood = 1.0 / sum(distancesFood) if len(distancesFood) != 0 else coeff['cfReward']
    closestGhost = sum([1 / (1 + coeff['cg'] * dist) for dist in distancesGhosts]) if len(distancesGhosts) != 0 else \
        coeff['cgPunish']
    remainingFood = 1 / (1 + coeff['lenf'] * len(food))

    value += closestFood + closestGhost + remainingFood
    return value


# Abbreviation
better = betterEvaluationFunction
