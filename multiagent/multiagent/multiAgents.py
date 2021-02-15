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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0
        foodDist = []
        foodNum = 0
        if childGameState.isWin():
            score += 1000000
        elif childGameState.isLose():
            score -= 1000000
        if childGameState.getNumFood() < currentGameState.getNumFood():
            score += 10000

        nfl = newFood.asList()
        for i in range(0, len(nfl)):
            foodDist.append(manhattanDistance(newPos, nfl[i]))

        if foodDist:
            score -= (100) * min(foodDist)

        for i in range(0, len(newGhostStates)):
            if newScaredTimes[i] > 0:
                score += (75) * newScaredTimes[i] / (1+manhattanDistance(newGhostStates[i].getPosition(), newPos))
            else:
                score += (25) * manhattanDistance(newGhostStates[i].getPosition(), newPos)
                if manhattanDistance(newGhostStates[i].getPosition(), newPos) < 2:
                    score -= 100000

        if manhattanDistance(currentGameState.getPacmanPosition(), newPos) == 0:
            score -= 1000

        return score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        legal = gameState.getLegalActions(0)
        best_action = 0
        best_value = -float("inf")
        for i in range(0, len(legal)):
            val = self.minimax_value(gameState.getNextState(0, legal[i]), agentIndex=1, depth=0)
            if val > best_value:
                best_action = i
                best_value = val
        return legal[best_action]

    def minimax_value(self, state, agentIndex, depth):
        if (state.isWin() or state.isLose() or depth==self.depth):
            return self.evaluationFunction(state)
        else:
            if agentIndex == 0:
                legal = state.getLegalActions(agentIndex)
                best_value = -float("inf")
                for i in range(0, len(legal)):
                    val = self.minimax_value(state.getNextState(agentIndex, legal[i]), agentIndex + 1, depth=depth)
                    if val > best_value:
                        best_value = val
                return best_value
            elif agentIndex == state.getNumAgents() - 1:
                legal = state.getLegalActions(agentIndex)
                best_value = float("inf")
                for i in range(0, len(legal)):
                    val = self.minimax_value(state.getNextState(agentIndex, legal[i]), agentIndex=0, depth=depth+1)
                    if val < best_value:
                        best_value = val
                return best_value
            else:
                legal = state.getLegalActions(agentIndex)
                best_value = float("inf")
                for i in range(0, len(legal)):
                    val = self.minimax_value(state.getNextState(agentIndex, legal[i]), agentIndex + 1, depth=depth)
                    if val < best_value:
                        best_value = val
                return best_value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legal = gameState.getLegalActions(0)
        best_action = 0
        best_value = -float("inf")
        a = -float("inf")
        b = float("inf")
        for i in range(0, len(legal)):
            val = self.minimax_value(gameState.getNextState(0, legal[i]), agentIndex=1,
                                     depth=0, a=a, b=b)
            if val > best_value:
                best_action = i
                best_value = val
            if best_value > b:
                return best_value
            a = max(a, best_value)
        return legal[best_action]



    def minimax_value(self, state, agentIndex, depth, a, b):
        if (state.isWin() or state.isLose() or depth == self.depth):
            return self.evaluationFunction(state)
        else:
            if agentIndex == 0:
                legal = state.getLegalActions(agentIndex)
                best_value = -float("inf")
                for i in range(0, len(legal)):
                    val = self.minimax_value(state.getNextState(agentIndex, legal[i]),
                                             agentIndex + 1, depth=depth, a=a, b=b)
                    if val > best_value:
                        best_value = val
                    if best_value > b:
                        return best_value
                    a = max(a, best_value)
                return best_value
            elif agentIndex == state.getNumAgents() - 1:
                legal = state.getLegalActions(agentIndex)
                best_value = float("inf")
                for i in range(0, len(legal)):
                    val = self.minimax_value(state.getNextState(agentIndex, legal[i]),
                                             agentIndex=0, depth=depth + 1, a=a, b=b)
                    if val < best_value:
                        best_value = val
                    if best_value < a:
                        return best_value
                    b = min(b, best_value)
                return best_value
            else:
                legal = state.getLegalActions(agentIndex)
                best_value = float("inf")
                for i in range(0, len(legal)):
                    val = self.minimax_value(state.getNextState(agentIndex, legal[i]),
                                             agentIndex + 1, depth=depth, a=a, b=b)
                    if val < best_value:
                        best_value = val
                    if best_value < a:
                        return best_value
                    b = min(b, best_value)
                return best_value

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
        "*** YOUR CODE HERE ***"
        legal = gameState.getLegalActions(0)
        best_action = 0
        best_value = -float("inf")
        for i in range(0, len(legal)):
            val = self.minimax_value(gameState.getNextState(0, legal[i]), agentIndex=1, depth=0)
            if val > best_value:
                best_action = i
                best_value = val
        return legal[best_action]

    def minimax_value(self, state, agentIndex, depth):
        if (state.isWin() or state.isLose() or depth==self.depth):
            return self.evaluationFunction(state)
        else:
            if agentIndex == 0:
                legal = state.getLegalActions(agentIndex)
                best_value = -float("inf")
                for i in range(0, len(legal)):
                    val = self.minimax_value(state.getNextState(agentIndex, legal[i]), agentIndex + 1, depth=depth)
                    if val > best_value:
                        best_value = val
                return best_value
            elif agentIndex == state.getNumAgents() - 1:
                legal = state.getLegalActions(agentIndex)
                avg_value = 0
                for i in range(0, len(legal)):
                    val = self.minimax_value(state.getNextState(agentIndex, legal[i]), agentIndex=0, depth=depth+1)
                    avg_value += val
                return avg_value / len(legal)
            else:
                legal = state.getLegalActions(agentIndex)
                avg_value = 0
                for i in range(0, len(legal)):
                    val = self.minimax_value(state.getNextState(agentIndex, legal[i]), agentIndex + 1, depth=depth)
                    avg_value += val
                return avg_value / len(legal)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    1.) I first iterated through the ghost states and found the distance between the ghost and pacman.
    2.) I used this to reward Pacman for having a high minimum distance from ghosts when they're active. When the
    ghosts are scared I subtracted it instead and weighted it by the time the ghost will be scared for to penalize
    Pacman for being far from scared ghosts.
    3.) I think penalized Pacman for having a large minimum distance to the next food pellet to as to encourage being
    by food. But this metric caused Pacman to stay still next to food for extended periods of time.
    4.) In order to combat this I heavily penalized Pacman for having food left on the board as this is the key to
    moving around and winning
    """
    "*** YOUR CODE HERE ***"

    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood()
    currGhostStates = currentGameState.getGhostStates()
    currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]
    score = currentGameState.getScore()
    minGhostDist = float("inf")


    for i in range(0, len(currGhostStates)):
        k = manhattanDistance(currGhostStates[i].getPosition(), currPos)
        if currScaredTimes[i] > 0:
            score -= (0.5) * currScaredTimes[i] * k
        else:
            if k < minGhostDist:
                minGhostDist = k

    if minGhostDist != float("inf"):
       score += (1) * minGhostDist

    minFoodDist = float("inf")
    nfl = currFood.asList()
    for i in range(0, len(nfl)):
        val = manhattanDistance(currPos, nfl[i])
        if val < minFoodDist:
            minFoodDist = val

    if minFoodDist != float("inf"):
        score -= (3) * minFoodDist

    score -= (20) * currentGameState.getNumFood()

    return score

# Abbreviation
better = betterEvaluationFunction
