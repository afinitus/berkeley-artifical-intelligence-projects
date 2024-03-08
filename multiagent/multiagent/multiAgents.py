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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        #we can make a score based on the food locations only as a good evaluation function
        food = newFood.asList()
        food_dist = [0]
        for pellet in food:
            #the project spec recommends us to do 1 over the distance and the closest food is then the biggest
            #value that this distance gives us
            food_dist.append(1.0 / manhattanDistance(newPos, pellet))
        food_score = max(food_dist)
        #to get the best estimate we will add all possible scores available
        return successorGameState.getScore() + food_score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        num_ghosts = gameState.getNumAgents() - 1
        #we follow the same format as in lecture where we have a max and min helper function
        def maxval(gameState, depth):
            #keep track of the depth, and check (init conditions) if the game is over or if we hit the depth limit
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)
            actions = gameState.getLegalActions(0)
            #we get the actions and if there are none then we also just return the current state eval
            if not actions:
                return (self.evaluationFunction(gameState), None)
            #we init a value v to negative infinity as the node values
            v = float("-inf")
            #iterate through the actions and create all the possible new states and for each new state we
            #will get the max of the minimizing agents starting with ghost 1
            for action in actions:
                newstate = gameState.generateSuccessor(0, action)
                minimizer = minval(newstate, depth, 1)
                #we choose the eval functon value that is the biggest out of all the minimized nodes here
                if minimizer[0] > v:
                    move = action
                    v = minimizer[0]
            return (v, move)

        def minval(gameState, depth, index):
            #keep track of the depth, and check (init conditions) if the game is over or if we hit the depth limit
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)
            actions = gameState.getLegalActions(index)
            #we get the actions and if there are none then we also just return the current state eval
            if not actions:
                return (self.evaluationFunction(gameState), None)
            #we init a value v to infinity as the node values
            v = float("inf")
            #iterate through the actions and create all the possible new states and for each new state we
            #will get the min of all the ghosts actions, adn once we have gone through all the ghosts we
            #switch back to the max or pacman play and increase the depth of the tree
            for action in actions:
                newstate = gameState.generateSuccessor(index, action)
                if index != num_ghosts:
                    nextagent = minval(newstate, depth, index + 1)
                else:
                    nextagent = maxval(newstate, depth + 1)
                #here we want the eval function to be the smallest of the maximized values
                if nextagent[0] < v:
                    move = action
                    v = nextagent[0]
            return (v, move)
        return maxval(gameState, 0)[1]



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        a = float("-inf")
        b = float("inf")
        num_ghosts = gameState.getNumAgents() - 1
        #we follow the same format as in lecture where we have a max and min helper function
        def maxval(gameState, depth, a, b):
            #keep track of the depth, and check (init conditions) if the game is over or if we hit the depth limit
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)
            actions = gameState.getLegalActions(0)
            #we get the actions and if there are none then we also just return the current state eval
            if not actions:
                return (self.evaluationFunction(gameState), None)
            #we init a value v to negative infinity as the node values
            v = float("-inf")
            #iterate through the actions and create all the possible new states and for each new state we
            #will get the max of the minimizing agents starting with ghost 1
            for action in actions:
                newstate = gameState.generateSuccessor(0, action)
                minimizer = minval(newstate, depth, 1, a, b)
                #we choose the eval functon value that is the biggest out of all the minimized nodes here
                if minimizer[0] > v:
                    move = action
                    v = minimizer[0]
                #all we need is the same function as minmax but now we add our alpha beta pruning
                #if statements as according to the description in lecture:
                if v > b:
                    return (v, move)
                if v > a:
                    a = v
            return (v, move)

        def minval(gameState, depth, index, a, b):
            #keep track of the depth, and check (init conditions) if the game is over or if we hit the depth limit
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)
            actions = gameState.getLegalActions(index)
            #we get the actions and if there are none then we also just return the current state eval
            if not actions:
                return (self.evaluationFunction(gameState), None)
            #we init a value v to infinity as the node values
            v = float("inf")
            #iterate through the actions and create all the possible new states and for each new state we
            #will get the min of all the ghosts actions, adn once we have gone through all the ghosts we
            #switch back to the max or pacman play and increase the depth of the tree
            for action in actions:
                newstate = gameState.generateSuccessor(index, action)
                if index != num_ghosts:
                    nextagent = minval(newstate, depth, index + 1, a, b)
                else:
                    nextagent = maxval(newstate, depth + 1, a, b)
                #here we want the eval function to be the smallest of the maximized values
                if nextagent[0] < v:
                    move = action
                    v = nextagent[0]
                #all we need is the same function as minmax but now we add our alpha beta pruning
                #if statements as according to the description in lecture:
                if v < a:
                    return (v, move)
                if v < b:
                    b = v
            return (v, move)
        return maxval(gameState, 0, a, b)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        num_ghosts = gameState.getNumAgents() - 1
        #we follow the same format as in lecture where we have a max and min helper function
        def maxval(gameState, depth):
            #keep track of the depth, and check (init conditions) if the game is over or if we hit the depth limit
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)
            actions = gameState.getLegalActions(0)
            #we get the actions and if there are none then we also just return the current state eval
            if not actions:
                return (self.evaluationFunction(gameState), None)
            #we init a value v to negative infinity as the node values
            v = float("-inf")
            #iterate through the actions and create all the possible new states and for each new state we
            #will get the max of the minimizing agents starting with ghost 1
            for action in actions:
                newstate = gameState.generateSuccessor(0, action)
                #only change here is that instead of minimizing we choose at random, so expec is called:
                minimizer = expecval(newstate, depth, 1)
                #we choose the eval functon value that is the biggest out of all the minimized nodes here
                if minimizer[0] > v:
                    move = action
                    v = minimizer[0]
            return (v, move)

        def expecval(gameState, depth, index):
            #keep track of the depth, and check (init conditions) if the game is over or if we hit the depth limit
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)
            actions = gameState.getLegalActions(index)
            #we get the actions and if there are none then we also just return the current state eval
            if not actions:
                return (self.evaluationFunction(gameState), None)
            #we init a value v to 0 for expectimax case
            v = 0
            #iterate through the actions and create all the possible new states and for each new state we
            #will get the min of all the ghosts actions, adn once we have gone through all the ghosts we
            #switch back to the max or pacman play and increase the depth of the tree
            for action in actions:
                newstate = gameState.generateSuccessor(index, action)
                if index != num_ghosts:
                    nextagent = expecval(newstate, depth, index + 1)
                else:
                    nextagent = maxval(newstate, depth + 1)
                #here we want to add the probability (1 over the total actions) times the value of the state as according to lec slides
                v += nextagent[0] / len(actions)
            return (v, None)
        return maxval(gameState, 0)[1]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I had used the same function that I wrote in part a, but I took into account the position of the ghosts as well now.
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    # we want to take into account both the ghost positions and the food positions to get our score for the state
    ghost_pos_score = 0
    #here we get our ghost score
    for i in range(len(newGhostStates)):
        #for each ghost we will use the distance and wether or not they are scared, so if they are
        #scared then the closer we are to them the better and vice versa if they are not scared, then
        #the farther the better
        scared = newScaredTimes[i]
        ghost = newGhostStates[i]
        ghost_dist = manhattanDistance(newPos, ghost.getPosition())
        if scared > 0:
            #close ghosts while scared should add points but if they are farther than 4 in distance than
            #there is no point of us adding it to the score as it is irrelevant
            #this was decided by varying the values from 1-9 and 4 gives the best output
            ghost_pos_score += max(4 - ghost_dist, 0)
        else:
            #same here except we add more urgency for ghosts closer to us, so we will lower the score
            #but if they are far away it does not matter as much
            ghost_pos_score -= max(4 - ghost_dist, 0)
    food = newFood.asList()
    food_dist = [0]
    for pellet in food:
            #the project spec recommends us to do 1 over the distance and the closest food is then the biggest
            #value that this distance gives us
        food_dist.append(1.0 / manhattanDistance(newPos, pellet))
    food_score = max(food_dist)
    #to get the best estimate we will add all possible scores available
    return currentGameState.getScore() + food_score + ghost_pos_score

# Abbreviation
better = betterEvaluationFunction
