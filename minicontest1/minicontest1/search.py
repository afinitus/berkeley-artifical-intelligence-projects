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
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    #contains all the nodes we have already looked in
    searched = []
    #contains all the nodes we will look into with LIFO policy
    nodes = util.Stack()
    nodes.push((problem.getStartState(), []))
    #follow the same format as code given in lecture for graph search here:
    while True:
        if nodes.isEmpty():
            return []
        state = nodes.pop()
        curr_node = state[0]
        curr_route = state[1]
        if problem.isGoalState(curr_node):
            return curr_route
        elif curr_node not in searched:
            searched.append(curr_node)
            for ancestors in problem.getSuccessors(curr_node):
                next_node = ancestors[0]
                next_move = ancestors[1]
                new_route = curr_route + [next_move]
                nodes.push((next_node, new_route)) 

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    #contains all the nodes we have already looked in
    searched = []
    #contains all the nodes we will look into but now a queue because we need FIFO
    nodes = util.Queue()
    nodes.push((problem.getStartState(), []))
    #follow the same format as code given in lecture for graph search here:
    while True:
        if nodes.isEmpty():
            return []
        state = nodes.pop()
        curr_node = state[0]
        curr_route = state[1]
        if problem.isGoalState(curr_node):
            return curr_route
        elif curr_node not in searched:
            searched.append(curr_node)
            for ancestors in problem.getSuccessors(curr_node):
                next_node = ancestors[0]
                next_move = ancestors[1]
                new_route = curr_route + [next_move]
                nodes.push((next_node, new_route)) 

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    #contains all the nodes we have already looked in
    searched = []
    #contains all the nodes we will look into but now a priority queue because we need least cost
    nodes = util.PriorityQueue()
    nodes.push((problem.getStartState(), []), 0)
    #follow the same format as code given in lecture for graph search here, but we only change
    #the priorities that we were passing in according to the path to be taken and path taken so far:
    while True:
        if nodes.isEmpty():
            return []
        state = nodes.pop()
        curr_node = state[0]
        curr_route = state[1]
        if problem.isGoalState(curr_node):
            return curr_route
        elif curr_node not in searched:
            searched.append(curr_node)
            for ancestors in problem.getSuccessors(curr_node):
                next_node = ancestors[0]
                next_move = ancestors[1]
                new_route = curr_route + [next_move]
                priority = ancestors[2]
                nodes.push((next_node, new_route), priority + problem.getCostOfActions(curr_route))
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    #contains all the nodes we have already looked in
    searched = []
    #contains all the nodes we will look into but now a priority queue because we need least cost
    nodes = util.PriorityQueue()
    nodes.push((problem.getStartState(), []), 0)
    #follow the same format as code given in lecture for graph search here, but we only change
    #the priorities that we were passing in according to the path to be taken and path taken so far:
    while True:
        if nodes.isEmpty():
            return []
        state = nodes.pop()
        curr_node = state[0]
        curr_route = state[1]
        if problem.isGoalState(curr_node):
            return curr_route
        elif curr_node not in searched:
            searched.append(curr_node)
            for ancestors in problem.getSuccessors(curr_node):
                next_node = ancestors[0]
                next_move = ancestors[1]
                new_route = curr_route + [next_move]
                priority = ancestors[2]
                nodes.push((next_node, new_route), priority + problem.getCostOfActions(curr_route) + heuristic(next_node, problem))

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
