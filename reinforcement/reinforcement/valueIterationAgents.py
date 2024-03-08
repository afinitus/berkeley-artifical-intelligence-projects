# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        #run the iteration for the total number of iterations
        for i in range(self.iterations):
            #make a copy of the current values for the states, so we can edit them
            statevals = self.values.copy()
            #get the list of all the current states
            possiblestates = self.mdp.getStates()
            for state in possiblestates:
            #make sure the game hasnt ended in this state
                if not self.mdp.isTerminal(state):
                #all possible actions from this state
                    possibleactions = self.mdp.getPossibleActions(state)
                    #we want to find the best action from this state by getting the highest Qvalue
                    QVals = []
                    for action in possibleactions:
                        QVals.append(self.getQValue(state, action))
                    highestval = max(QVals)
                    statevals[state] = highestval
            self.values = statevals.copy()
            

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        #first get all the pairs of newstates and probabilities of each state
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        #set our current Q value to be 0
        Q = 0
        #iterate over all possible new states and their probabilities
        for S, P in transitions:
            #get the reward for entering the new state
            R = self.mdp.getReward(state, action, S)
            #use the value iteration formula as in lecture to get one of our sum values
            qsummation = P*(R + self.discount*self.values[S])
            #add this value to our overall Q value
            Q += qsummation
        return Q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        #we need to first check the base case of if we are in a final state
        if self.mdp.isTerminal(state):
            return None
        #first we make a dictionary to hold all the policies/actions and their assocaited values
        policies = util.Counter()
        #we want to iterate through all possible actions in our current state
        possibleactions = self.mdp.getPossibleActions(state)
        #We will follow the same formula format as in lecture, where for each action we want 
        # to get the Qvalue and store it as a policy, and in the end we want to return the 
        #argMax of our policies
        for action in possibleactions:
            policies[action] = self.getQValue(state, action)
        return policies.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
