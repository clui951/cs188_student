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

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)  # returns list of tuples of (new_state, probability)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # V_k+1(s) = max_a { summ_s' { T(s,a,s') * [R(s,a,s') + alpha*V_k(s')] } }

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # print "HERE"
        # print "o"
        # print self.iterations
        while self.iterations != 0:
          # print self.iterations
          updates = []
          for state in self.mdp.getStates():
            stateActions = self.mdp.getPossibleActions(state)
            actionResults = []      # tuples of (action,summed result)
            for action in stateActions:
              # print state, " with action " , action, " results in " , self.mdp.getTransitionStatesAndProbs(state,action)
              Q_val = self.computeQValueFromValues(state,action)
              # actionSum = 0
              # for transitions in mdp.getTransitionStatesAndProbs(state,action):
                # resultingState = transitions[0]    # resulting state
                # prob = transitions[1]    # probability
                # reward = mdp.getReward(state, action, resultingState)   # reward from transition
                # miniSum = prob * (reward + (discount * self.values[resultingState]))
                # actionSum += miniSum
              actionResults += [(action,Q_val)]
            v_k1_val = -999999999
            flag = False
            for action in actionResults:
                if action[1] > v_k1_val:
                  v_k1_val = action[1]
                flag = True
            if flag:
              # self.values[state] = v_k1_val         # do not update real time
              updates += [(state,v_k1_val)]
          self.iterations -= 1

          # update values
          for update in updates:
            self.values[update[0]] = update[1]

        # print "\nHEY"
        # print self.values




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
        "*** YOUR CODE HERE ***"
        summ = 0
        for trans in self.mdp.getTransitionStatesAndProbs(state,action):
          T_val = trans[1]    # transition probability
          resultingState = trans[0]
          reward = self.mdp.getReward(state, action, resultingState)
          s_prime_val = self.values[resultingState]
          summ += T_val * (reward + (s_prime_val * self.discount))
        return summ

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
          return None
        maxVal = -9999999999
        maxAct = None
        for action in self.mdp.getPossibleActions(state):
          tempVal = self.computeQValueFromValues(state,action)
          if tempVal > maxVal:
            maxAct = action
            maxVal = tempVal
        return maxAct 


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
