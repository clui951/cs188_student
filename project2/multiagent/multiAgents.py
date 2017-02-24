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
# from searchAgents import mazeDistance
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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        # print newFood[1]
        # print newFood
        # print newScaredTimes
        # print "\n NEW MOVE "
        ghostPos = []
        for ghost,time in zip(newGhostStates,newScaredTimes):
          # print time
          if time == 0:
            ghostPos += [(ghost.getPosition())]
          # else:
            # print "NOT AVOIDING"

        val = 300000

        ghostSum = 0
        for ghost in ghostPos:
          if (manhattanDistance(ghost, newPos) <= 2):
            ghostSum = -100000
        # print "GHOST SUM: " , ghostSum

        foods = []
        foodCount = 0
        for pos in newFood.asList():
          foodCount += 1  
          foods += [manhattanDistance(pos , newPos)]
        # print "min(foods): " , min(foods)
        # print "FOOD COUNT: " , foodCount
        if len(foods) == 0:
          return 1000000000
        val += (ghostSum - (min(foods))  - 100 * foodCount)
        # print "VAL IS: ", val
        return val

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

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        evalFunc = self.evaluationFunction

        def max_agent(gameState, currDepth, totalGhosts, currGhost):
          if gameState.isWin() or gameState.isLose():
            return evalFunc(gameState)
          elif (currDepth  == depth):    
            return evalFunc(gameState)
          val = []
          for action in gameState.getLegalActions(0):   # it is always pacman maxxing
            succ = gameState.generateSuccessor(0,action)
            val += [min_agent(succ, currDepth , totalGhosts, currGhost)]
          return max(val)

        def min_agent(gameState, currDepth, totalGhosts, currGhost):
          if gameState.isWin() or gameState.isLose():
            return evalFunc(gameState)
          val = []
          if currGhost == totalGhosts:
            for action in gameState.getLegalActions(currGhost):
              val += [ max_agent(gameState.generateSuccessor(currGhost,action) , currDepth + 1, totalGhosts , 1) ]
          else:
            for action in gameState.getLegalActions(currGhost):
              val += [min_agent(gameState.generateSuccessor(currGhost,action) , currDepth , totalGhosts , currGhost + 1)]
          return min(val)

        val = []
        actions = []
        startingActions = gameState.getLegalActions(0)
        for action in startingActions:
          val += [min_agent(gameState.generateSuccessor(0, action), 0, gameState.getNumAgents() - 1, 1)]
          actions += [action]
        ind = val.index(max(val))
        return actions[ind]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        evalFunc = self.evaluationFunction
        def max_agent(gameState, currDepth, totalGhosts, currGhost, alpha, beta):
            if gameState.isWin() or gameState.isLose():
                return evalFunc(gameState)
            elif (currDepth  == depth):    
                return evalFunc(gameState)
            val = -9999999999
            for action in gameState.getLegalActions(0):   # it is always pacman maxxing
                succ = gameState.generateSuccessor(0,action)
                tempVal = min_agent(succ, currDepth , totalGhosts, currGhost, alpha, beta)
                val = max(val,tempVal)
                if (val > beta):
                    break
                    # print "PRUNE MAX"
                alpha = max(val,alpha)
                # print "MAX OF " , val , " " , alpha
            return val


        def min_agent(gameState, currDepth, totalGhosts, currGhost, alpha, beta):
            if gameState.isWin() or gameState.isLose():
                return evalFunc(gameState)
            val = 9999999999
            if currGhost == totalGhosts:
                for action in gameState.getLegalActions(currGhost):
                    tempVal = max_agent(gameState.generateSuccessor(currGhost,action) , currDepth + 1, totalGhosts , 1, alpha, beta) 
                    # print "\nALPHA BETA is ", alpha, " " , beta
                    # print "TEMPVAL is ", tempVal
                    val = min(val,tempVal)
                    # print "VAL is " , val
                    if val < alpha:
                        # print "PRUNE MIN"
                        return val
                    beta = min(val, beta)
            else:
                for action in gameState.getLegalActions(currGhost):
                    tempVal = min_agent(gameState.generateSuccessor(currGhost,action) , currDepth , totalGhosts , currGhost + 1, alpha, beta)
                    val = min(val,tempVal)
                    if val < alpha:
                        # print "PRUNE MIN"
                        return val
                    beta = min(val,beta)
            return val

        val = -9999999999
        alpha = -9999999999
        beta = 99999999999
        actionFin = None
        startingActions = gameState.getLegalActions(0)
        for action in startingActions:
            tempVal = min_agent(gameState.generateSuccessor(0, action), 0, gameState.getNumAgents() - 1, 1, alpha,beta)
            if tempVal > val:
                val = tempVal
                actionFin = action
            alpha = max(val , alpha)
        return actionFin




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
        depth = self.depth
        evalFunc = self.evaluationFunction
        def max_agent(gameState, currDepth, numGhosts):
            if (gameState.isWin() or gameState.isLose()):
                return evalFunc(gameState)
            elif currDepth == depth:
                return evalFunc(gameState)
            val = -99999999
            for action in gameState.getLegalActions(0):
                tempState = gameState.generateSuccessor(0,action)
                val = max(val , expect_agent(tempState, currDepth, numGhosts, 1))
            return val

        def expect_agent(gameState, currDepth, numGhosts, currGhost):
            if (gameState.isWin() or gameState.isLose()):
                # print "WIN or LOSE"
                return evalFunc(gameState)
            elif currDepth == depth:
                return evalFunc(gameState)
            actionCount = 0
            val = 0
            if (currGhost == numGhosts):
                for action in gameState.getLegalActions(currGhost):
                    actionCount += 1
                    tempState = gameState.generateSuccessor(currGhost,action)
                    val += max_agent(tempState, currDepth + 1, numGhosts)
            else:
                for action in gameState.getLegalActions(currGhost):
                    actionCount += 1
                    tempState = gameState.generateSuccessor(currGhost,action)
                    val += expect_agent(tempState, currDepth, numGhosts, currGhost + 1)
            return val/actionCount

        val = -99999999999
        actionFin = None
        for action in gameState.getLegalActions(0):
            tempState = gameState.generateSuccessor(0,action)
            tempVal = expect_agent(tempState, 0, gameState.getNumAgents() - 1, 1)
            if tempVal > val:
                val = tempVal
                actionFin = action
        return actionFin



def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: 
        We want to be near the closestfood possible, so go through all the food, and take the minimum distance to it because we want the distance to be less.
        Set it to 1 if no food, just to avoid division by 0 error.
        Next, we find the distances to all ghosts, trying to be further from them
        We will actively actively seek them out only if they can be eaten within the time limit
    """
    "*** YOUR CODE HERE ***"
    currPos = currentGameState.getPacmanPosition()
    foodCount = currentGameState.getNumFood()

    closestfood = 999999999
    foodDists = []
    for food in currentGameState.getFood().asList():
        foodDists += [manhattanDistance(currPos, food)]
    if (len(foodDists) != 0):
      closestFood = min(foodDists)
    else:
      foodDists = 1

    nearestGhostDistance = 999999999
    ghostEat = 0
    ghostDists = []
    for ghost in currentGameState.getGhostStates():
      ghostX = int(ghost.getPosition()[0]) 
      ghostY = int(ghost.getPosition()[1])
      ghostDists += [mazeDistance(currPos, (ghostX,ghostY), currentGameState)]

    nearestGhost = min(ghostDists)
    if ghost.scaredTimer > min(ghostDists):
      ghostEat = 250

    return currentGameState.getScore() + (ghostEat - foodCount - nearestGhost) + 1/closestfood  

# Abbreviation
better = betterEvaluationFunction






##########################
# BEGIN PROJECT 1 IMPORTS





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

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    closed = set([])
    fringe = util.Stack()
    closed.add(problem.getStartState())
    for successors in problem.getSuccessors(problem.getStartState()):
        fringe.push( ( successors[0] , [successors[1]] ) )
    while True:
        if fringe.isEmpty():
            return []
        nodes = fringe.pop()
        if problem.isGoalState(nodes[0]):
            return nodes[1]
        if nodes[0] not in closed:
            closed.add(nodes[0])
            for successors in problem.getSuccessors(nodes[0]):
                # print successors[0]
                fringe.push((successors[0], nodes[1] + [successors[1]] ))

    # nodes = (successor[state, action, cost], [actions to prev])


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    closed = set([])
    fringe = util.Queue()
    closed.add(problem.getStartState())
    for successors in problem.getSuccessors(problem.getStartState()):
        fringe.push( ( successors[0] , [successors[1]] ) )
    while True:
        if fringe.isEmpty():
            return []
        nodes = fringe.pop()
        # print "NODE BEING SEARCHED: " , nodes
        if problem.isGoalState(nodes[0]):
            # print "last node: ", nodes
            # print "Path is: ", nodes[1]
            # print
            return nodes[1]
        if nodes[0] not in closed:
            closed.add(nodes[0])
            for successors in problem.getSuccessors(nodes[0]):
                # print successors[0]
                fringe.push((successors[0], nodes[1] + [successors[1]] ))

# STATES: 
# getSuccessors returns [ ( state , action , cost ) ]
# FRINGE / NODES : ( state , [path so far] )
# NODES VS STATES
    # STATES ARE ONLY INFO OF THE BOARD CONFIG SO FAR
    # NODES ARE THINGS ON THE FRINGE, WHICH IS (STATES, PATH SO FAR)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    closed = set([])
    fringe = util.PriorityQueue()
    closed.add(problem.getStartState())
    for successors in problem.getSuccessors(problem.getStartState()):
        fringe.push( ( successors[0] , successors[2], [successors[1]] ), successors[2] )
    while True:
        if fringe.isEmpty():
            return []
        nodes = fringe.pop()
        if problem.isGoalState(nodes[0]):
            return nodes[2]
        if nodes[0] not in closed:
            closed.add(nodes[0])
            for successors in problem.getSuccessors(nodes[0]):
                # print successors[0]
                fringe.push((successors[0], nodes[1] + successors[2], nodes[2] + [successors[1]] ), nodes[1] + successors[2] )

# getSuccessors returns [ ( state , action , cost ) ]
# FRINGE / NODES : ( state , cost so far, [path so far] )



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    closed = set([])
    fringe = util.PriorityQueue()
    closed.add(problem.getStartState())
    for successors in problem.getSuccessors(problem.getStartState()):
        fringe.push( ( successors[0] , successors[2], [successors[1]] ), successors[2] + heuristic(successors[0],problem) )
    while True:
        if fringe.isEmpty():
            return []
        nodes = fringe.pop()
        if problem.isGoalState(nodes[0]):
            return nodes[2]
        if nodes[0] not in closed:
            closed.add(nodes[0])
            for successors in problem.getSuccessors(nodes[0]):
                # print successors[0]
                fringe.push((successors[0], nodes[1] + successors[2], nodes[2] + [successors[1]] ), nodes[1] + successors[2] + heuristic(successors[0],problem))

# getSuccessors returns [ ( state , action , cost ) ]
# FRINGE / NODES : ( state , cost so far, [path so far] )





# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch




class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost


def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(bfs(prob))


class Actions:
    """
    A collection of static methods for manipulating move actions.
    """
    # Directions
    _directions = {Directions.NORTH: (0, 1),
                   Directions.SOUTH: (0, -1),
                   Directions.EAST:  (1, 0),
                   Directions.WEST:  (-1, 0),
                   Directions.STOP:  (0, 0)}

    _directionsAsList = _directions.items()

    TOLERANCE = .001

    def reverseDirection(action):
        if action == Directions.NORTH:
            return Directions.SOUTH
        if action == Directions.SOUTH:
            return Directions.NORTH
        if action == Directions.EAST:
            return Directions.WEST
        if action == Directions.WEST:
            return Directions.EAST
        return action
    reverseDirection = staticmethod(reverseDirection)

    def vectorToDirection(vector):
        dx, dy = vector
        if dy > 0:
            return Directions.NORTH
        if dy < 0:
            return Directions.SOUTH
        if dx < 0:
            return Directions.WEST
        if dx > 0:
            return Directions.EAST
        return Directions.STOP
    vectorToDirection = staticmethod(vectorToDirection)

    def directionToVector(direction, speed = 1.0):
        dx, dy =  Actions._directions[direction]
        return (dx * speed, dy * speed)
    directionToVector = staticmethod(directionToVector)

    def getPossibleActions(config, walls):
        possible = []
        x, y = config.pos
        x_int, y_int = int(x + 0.5), int(y + 0.5)

        # In between grid points, all agents must continue straight
        if (abs(x - x_int) + abs(y - y_int)  > Actions.TOLERANCE):
            return [config.getDirection()]

        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if not walls[next_x][next_y]: possible.append(dir)

        return possible

    getPossibleActions = staticmethod(getPossibleActions)

    def getLegalNeighbors(position, walls):
        x,y = position
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        neighbors = []
        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_x = x_int + dx
            if next_x < 0 or next_x == walls.width: continue
            next_y = y_int + dy
            if next_y < 0 or next_y == walls.height: continue
            if not walls[next_x][next_y]: neighbors.append((next_x, next_y))
        return neighbors
    getLegalNeighbors = staticmethod(getLegalNeighbors)

    def getSuccessor(position, action):
        dx, dy = Actions.directionToVector(action)
        x, y = position
        return (x + dx, y + dy)
    getSuccessor = staticmethod(getSuccessor)
