# factorOperations.py
# -------------------
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


from bayesNet import Factor
import operator as op
import util

def joinFactorsByVariableWithCallTracking(callTrackingList=None):


    def joinFactorsByVariable(factors, joinVariable):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on 
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that 
        contain that variable.

        Returns a tuple of 
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print "Factor failed joinFactorsByVariable typecheck: ", factor
            raise ValueError, ("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +  
                               "joinVariable: " + str(joinVariable) + "\n" +
                               ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))
        
        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable

joinFactorsByVariable = joinFactorsByVariableWithCallTracking()


def joinFactors(factors):
    """
    Question 1: Your join implementation 

    Input factors is a list of factors.  
    
    You should calculate the set of unconditioned variables and conditioned 
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input 
    (such as getProbability and setProbability) can handle 
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print "Factor failed joinFactors typecheck: ", factor
            raise ValueError, ("unconditionedVariables can only appear in one factor. \n"
                    + "unconditionedVariables: " + str(intersect) + 
                    "\nappear in more than one input factor.\n" + 
                    "Input factors: \n" +
                    "\n".join(map(str, factors)))


    "*** YOUR CODE HERE ***"
    # print "\n IN FACTORS: " , factors , "\n"
    if len(factors) == 1:
        return factors[0]
    factor1 = factors[0]
    factor2 = factors[1]
    finalFactor = multiplyTwoFacts(factor1, factor2)
    for newFact in factors[2:]:
        finalFactor = multiplyTwoFacts(finalFactor,newFact)
    return finalFactor

def multiplyTwoFacts(factor1, factor2):
    # print factor1.variableDomainsDict()
    uncond = []
    cond = []
    uncond += factor1.unconditionedVariables()
    uncond += factor2.unconditionedVariables()
    cond += factor1.conditionedVariables()
    cond += factor2.conditionedVariables()
    uncond = list(set(uncond))
    cond = list(set(cond) - set(uncond))
    finalFactor = Factor(uncond, cond, factor1.variableDomainsDict())

    # multiply probabilities
    # print factor1
    # print factor2
    for assign1 in factor1.getAllPossibleAssignmentDicts():
        for assign2 in factor2.getAllPossibleAssignmentDicts():
            prob1 = factor1.getProbability(assign1)
            prob2 = factor2.getProbability(assign2)
            finalProb = prob1 * prob2
            # print "ASSIGN1: " , assign1 , " " , prob1
            # print "ASSIGN2: " , assign2 , " " , prob2
            if consistentDict(assign1, assign2):
                temp = assign1.copy()
                temp.update(assign2)
                # print "Setting " , temp , " to " , finalProb
                finalFactor.setProbability(temp, finalProb)
            # else:
                # print "NO UPDATE"
            # print " "

    return finalFactor

def consistentDict(d1,d2):
    for key in d2:
        # print key
        if key in d1:
            if d2[key] != d1[key]:
                return False
    return True


def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor, eliminationVariable):
        """
        Question 2: Your eliminate implementation 

        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.
        
        You should calculate the set of unconditioned variables and conditioned 
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print "Factor failed eliminate typecheck: ", factor
            raise ValueError, ("Elimination variable is not an unconditioned variable " \
                            + "in this factor\n" + 
                            "eliminationVariable: " + str(eliminationVariable) + \
                            "\nunconditionedVariables:" + str(factor.unconditionedVariables()))
        
        if len(factor.unconditionedVariables()) == 1:
            print "Factor failed eliminate typecheck: ", factor
            raise ValueError, ("Factor has only one unconditioned variable, so you " \
                    + "can't eliminate \nthat variable.\n" + \
                    "eliminationVariable:" + str(eliminationVariable) + "\n" +\
                    "unconditionedVariables: " + str(factor.unconditionedVariables()))

        "*** YOUR CODE HERE ***"
        # print factor
        # print "ELIMINATE: " , eliminationVariable
        # print factor.getAllPossibleAssignmentDicts()[0] , " " , factor.getProbability( factor.getAllPossibleAssignmentDicts()[0] )
        uncond = factor.unconditionedVariables()
        cond = factor.conditionedVariables()
        # print uncond
        # print cond
        # print eliminationVariable
        try:
            uncond.remove(eliminationVariable)
        except ValueError:
            cond.remove(eliminationVariable)
        finalFactor = Factor(uncond, cond, factor.variableDomainsDict())
        leng = 1
        for domain in factor.variableDomainsDict():
            values = factor.variableDomainsDict()[domain]
            leng = leng * (len(values))
        # print "midleng: " , leng
        # print len(factor.variableDomainsDict()[eliminationVariable])
        leng = leng / len(factor.variableDomainsDict()[eliminationVariable])
        # print "LENG: " , leng
        # print leng
        # assign = finalFactor.getAllPossibleAssignmentDicts()[0]
        # factor.getProbability(assign)
        for assign in finalFactor.getAllPossibleAssignmentDicts():
            # print "ASSIGN: " ,  assign
            tempProb = 0
            for ogAssign in factor.getAllPossibleAssignmentDicts():
                # print "OGASSIGN: " , ogAssign
                # print "assign: " , assign
                # print "ogAssign: " , ogAssign
                temp = ogAssign
                temp.update(assign)
                tempProb += factor.getProbability(temp)
            # print ""
            finalFactor.setProbability(assign, tempProb/leng)
        return finalFactor




    return eliminate

eliminate = eliminateWithCallTracking()


def normalize(factor):
    """
    Question 3: Your normalize implementation 

    Input factor is a single factor.

    The set of conditioned variables for the normalized factor consists 
    of the input factor's conditioned variables as well as any of the 
    input factor's unconditioned variables with exactly one entry in their 
    domain.  Since there is only one entry in that variable's domain, we 
    can either assume it was assigned as evidence to have only one variable 
    in its domain, or it only had one entry in its domain to begin with.
    This blurs the distinction between evidence assignments and variables 
    with single value domains, but that is alright since we have to assign 
    variables that only have one value in their domain to that single value.

    Return a new factor where the sum of the all the probabilities in the table is 1.
    This should be a new factor, not a modification of this factor in place.

    If the sum of probabilities in the input factor is 0,
    you should return None.

    This is intended to be used at the end of a probabilistic inference query.
    Because of this, all variables that have more than one element in their 
    domain are assumed to be unconditioned.
    There are more general implementations of normalize, but we will only 
    implement this version.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    variableDomainsDict = factor.variableDomainsDict()
    for conditionedVariable in factor.conditionedVariables():
        if len(variableDomainsDict[conditionedVariable]) > 1:
            print "Factor failed normalize typecheck: ", factor
            raise ValueError, ("The factor to be normalized must have only one " + \
                            "assignment of the \n" + "conditional variables, " + \
                            "so that total probability will sum to 1\n" + 
                            str(factor))

    "*** YOUR CODE HERE ***"
    # print "INPUT FACTOR: " , factor
    # print "POSSIBLE: " , factor.getAllPossibleAssignmentDicts()
    # determine which variables are conditioned
    dictt = {}
    uncond = factor.unconditionedVariables()
    cond = factor.conditionedVariables()
    # print cond
    for assign in factor.getAllPossibleAssignmentDicts():
        # print assign
        for variable in assign:
            # print variable
            if variable in dictt:
                dictt[variable].add(assign[variable])
            else:
                dictt[variable] = set([assign[variable]])
    # print "DICTT: " , dictt
    for key in dictt:
        # print key
        if len(dictt[key]) == 1:
            if key not in cond:
                cond += [key]
            if key in uncond:
                uncond.remove(key)
    # print "COND: ", cond
    # print "UNCOND: ", uncond
    summ = 0
    for dom in factor.getAllPossibleAssignmentDicts():
        summ += factor.getProbability(dom)
    if summ == 0:
        return None
    finalFactor = Factor(uncond , cond, factor.variableDomainsDict())
    for dom in factor.getAllPossibleAssignmentDicts():
        temp = factor.getProbability(dom)
        temp = temp / summ
        finalFactor.setProbability(dom, temp)
    return finalFactor

