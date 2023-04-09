#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi cos
"""

import chess
import numpy as np
import sys


class Aichess():

    """
    A class to represent the game of chess.

    ...

    Attributes:
    -----------
    chess : Chess
        represents the chess game

    Methods:
    --------
    startGame(pos:stup) -> None
        Promotes a pawn that has reached the other side to another, or the same, piece



    """

    def __init__(self, TA, myinit=True):

        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.pathToTarget = []
        self.currentStateW = self.chess.currentStateW.copy()
        self.depthMax = 4

        self.checkMatestates = ([[0, 0, 2], [2, 4, 6]], [[0, 1, 2], [2, 4, 6]], [[0, 2, 2], [2, 4, 6]], [[0, 6, 2], [2, 4, 6]], [[0, 7, 2], [2, 4, 6]])

        # This 2 dictionaries will contain the Q values for each state and its possible actions. One will be for white pieces and the other for blacks
        self.QtableW = {}
        self.QtableB = {}
        # In this 2 diccionaries we will get the count of the frequency of each state-action relation
        self.freq_W = {}
        self.freq_B = {}
        # This variable gives us the learing rate
        self.alpha = 0.5
        # This variable will give us the discount factor
        self.gamma = 0.7
        # This varaible will help us determinaiting if we do exploration or exploitation, we are using the epsilon greedy technique
        self.epsilon = 0.8

    def getCurrentState(self):
    
        return self.myCurrentStateW
    
    
    def getListNextStatesW(self, myState):
    
        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates


    def isVisited(self, mystate):
    
        if mystate in self.listVisitedStates:
            return True
        else:
            return False

    def isCheckMate1(self, mystate):
        if mystate in self.checkMatestates:
            return True
        mystate2 = [mystate[1], mystate[0]]
        if mystate2 in self.checkMatestates:
            return True

        return False

    # Function that given a state, turns all its data into tuples. Therefor,we will able to index it in a dictionary
    def stateToTuple(self, mystate):
        state = []
        for piece in range(0,len(mystate)):
            state.append(tuple(mystate[piece]))
        return tuple(state)

    # Function that will return us the reward of a given state, thought for the first starting case
    def getReward1(self, mystate):
        min = 17

        for targetState in self.checkMatestates:
            for pieceT in targetState:
                for piece in mystate:
                    x = abs(piece[0] - pieceT[0]) + abs(piece[1] - pieceT[1]) # Distancia Manhatan
                    if x < min:
                        min = x

        if self.isCheckMate1(mystate):
                return 1000
        else:
            return -1 * min # Moves that get us closer to check mate have a better reward

    # Auxiliar function that will decide if we do exploitation or exploration
    def getGreedy(self):
        # We generate a random number between 0 and 1
        num = np.random.uniform(0, 1);
        # If the generated number is smaller than our epsilon, we will do Exploitation
        if num <= self.epsilon:
            return True;
        # Else, we will explore
        else:
            return False;


    # Auxiliar function to initialize a state in the Q table and frequency table
    def initializeState(self, state, tuple):

        # We create a tuple version of the state to make it easier to operate with
        if not tuple:
            tupledState = self.stateToTuple(state)
        else:
            tupledState = state
            state = []
            for element in tupledState:
                state.append(list(element))

        self.QtableW[tupledState] = {}
        self.freq_W[tupledState] = {}
        for action in self.getListNextStatesW(state):
            # We make sure that staying in the same state its self it's not an action, since we are forced to move one piece
            if action != state:
                tupledAction = self.stateToTuple(action)
                self.QtableW[tupledState][tupledAction] = 0
                self.freq_W[tupledState][tupledAction] = 0


    # Auxiliar function that returns you the action with a maximum Q value
    def getMaximumFuture(self, currentState, turn):
        # If white's turn
        if turn:
            # If we haven't seen this state yet, we add it to the table and initialize it
            if currentState not in self.QtableW:
                self.initializeState(currentState,True)

            # We get the best possible action, this beeing the one with the biggest Q value
            maximum = max(self.QtableW[currentState].values())

            # If the maximum is 0, this means no actions have been previously explored, there for we will choose a random one
            if maximum == 0:
                pos = np.random.choice(range(0, len(self.QtableW[currentState])))
            # Else, we get the biggest one
            else:
                pos = list(self.QtableW[currentState].values()).index(maximum)

            action = list(self.QtableW[currentState].keys())[pos]
        return action


    # This function returns the path to solution, ussing the Q table
    def constructPath(self, currentState, turn):
        print("'hello")
        state = currentState
        path = [state]

        # If whites turn
        if turn:
            while not self.isCheckMate1(state):
                print(state)
                next = self.getMaximumFuture(self.stateToTuple(state),True)
                state = []
                for element in next:
                    state.append(list(element))
                path.append(state)

        return path


    # Q-learing function for question 1
    def Qlearing1(self,currentState):

        loss = 1 # This variable will allow us to evaluate the accuracy of the predictions
        iterations = 0 # This variable will count each episode we iterate through. One episode ends when it arrives to a Checkamte state

        # We iterate through a maximum number of episodes. If our predictions are already good, we stop the learing process
        while loss > 0 and iterations < 1000:

            state = currentState
            iterations += 1 # We increase the number of episodes

            # Each episode will last until we arrive to a check-mate state
            while not self.isCheckMate1(state):

                # We create a tuple version of the state to make it easier to operate with
                tupledState = self.stateToTuple(state)

                # If we haven't seen this state yet, we add it to the table and initialize it
                if tupledState not in self.QtableW:
                    self.initializeState(state, False)



                # We decide if we do exploitation or exploration
                greedy = self.getGreedy()

                # If exploitation
                if greedy:
                    # We do the action that has a maximum Q
                    next = self.getMaximumFuture(tupledState, True)

                # If exploration
                else:
                    # We get a random action from the possible future ones
                    pos = np.random.choice(range(0, len(self.QtableW[tupledState])))
                    next = list(self.QtableW[tupledState].keys())[pos]

                listNext = []
                for element in next:
                    listNext.append(list(element))

                # We get the reward of the current state
                reward = self.getReward1(listNext)
                maximum = self.getMaximumFuture(next, True)

                # We get the mistake of our prediction
                sample = reward + self.gamma * self.QtableW[next][maximum]
                loss = sample - self.QtableW[tupledState][next]

                # We update the Q table and frequency table
                self.freq_W[tupledState][next] += 1
                self.QtableW[tupledState][next] += 0.3 * loss

                # We update the current state
                state = listNext
                print(":)", loss)

        # Once we have updated our Q table, we reconstruct the path to the checkmate
        path = self.constructPath(currentState, True)
        print('Path to Check-mate: ', path)
        print(' Number of movements: ', len(path)-1)


def translate(s):
    """
    Translates traditional board coordinates of chess into list indices
    """

    try:
        row = int(s[0])
        col = s[1]
        if row < 1 or row > 8:
            print(s[0] + "is not in the range from 1 - 8")
            return None
        if col < 'a' or col > 'h':
            print(s[1] + "is not in the range from a - h")
            return None
        dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        return (8 - row, dict[col])
    except:
        print(s + "is not in the format '[number][letter]'")
        return None







if __name__ == "__main__":

    # intiialize board
    TA = np.zeros((8, 8))
    # white pieces
    TA[7][0] = 2
    TA[7][4] = 6
    # black pieces
    TA[0][4] = 12

    # initialise board
    print("stating AI chess... ")
    aichess = Aichess(TA, True)

    print("printing board")
    aichess.chess.boardSim.print_board()

    # get list of next states for current state
    print("current State",aichess.currentStateW)
    aichess.getListNextStatesW(aichess.currentStateW)
    print("list next states ",aichess.listNextStates)


    print('Q-Learning case 1');
    aichess.Qlearing1(aichess.currentStateW);
    # starting from current state find the end state (check mate) - recursive function
    aichess.chess.boardSim.listVisitedStates = []
    # find the shortest path, initial depth 0
#    depth = 0

    MovesToMake = ['1e','2e','2e','3e','3e','4d','4d','3c']

    for k in range(int(len(MovesToMake)/2)):

        print("k: ",k)

        print("start: ",MovesToMake[2*k])
        print("to: ",MovesToMake[2*k+1])

        start = translate(MovesToMake[2*k])
        to = translate(MovesToMake[2*k+1])

        print("start: ",start)
        print("to: ",to)

        aichess.chess.moveSim(start, to)


    aichess.chess.boardSim.print_board()
    print("#Move sequence...  ")

    print("#Current State...  ", aichess.chess.board.currentStateW)
