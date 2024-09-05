import numpy as np


def rc(o, h):
    return 2*o*h + h*h

def unitary_rnn(o, h):
    return 2*o*h + h*h 

def lstm(o, h, l):
    return 4*l*((o+h)*h + h) + o*h

def gru(o, h, l):
    return 3*l*((o+h)*h + h) + o*h

def rfm(o, h):
    return 2*o*h + h