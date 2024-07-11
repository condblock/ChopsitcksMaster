import pytest
import numpy as np
from main import getSingleTheta

def test_getSingleTheta():
    state = np.array([[1, 1], [1, 1]])


    assert getSingleTheta(state) == np.array([[1,1,1,1,1,0,0,0,0,0,0,0]])