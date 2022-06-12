import numpy as np
from scipy.signal import savgol_filter

def smoothen(airfoil, window_length=11, polyorder=3):
    """
    Smooths the airfoil coordinates using a Savitzky-Golay filter.
    Inputs:
    - airfoil: numpy array with the airfoil coordinates
    """
    
    airfoil_smooth = savgol_filter(airfoil, window_length, polyorder)
    return airfoil_smooth