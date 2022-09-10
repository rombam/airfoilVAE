import numpy as np

def read_latent(filename):
    """
    Reads an airfoil's latent variables from a .dat file.
    Inputs:
    - filename: string of the filename to read the airfoil latent variables from
    Outputs:    
    - latent: numpy array with the latent variables
    """
    with open(filename, 'r') as datfile:
        print(f'Reading airfoil from: {filename}')
        latent = np.loadtxt(datfile, unpack = True)
    return latent

def read_airfoil(filename):
    """
    Read an airfoil coordinates from a .dat file.
    Inputs:
    - filename: string of the filename to read the airfoil from
    Outputs:    
    - airfoil: numpy array with the airfoil coordinates [np.array]
    """
    with open(filename, 'r') as datfile:
        print(f'Reading airfoil coordinates from: {filename}')
        airfoil = np.loadtxt(datfile, unpack = True)
        # airfoil[0] -> x coordinates
        # airfoil[1] -> y coordinates
    return airfoil

def save_airfoil(airfoil, filename, n_points = 198):
    """
    Saves an airfoil's x and y coordinates to a .dat file. Uses cosine spacing.
    Inputs:
    - airfoil: numpy array of Y airfoil coordinates
    - filename: string of the filename to save the airfoil to
    """

    # X: cosine spacing
    points_per_surf = int(n_points/2)
    x = list(reversed([0.5*(1-np.cos(ang)) for ang in np.linspace(0,np.pi,points_per_surf+2)]))
    aux_x = list([0.5*(1-np.cos(ang)) for ang in np.linspace(0,np.pi,points_per_surf+2)[1:points_per_surf+1]])
    [x.append(i) for i in aux_x]
    x.append(1.0)
    
    # Y
    y = []
    origin = (airfoil[0] + airfoil[points_per_surf])/2
    y.append(0.0)
    [y.append(j) for j in airfoil[0:points_per_surf].tolist()]
    y.append(origin)
    aux_y = list(airfoil[points_per_surf:n_points].tolist())
    [y.append(k) for k in aux_y]
    y.append(0.0)
    
    with open(filename, 'w', newline='') as datfile:
        for i in range(len(x)):
            print(f'{x[i]:.8E} {y[i]:.8E}', file=datfile)
    
    print('Airfoil saved successfully!')