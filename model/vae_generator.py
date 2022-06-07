import numpy as np
import time
import json
import torch
from model.model_utils import load_model, decode_latent
from utils.io import read_latent, save_airfoil
from utils.scale import denormalize

print('--- VAE Airfoil Generator ---\n')

# Load input parameters and model
start_time = time.time()
try:
    with open('./params/model_parameters.json', 'r') as f:
        parameters = json.load(f)
    with open('./params/scaler_dict.json', 'r') as fp:
        scaler_bounds = json.load(fp)   
except:
    print('There was an error loading model parameters. Check your inputs.')
print(parameters)
model = load_model(parameters)

# Decode and transform the input latent variables
latent_airfoil = torch.Tensor(read_latent('input_latent.dat'))
airfoil_coords = decode_latent(latent_airfoil, model, device = parameters['device'])
airfoil_denorm = denormalize(airfoil_coords, scaler_bounds)
save_airfoil(airfoil_denorm, 'output_airfoil.dat')

print(f'Elapsed time: {np.round(time.time() - start_time, 4)} s')
    