def normalize(data, scaler):
    """
    Normalizes numpy array data using a pre-fitted scaler saved in scaler_dict.json as a dictionary.
    Inputs:
        - data: data to be normalized. [np.array]
        - scaler: scaler bounds for the normalization (min, max for every feature). [dict]
    Outputs:
        - normalized data. [np.array]
    """
    
    data_norm = data

    for idx, key in enumerate(scaler.keys()):
        data_norm[idx] = (data[idx] - scaler[key]['min']) / (scaler[key]['max'] - scaler[key]['min'])
        idx += 1
    return data_norm

def denormalize(data, scaler):
    """
    Denormalizes numpy array data using a pre-fitted scaler saved in scaler_dict.json as a dictionary.
    Inputs:
        - data: data to be denormalized. [np.array]
        - scaler: scaler bounds for the normalization (min, max for every feature). [dict]
    Outputs:
        - normalized data. [np.array]
    """

    data_denorm = data
    
    for idx, key in enumerate(scaler.keys()):
        data_denorm[idx] = data[idx] * (scaler[key]['max'] - scaler[key]['min']) + scaler[key]['min']

    return data_denorm