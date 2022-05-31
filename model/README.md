# Generator script

Airfoil shapes can be generated using the optimized model through the *vae_generator.py* script.  
Python package requirements can be found in *requirements.txt*.  

## Usage
1) Modify *input_latent.dat* with 4 values corresponding to the 4 latent variables of the model. These can be modified freely, but one must take into account that the bigger the absolute value of the latent variables, the less coherent the airfoils will tend to be. Good starting points are values between -2 and 2 approximately, with smaller values being more conservative and bigger values stretching the possibilities of the generator.
2) Run *vae_generator.py*. This will generate a file called *output_airfoil.dat* that contains 201 chord-normalized x-y coordinate pairs with cosine spacing in the x axis for the target airfoil.  

Model parameters can be changed inside *./params/model_parameters.json*. However, this will prompt the generator script to look for a different model checkpoint inside *./model/*, which will not be trained. In order to use it, it must be trained and placed in this folder beforehand. *Scaler_dict.py* contains the bounds of the original dataset scaler, since the network itself outputs normalized values that have to be de-normalized. Do not modify.

