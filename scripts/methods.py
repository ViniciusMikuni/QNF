import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import h5py as h5
import re
from tensorflow import keras
tfd = tfp.distributions
tfb = tfp.bijectors


def _network(data_shape,cond_shape):
    made = tfb.AutoregressiveNetwork(params=2, 
                                     hidden_units=[32,32], 
                                     event_shape=data_shape,
                                     activation='leaky_relu',
                                     conditional=True,
                                     conditional_event_shape=cond_shape,
                                    )
    return made

def make_bijector_kwargs(bijector, name_to_kwargs):
    #Hack to pass the conditional information through all the bijector layers
    if hasattr(bijector, 'bijectors'):
        return {b.name: make_bijector_kwargs(b, name_to_kwargs) for b in bijector.bijectors}
    else:
        for name_regex, kwargs in name_to_kwargs.items():
            if re.match(name_regex, bijector.name):
                return kwargs
    return {}

def MADE(data_shape, cond_shape,ntransform=8):
    # Density estimation with MADE.
    
    bijectors = []
    for i in range(ntransform):        
        bijectors.append(tfb.MaskedAutoregressiveFlow(_network(data_shape,cond_shape),name='made{}'.format(i)))
    chain = tfb.Chain(bijectors)
    distribution = tfd.TransformedDistribution(
        #distribution=tfd.Sample(tfd.Uniform(), sample_shape=[data_shape]),
        distribution=tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[data_shape]),
        bijector=chain)

    # Construct and fit model.
    x_ = keras.layers.Input(shape=(data_shape,), dtype=tf.float32)
    c_ = keras.layers.Input(shape=(cond_shape,), dtype=tf.float32)
    bijector_kwargs = make_bijector_kwargs(
                distribution.bijector, {'made.': {'conditional_input': c_}})
    logprob_ = distribution.log_prob(x_, bijector_kwargs=bijector_kwargs)
    model = keras.Model([x_,c_], logprob_)

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=5e-4),loss=lambda _, log_prob: -log_prob)
    
    return model, distribution

def DataLoaderCaloGAN(file_name,nevts=-1):
    '''
    Inputs:
    - name of the file to load
    - number of events to use
    Outputs:
    - Generated particle energy (nevts,1)
    - Energy deposition in each layer (nevts,3)
    '''
    
    with h5.File(file_name,"r") as h5f:
        if nevts <0:
            nevts = len(h5f['energy'])
        e = h5f['energy'][:].astype(np.float32)
        layer0= h5f['layer_0'][:].astype(np.float32)/1000.0
        layer1= h5f['layer_1'][:].astype(np.float32)/1000.0
        layer2= h5f['layer_2'][:].astype(np.float32)/1000.0


    layer_energies = [np.sum(layer,(1,2)) for layer in [layer0,layer1,layer2]]
    layer_energies = np.transpose(layer_energies)

    return e/100.,np.ma.log(layer_energies/e + 1.0).filled(0)
