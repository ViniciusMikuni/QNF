import methods
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
import argparse
import h5py as h5
import gc

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/SCRATCH/SGM/gamma.hdf5', help='Folder containing files')
    flags = parser.parse_args()

    e,layer_energies = methods.DataLoaderCaloGAN(flags.data_folder)
    model,dist = methods.MADE(e.shape[1],layer_energies.shape[1])
    batch_size = 1024
    myhistory = model.fit([e,layer_energies],
                          y=np.ones((len(e),0), dtype=np.float32), #dummy labels 
                          batch_size=batch_size,
                          epochs=100,
                          verbose = 1)

    # print(dist.bijector.forward([1.0],conditional_input= [0.0,0.0,0.0]).numpy())
    # print(dist.log_prob([0.5],bijector_kwargs={'conditional_input': [0.0,0.0,0.0]}))


    
    
