#!/usr/bin/env python
# coding: utf-8

# # IE 8990 Course Project

# ## Importing library

# In[1]:


# General Libraries #########################

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# In[ ]:





# In[2]:


# Loading GPU ##############################

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# ## Custom Activation Function

# In[3]:


## Custom Activation Function

from tensorflow.keras import backend as K

def Swish(inputs):
    return inputs*K.sigmoid(inputs)
    #return K.maximum(inputs,0)
    #return K.minimum(K.maximum((K.sign(inputs)*K.pow(K.abs(inputs),1/3)),0),
    #inputs*K.sigmoid(inputs))
    #return K.maximum(K.softsign(inputs),0) + K.minimum(inputs*K.sigmoid(inputs), 
    #0.05*x*K.sigmoid(inputs))


def CustFunc(inputs,p=0.00):
    return K.maximum(K.softsign(inputs),0) + K.minimum(inputs*K.sigmoid(inputs),
                                                       p*inputs*K.sigmoid(inputs))
    #return K.maximum(K.softsign(inputs),0) + K.minimum(inputs*K.sigmoid(inputs), 0)
    
def CustFunc05(inputs,p=0.05):
    return K.maximum(K.softsign(inputs),0) + K.minimum(inputs*K.sigmoid(inputs),
                                                       p*inputs*K.sigmoid(inputs))
    #return K.maximum(K.softsign(inputs),0) + K.minimum(inputs*K.sigmoid(inputs), 0)
def CustFunc15(inputs,p=0.15):
    return K.maximum(K.softsign(inputs),0) + K.minimum(inputs*K.sigmoid(inputs),
                                                       p*inputs*K.sigmoid(inputs))
    #return K.maximum(K.softsign(inputs),0) + K.minimum(inputs*K.sigmoid(inputs), 0)
def CustFunc25(inputs,p=0.25):
    return K.maximum(K.softsign(inputs),0) + K.minimum(inputs*K.sigmoid(inputs),
                                                       p*inputs*K.sigmoid(inputs))
    #return K.maximum(K.softsign(inputs),0) + K.minimum(inputs*K.sigmoid(inputs), 0)
def CustFunc35(inputs,p=0.35):
    return K.maximum(K.softsign(inputs),0) + K.minimum(inputs*K.sigmoid(inputs),
                                                       p*inputs*K.sigmoid(inputs))
    #return K.maximum(K.softsign(inputs),0) + K.minimum(inputs*K.sigmoid(inputs), 0)
def CustFunc45(inputs,p=0.45):
    return K.maximum(K.softsign(inputs),0) + K.minimum(inputs*K.sigmoid(inputs),
                                                       p*inputs*K.sigmoid(inputs))
    #return K.maximum(K.softsign(inputs),0) + K.minimum(inputs*K.sigmoid(inputs), 0)

    
def CustFunc1(inputs):
    #return K.maximum(K.softsign(inputs),0) + K.minimum(inputs*K.sigmoid(inputs),
    #0.05*inputs*K.sigmoid(inputs))
    return K.maximum(K.softsign(inputs),0) + K.minimum(inputs*K.sigmoid(inputs), 0)


# In[4]:


# Plotting the Custom Function


### Setting the Coordinate Space

x = np.linspace(-5, 5, 100)


# In[5]:


### Several Combinations of Mixed Function

X = [0, .1, .25,.33,.5,.67,.75,.90,1]

for i in X: 
    plt.plot(x,CustFunc(x,i), label="p = {}".format(i))
    
plt.legend()
plt.show()


# ## Importing Data

# In[6]:


# Loading Datasets ########################

from tensorflow.keras.datasets import mnist


# In[7]:


# Setting the training and test variables

## This loads the data row by row to make it a time series based dataset.

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0


# ## LSTM Model

# ### Base Model

# In[8]:


# Making the model


# Input Layer
model = keras.Sequential()
model.add(keras.Input(shape = (None, 28))) #28 pixels per time step

# Hidden Layers
model.add(
    layers.LSTM(256, return_sequences = True, activation = 'tanh', 
                recurrent_activation='sigmoid'))
model.add(
    layers.LSTM(256, activation = 'tanh',recurrent_activation='sigmoid'))

# Output Layer
model.add(layers.Dense(10))

# Model Summary
print(model.summary())


# ### Reference for LSTM Model
# 
# def __init__(self,
#                units,
#                activation='tanh',
#                recurrent_activation='sigmoid',
#                use_bias=True,
#                kernel_initializer='glorot_uniform',
#                recurrent_initializer='orthogonal',
#                bias_initializer='zeros',
#                unit_forget_bias=True,
#                kernel_regularizer=None,
#                recurrent_regularizer=None,
#                bias_regularizer=None,
#                kernel_constraint=None,
#                recurrent_constraint=None,
#                bias_constraint=None,
#                dropout=0.,
#                recurrent_dropout=0.,
#                *'*kwargs):
#     super(LSTMCell, self).__init__(
#         units,
#         activation=activation,
#         recurrent_activation=recurrent_activation,
#         use_bias=use_bias,
#         kernel_initializer=kernel_initializer,
#         recurrent_initializer=recurrent_initializer,
#         bias_initializer=bias_initializer,
#         unit_forget_bias=unit_forget_bias,
#         kernel_regularizer=kernel_regularizer,
#         recurrent_regularizer=recurrent_regularizer,
#         bias_regularizer=bias_regularizer,
#         kernel_constraint=kernel_constraint,
#         recurrent_constraint=recurrent_constraint,
#         bias_constraint=bias_constraint,
#         dropout=dropout,
#         recurrent_dropout=recurrent_dropout,
#         implementation=kwargs.pop('implementation', 2),
#         **kwargs)

# In[9]:


# Compiling the Model


model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)


# In[10]:



# Fitting the Model
history = model.fit(x_train, y_train, batch_size = 64, epochs = 10, verbose = 2, )


# Evaluate the Model

model.evaluate(x_test, y_test, batch_size = 64, verbose = 2)


# In[11]:


loss_base = history.history['loss']
acc_base = history.history['accuracy']
epochs = range(1,11)
plt.plot(epochs, loss_base, 'g', label='loss')
plt.plot(epochs, acc_base, 'b', label='accuracy')
plt.title('Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# ### Squish Function

# In[12]:


# Making the model


# Input Layer
modelAF = keras.Sequential()
modelAF.add(keras.Input(shape = (None, 28))) #28 pixels per time step

# Hidden Layers
modelAF.add(
    layers.LSTM(256, return_sequences = True, activation = CustFunc, 
                recurrent_activation='sigmoid'))
modelAF.add(
    layers.LSTM(256, activation = CustFunc,recurrent_activation='sigmoid'))

# Output Layer
modelAF.add(layers.Dense(10))

# Model Summary
print(modelAF.summary())


# In[13]:


# Compiling the Model


modelAF.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)


# Fitting the Model

history_AF = modelAF.fit(x_train, y_train, batch_size = 64, epochs = 10, verbose = 2)


# Evaluate the Model

modelAF.evaluate(x_test, y_test, batch_size = 64, verbose = 2)


# In[14]:


loss_AF = history_AF.history['loss']
acc_AF = history_AF.history['accuracy']


# In[15]:


# Making the model


# Input Layer
modelRAF = keras.Sequential()
modelRAF.add(keras.Input(shape = (None, 28))) #28 pixels per time step

# Hidden Layers
modelRAF.add(
    layers.LSTM(256, return_sequences = True, activation = 'tanh', 
                recurrent_activation= CustFunc))
modelRAF.add(
    layers.LSTM(256, activation = 'tanh',recurrent_activation = CustFunc))

# Output Layer
modelRAF.add(layers.Dense(10))

# Model Summary
print(modelRAF.summary())


# In[16]:


# Compiling the Model


modelRAF.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Fitting the Model

history_RAF = modelRAF.fit(x_train, y_train, batch_size = 64, epochs = 10, verbose = 2)


# Evaluate the Model

modelRAF.evaluate(x_test, y_test, batch_size = 64, verbose = 2)


# In[17]:


loss_RAF = history_RAF.history['loss']
acc_RAF = history_RAF.history['accuracy']


# ### ReLU function

# In[18]:


# Making the model


# Input Layer
modelR = keras.Sequential()
modelR.add(keras.Input(shape = (None, 28))) #28 pixels per time step

# Hidden Layers
modelR.add(
    layers.LSTM(256, return_sequences = True, activation = 'relu', 
                recurrent_activation='sigmoid'))
modelR.add(
    layers.LSTM(256, activation = 'relu',recurrent_activation='sigmoid'))

# Output Layer
modelR.add(layers.Dense(10))

# Model Summary
print(modelR.summary())


# In[19]:


# Compiling the Model


modelR.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Fitting the Model

historyR = modelR.fit(x_train, y_train, batch_size = 64, epochs = 10, verbose = 2)


# Evaluate the Model

modelR.evaluate(x_test, y_test, batch_size = 64, verbose = 2)


# In[20]:


loss_R = historyR.history['loss']
acc_R = historyR.history['accuracy']


# In[21]:


# Making the model


# Input Layer
modelRR = keras.Sequential()
modelRR.add(keras.Input(shape = (None, 28))) #28 pixels per time step

# Hidden Layers
modelRR.add(
    layers.LSTM(256, return_sequences = True, activation = 'tanh', 
                recurrent_activation='relu'))
modelRR.add(
    layers.LSTM(256, activation = 'tanh',recurrent_activation='relu'))

# Output Layer
modelRR.add(layers.Dense(10))

# Model Summary
print(modelRR.summary())


# In[22]:


# Compiling the Model


modelRR.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Fitting the Model

historyRR = modelRR.fit(x_train, y_train, batch_size = 64, epochs = 10, verbose = 2)


# Evaluate the Model

modelRR.evaluate(x_test, y_test, batch_size = 64, verbose = 2)


# In[23]:


loss_RR = historyRR.history['loss']
acc_RR = historyRR.history['accuracy']


# ### Swish Function

# In[25]:


# Making the model


# Input Layer
modelS = keras.Sequential()
modelS.add(keras.Input(shape = (None, 28))) #28 pixels per time step

# Hidden Layers
modelS.add(
    layers.LSTM(256, return_sequences = True, activation = Swish, 
                recurrent_activation='sigmoid'))
modelS.add(
    layers.LSTM(256, activation = Swish,recurrent_activation='sigmoid'))

# Output Layer
modelS.add(layers.Dense(10))

# Model Summary
print(modelS.summary())


# In[26]:


# Compiling the Model


modelS.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Fitting the Model

historyS = modelS.fit(x_train, y_train, batch_size = 64, epochs = 10, verbose = 2)


# Evaluate the Model

modelS.evaluate(x_test, y_test, batch_size = 64, verbose = 2)


# In[27]:


loss_S = historyS.history['loss']
acc_S = historyS.history['accuracy']


# In[28]:


# Making the model


# Input Layer
modelSR = keras.Sequential()
modelSR.add(keras.Input(shape = (None, 28))) #28 pixels per time step

# Hidden Layers
modelSR.add(
    layers.LSTM(256, return_sequences = True, activation = 'tanh', 
                recurrent_activation=Swish))
modelSR.add(
    layers.LSTM(256, activation = 'tanh',recurrent_activation=Swish))

# Output Layer
modelSR.add(layers.Dense(10))

# Model Summary
print(modelSR.summary())


# In[29]:


# Compiling the Model


modelSR.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Fitting the Model

historySR = modelSR.fit(x_train, y_train, batch_size = 64, epochs = 10, verbose = 2)


# Evaluate the Model

modelSR.evaluate(x_test, y_test, batch_size = 64, verbose = 2)


# In[30]:


loss_SR = historySR.history['loss']
acc_SR = historySR.history['accuracy']


# ### SoftMax Function

# In[31]:


# Making the model


# Input Layer
modelSM = keras.Sequential()
modelSM.add(keras.Input(shape = (None, 28))) #28 pixels per time step

# Hidden Layers
modelSM.add(
    layers.LSTM(256, return_sequences = True, activation = 'softmax', 
                recurrent_activation='sigmoid'))
modelSM.add(
    layers.LSTM(256, activation = 'softmax',recurrent_activation='sigmoid'))

# Output Layer
modelSM.add(layers.Dense(10))

# Model Summary
print(modelSM.summary())


# In[32]:


# Compiling the Model


modelSM.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Fitting the Model

historySM = modelSM.fit(x_train, y_train, batch_size = 64, epochs = 10, verbose = 2)


# Evaluate the Model

modelSM.evaluate(x_test, y_test, batch_size = 64, verbose = 2)


# In[33]:


loss_SM = historySM.history['loss']
acc_SM = historySM.history['accuracy']


# In[34]:


# Making the model


# Input Layer
modelSMR = keras.Sequential()
modelSMR.add(keras.Input(shape = (None, 28))) #28 pixels per time step

# Hidden Layers
modelSMR.add(
    layers.LSTM(256, return_sequences = True, activation = 'tanh', 
                recurrent_activation='softmax'))
modelSMR.add(
    layers.LSTM(256, activation = 'tanh',recurrent_activation='softmax'))

# Output Layer
modelSMR.add(layers.Dense(10))

# Model Summary
print(modelSM.summary())


# In[35]:


# Compiling the Model


modelSMR.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Fitting the Model

historySMR = modelSMR.fit(x_train, y_train, batch_size = 64, epochs = 10, verbose = 2)


# Evaluate the Model

modelSMR.evaluate(x_test, y_test, batch_size = 64, verbose = 2)


# In[36]:


loss_SMR = historySMR.history['loss']
acc_SMR = historySMR.history['accuracy']


# In[37]:


## Results Analysis


# In[41]:


# Plotting Accuracy


epochs = range(1,11)
plt.plot(epochs, acc_base, 'b', label='Base Model')
plt.plot(epochs, acc_AF, 'r-', label='Squish in Main')
plt.plot(epochs, acc_RAF, 'r--', label='Squish in Recurrent')
plt.plot(epochs, acc_R, 'g-', label='ReLU in Main')
plt.plot(epochs, acc_RR, 'g--', label='ReLU in Recurrent')
plt.plot(epochs, acc_S, 'b-', label='Swish in Main')
#plt.plot(epochs, acc_SR, 'b--', label='Swish in Recurrent')
#plt.plot(epochs, acc_SM, 'm-', label='SoftMax in Main')
#plt.plot(epochs, acc_SM, 'm--', label='SoftMax in Recurrent')
plt.title('Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[40]:


# Plotting Loss


epochs = range(1,11)
plt.plot(epochs, loss_base, 'b', label='Base Model')
plt.plot(epochs, loss_AF, 'r-', label='Squish in Main')
plt.plot(epochs, loss_RAF, 'r--', label='Squish in Recurrent')
plt.plot(epochs, loss_R, 'g-', label='ReLU in Main')
plt.plot(epochs, loss_RR, 'g--', label='ReLU in Recurrent')
plt.plot(epochs, loss_S, 'b-', label='Swish in Main')
#plt.plot(epochs, loss_SR, 'b--', label='Swish in Recurrent')
#plt.plot(epochs, loss_SM, 'm-', label='SoftMax in Main')
#plt.plot(epochs, loss_SM, 'm--', label='SoftMax in Recurrent')
plt.title('Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


### Loss Landscapes


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




