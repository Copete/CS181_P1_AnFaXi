# Necessary modules and functions to run DL model on P1 data

# General modules
import numpy as np
# Deep learning modules from keras library
from keras.layers import Dense # For dense layers
from keras.models import Sequential # For sequential layering
from keras.callbacks import EarlyStopping # For stopping execution
from sklearn.metrics import mean_squared_error


# Function to train multi-layered neural network of a given number of nodes
def train_model_DL(X_train,Y_train,n_nodes):
    """ n_nodes is 1-D numpy array with number of nodes on each layer
        e.g. [10,20,30] is a model with 3 (hidden) layers,
        with 10/20/30 nodes on the first/second/third layers
        Returns trained DL model """
    input_shape = (X_train.shape[1],) # Shape of input data
    # Initialize model
    model_DL = Sequential()
    for i in range(len(n_nodes)):
        if i == 0:
            # First layer
            model_DL.add(Dense(n_nodes[i],activation='relu',input_shape=input_shape))
        else:
            # Subsequent layers
            model_DL.add(Dense(n_nodes[i],activation='relu'))
    # Output layer
    model_DL.add(Dense(1))
    # Compile model
    model_DL.compile(optimizer='adam',loss='mean_squared_error')
    # Print model summary
    model_DL.summary()
    # Early stopping monitor w/ patience=3 (stop after 3 runs without improvements)
    early_stopping_monitor = EarlyStopping(patience=3)
    # Fit model using 20% of data for validation
    model_DL.fit(X_train, Y_train, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])
    Y_train_DLpred = model_DL.predict(X_train)
    mse_DL = mean_squared_error(Y_train, Y_train_DLpred)
    print('DONE. Mean Squared Error: ', mse_DL)
    return model_DL

"""
## SAMPLE USE OF TRAIN_MODEL_DL
# Train with 1 layer, same nodes as input
n_nodes = np.array([[X_train.shape[1]]])
model_DL = []
for n in n_nodes:
    model_DL = train_model_DL(X_train,Y_train,n)

## TO SAVE TRAINED MODEL
model_DL.save('model_DL.h5')

## TO PREDICT ON TEST DATA AND WRITE RESULTS TO CSV FILE
Y_test_pred  = model_DL.predict(X_test)
def write_to_file(filename, predictions):
    # Function to write predictions to CSV file
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")
write_to_file("DL_pred.csv", Y_test_pred)
"""
