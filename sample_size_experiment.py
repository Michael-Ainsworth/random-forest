from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD, Adam
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping

__all__ = ['random_forest_classifier_model', 
        'parametric_deep_neural_network',
        'parametric_convolutional_neural_network']

def random_forest_classifier_model(X_train, y_train, num_trees, max_depth, verbose):
        
    rf_model = RandomForestClassifier(n_estimators = num_trees, 
                                    max_depth = None, 
                                    verbose = verbose
                                    )
    rf_model.fit(X_train, y_train)
    
    return rf_model



def parametric_deep_neural_network(X_train, y_train, epochs, batch_size, 
                                learning_rate, validation_split, 
                                verbose
                                ):

    dnn_model = Sequential()

    dnn_model.add(Dense(8, activation = 'relu'))
    dnn_model.add(Dense(8, activation = 'relu'))
    dnn_model.add(Dense(units = 1, activation = 'sigmoid'))

    # sgd_optimizer = SGD(lr = 0.001, momentum = 0.9)
    adam_optimizer = Adam(learning_rate=learning_rate)
    dnn_model.compile(optimizer = adam_optimizer, 
                    loss = 'binary_crossentropy', 
                    metrics = ['accuracy']
                    )

    dnn_model.fit(x = X_train,
                 y = y_train,
                 epochs = epochs,
                 batch_size = batch_size,
                 validation_split = validation_split,
                 verbose = verbose
                 )

    # Plotting functionality to visualize training loss and validation loss
#     losses = pd.DataFrame(dnn_model.history.history)
#     losses[['loss','val_loss']].plot(figsize = (10,7))
#     plt.title('Training Loss and Validation Loss over each Epoch')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')

    return dnn_model



def parametric_convolutional_neural_network(X_train, y_train, complexity, epochs, 
                                        batch_size, learning_rate, 
                                        validation_split, verbose
                                        ):
    
    cnn_model = Sequential()
    
    cnn_model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (32,32,3)))
    cnn_model.add(Conv2D(64, (3, 3), activation = 'relu'))
    cnn_model.add(MaxPooling2D((2,2)))

    if complexity > 2:
        for _ in range(1, complexity-1):
            cnn_model.add(Conv2D(64, (3, 3), activation = 'relu'))
            
        cnn_model.add(MaxPooling2D((2,2)))

    cnn_model.add(Flatten())
    cnn_model.add(Dense(64, activation = 'relu'))
    cnn_model.add(Dense(1, activation = 'sigmoid'))
    
    adam_optimizer = Adam(learning_rate=learning_rate)
#     sgd = SGD(learning_rate=learning_rate)
    cnn_model.compile(optimizer = adam_optimizer, 
                    loss = 'binary_crossentropy', 
                    metrics = ['accuracy']
                    )
    
    cnn_model.fit(x = X_train,
                 y = y_train,
                 epochs = epochs,
                 batch_size = batch_size,
                 validation_split = validation_split,
                 verbose = verbose
                 )
    
    return cnn_model