import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

def dnn_clean_data(df, random_state):
    df.drop('agecode', axis=1, inplace=True)
    targets = ['Insominia', 'shizopherania', 'vascula_demetia', 'ADHD', 'Bipolar']
    features = [x for x in df.columns if x not in targets]
    for col in targets:
        df[col] = df[col].str[:1].map({'P': 1, 'N': 0})
        
    ode = OrdinalEncoder()
    categ_np = ode.fit_transform(df[features].drop('age', axis=1))
    
    X = np.append(categ_np, np.array(df['age']).reshape((500, 1)), 1)
    y = np.array(df[targets])
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=random_state)

    return X_train, X_val, y_train, y_val

def dnn_model(X_train, X_val, y_train, y_val, multilabel = True, single_pred = None):
	
    model = Sequential()
    model.add(Dense(20, input_shape=(X_train.shape[1], ), activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(20, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(40, activation="relu"))
    model.add(Dropout(0.1))
    
    if multilabel:
        model.add(Dense(5, activation="sigmoid"))
    else:
        model.add(Dense(1, activation="sigmoid"))
    
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    if multilabel:
        history = model.fit(X_train, y_train, epochs=40, validation_data=(X_val, y_val), verbose=0)
        
        pred = model.predict(X_train, verbose = 0)
        train_accuracy = np.sum(np.round(pred) == y_train) / (len(y_train)*5)
        
        pred = model.predict(X_val, verbose = 0)
        val_accuracy = np.sum(np.round(pred) == y_val) / (len(y_val)*5)
        return train_accuracy, val_accuracy
        
    else:
        target_dict = {'insomnia': 0, 'schizophrenia': 1, 'vascular_demetia': 2, 'adhd': 3, 'bipolar': 4}
        col = target_dict[single_pred]
        history = model.fit(X_train, y_train[:, col], epochs=40, validation_data=(X_val, y_val[:, col]), verbose = 0)

        pred = model.predict(X_train, verbose=0)
        train_accuracy = (np.round(pred) == y_train[:, col].reshape((-1, 1))).sum()/len(y_train)

        pred = model.predict(X_val, verbose=0)
        val_accuracy = (np.round(pred) == y_val[:, col].reshape((-1, 1))).sum()/len(y_val)
        return train_accuracy, val_accuracy
