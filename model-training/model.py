import tensorflow as tf
import os
import optuna
from utils import plot_curve

def objective(trial):
    # Define the search space for hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-3)
    filters = trial.suggest_categorical("filters", [16, 32, 64])
    dense_units = trial.suggest_categorical("dense_units", [32, 64, 128])

    # Build the model with the suggested hyperparameters
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(filters, 3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, padding='valid'),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_units, activation='relu'),
        tf.keras.layers.Dense(26, activation='softmax')
    ])

    # Compile the model with the suggested learning rate
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        metrics=["accuracy"]
    )

    # Train the model
    history = model.fit(
        train_datax,
        epochs=5,
        steps_per_epoch=len(train_datax),
        validation_data=validation_data,
        validation_steps=len(validation_data)
    )

    # Return the validation accuracy as the objective value
    return history.history['val_accuracy'][-1]



valid_dir = "/content/testData"
train_dirx = "/content/datax7"

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                              rotation_range=0.2,
                                                              shear_range=0.2,
                                                              zoom_range=0.2,
                                                              width_shift_range=0.2,
                                                              height_shift_range=0.3,
                                                              horizontal_flip=True)

validation_data = datagen.flow_from_directory(directory=valid_dir,
                                              batch_size=32,
                                              target_size=(224,224),
                                              class_mode="categorical",
                                              shuffle=True,
                                              seed=42)

train_datax = datagen.flow_from_directory(directory=train_dirx,
                                              batch_size=32,
                                              target_size=(224,224),
                                              class_mode="categorical",
                                              shuffle=True,
                                              seed=42)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial
print("Validation accuracy: {}".format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

modelx = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(filters = trail.params['filters'],
                           kernel_size = (3,3),
                           activation = 'relu',
                           input_shape = (224,224,3)),
    
    tf.keras.layers.Conv2D(filters = trail.params['filters'], kernel_size = (3,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size = 2,
                              padding = 'valid'),
    
    tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size = 2,
                              padding = 'valid'),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(trail.params['dense_units'], activation = 'relu'),
    tf.keras.layers.Dense(26, activation = 'softmax')
])

modelx.compile(loss = "categorical_crossentropy",
              optimizer = tf.keras.optimizers.RMSprop(trail.params['learning_rate']),
              metrics = ["accuracy"])

historyx = modelx.fit(train_datax,
                   epochs = 50,
                   steps_per_epoch = len(train_datax),
                   validation_data = validation_data,
                   validation_steps = len(validation_data))

modelx.save("sign_model_with_1_optuna")

plot_curve(historyx.history)
