import matplotlib.pyplot as plt

# plot the curves separately
def plot_curve(history):
    loss = history["loss"]
    val_loss = history["val_loss"]
    accuracy = history["accuracy"]
    val_accuracy = history["val_accuracy"]
    
    epochs = range(len(history["loss"]))
    
    plt.figure()
    plt.plot(epochs, loss, label="training_loss", )
    plt.plot(epochs, val_loss, label="validation_loss", linestyle='dashed') 
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()
    
    plt.figure()
    plt.plot(epochs, accuracy, label="training_accuracy")
    plt.plot(epochs, val_accuracy, label="validation_accuracy", linestyle='dashed') 
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.legend()