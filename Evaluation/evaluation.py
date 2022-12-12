import logging
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras.layers


# configuring logging operations
logging.basicConfig(filename='development_logs.log', level=logging.INFO,format='%(levelname)s:%(asctime)s:%(message)s')

class Metrics:
    """
        This class is used to evaluate the models by returning their performance metrics.
        Author:
            Vikas
    """

    def __init__(self):
        logging.info('Entered in "Metrics" class.')
        pass

    def accuracy_and_loss(self, features_test, labels_test, model):
        """
            This function will evaluate the model on the test set.
            Parameters:
                features_test :
                labels_test :
            Returns:
                    accuracy : accuracy of model.
                    loss : Loss of model.
        """
        logging.info('Entered the "accuracy_and_loss" method of the "Metrics" class.')  # logging operation

        try:
            self.features_test = features_test
            self.labels_test = labels_test
            # Evaluate the trained model.
            model_evaluation_history = model.evaluate(self.features_test, self.labels_test)
            # Get the loss and accuracy from model_evaluation_history.
            model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
            # Display the accuracy and loss of model.
            logging.info("\nAccuracy of model : " + str(model_evaluation_accuracy))
            logging.info("\nLoss of model : " + str(model_evaluation_loss))

            logging.info('Exited the "accuracy_and_loss" method of the "Metrics" class.')  # logging operation
            return model_evaluation_loss, model_evaluation_accuracy

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in "accuracy_and_loss" method of the "Metrics" class. Exception message:' + str(
                e))

            logging.info(
                '"accuracy_and_loss" operation is unsuccessful.Exited the "accuracy_and_loss" method of the "Metrics" class.')

    def plot_metric(self,model_training_history, metric_name_1, metric_name_2, plot_name):
        """
            This function will plot the metrics passed to it in a graph.
            Args:
                model_training_history: A history object containing a record of training and validation loss values and metrics values at successive epochs
                metric_name_1: The name of the first metric that needs to be plotted in the graph.
                metric_name_2: The name of the second metric that needs to be plotted in the graph.
                plot_name: The title of the graph.

        """
        logging.info('Entered the "plot_metric" method of the "Metrics" class.')  # logging operation
        try:
            # Get metric values using metric names as identifiers.
            metric_value_1 = model_training_history.history[metric_name_1]
            metric_value_2 = model_training_history.history[metric_name_2]

            # Construct a range object which will be used as x-axis (horizontal plane) of the graph.
            epochs = range(len(metric_value_1))

            # Plot the Graph.
            plt.plot(epochs, metric_value_1, 'blue', label=metric_name_1)
            plt.plot(epochs, metric_value_2, 'red', label=metric_name_2)

            # Add title to the plot.
            plt.title(str(plot_name))

            # Add legend to the plot.
            plt.legend()
            logging.info('Exited the "accuracy_and_loss" method of the "Metrics" class.')  # logging operation
            return

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in "accuracy_and_loss" method of the "Metrics" class. Exception message:' + str(
            e))

            logging.info(
                '"accuracy_and_loss" operation is unsuccessful.Exited the "accuracy_and_loss" method of the "Metrics" class.')
