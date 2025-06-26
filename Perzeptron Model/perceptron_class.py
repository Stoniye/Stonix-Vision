import random

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.01):
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(num_inputs)]
        self.bias = random.uniform(-0.5, 0.5)
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias #calculate possibility
        return 1 if summation >= 0 else 0 #return prediction

    def train(self, training_data, epochs):
        for epoch in range(epochs): #iterate through every epoch
            total_error = 0
            for inputs, target_output in training_data: #iterate trough every data in training data

                prediction = self.predict(inputs) #try to predict
                error = target_output - prediction #calculate loss (error)
                total_error += abs(error)

                if error != 0:
                    #tweak weight and bias based on the loss (error)
                    for i in range(len(self.weights)):
                        self.weights[i] += self.learning_rate * error * inputs[i]
                    self.bias += self.learning_rate * error


            #ouput epoch stats
            if epoch % (epochs // 10 if epochs >= 10 else 1) == 0:
                print(f"Epoch {epoch+1}/{epochs}, Total Error: {total_error}")
            if total_error == 0:
                print(f"Finished after {epoch+1} Epoch.")
                break