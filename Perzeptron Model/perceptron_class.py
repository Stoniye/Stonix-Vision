import random

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.01):
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(num_inputs)]
        self.bias = random.uniform(-0.5, 0.5)
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return 1 if summation >= 0 else 0

    def train(self, training_data, epochs):
        for epoch in range(epochs):
            total_error = 0
            for inputs, target_output in training_data:
                normalized_inputs = (inputs[0] / 255.0, inputs[1] / 255.0)

                prediction = self.predict(normalized_inputs)
                error = target_output - prediction
                total_error += abs(error)

                if error != 0:
                    for i in range(len(self.weights)):
                        self.weights[i] += self.learning_rate * error * normalized_inputs[i]
                    self.bias += self.learning_rate * error

            if epoch % (epochs // 10 if epochs >= 10 else 1) == 0:
                print(f"Epoch {epoch+1}/{epochs}, Total Error: {total_error}")
            if total_error == 0:
                print(f"Finished after {epoch+1} Epoch.")
                break