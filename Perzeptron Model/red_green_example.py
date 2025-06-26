import random
import perceptron_class as perceptron

def save_weights_and_bias(model, filename):
    with open(filename, 'w') as f:
        f.write(f"{model.bias}\n")
        for weight in model.weights:
            f.write(f"{weight}\n")

def load_weights_and_bias(model, filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            if not lines:
                print(f"Error: {filename} is empty")
                return False

            model.bias = float(lines[0].strip())

            loaded_weights = []
            for line in lines[1:]:
                loaded_weights.append(float(line.strip()))

            if len(loaded_weights) != len(model.weights):
                print(f"Warning: Loaded weight ({len(loaded_weights)}) do not match with model weights ({len(model.weights)})!")
            model.weights = loaded_weights

        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def generate_pixel_data(num_samples):
    data = []

    for _ in range(num_samples):
        r = 0
        g = 0
        color = 0
        if random.randint(0, 1) == 1:
            g = random.randint(150, 255)
            r = random.randint(0, 100)
            color = 0
        else:
            r = random.randint(150, 255)
            g = random.randint(0, 100)
            color = 1
        data.append(((r / 255.0, g / 255.0), color))

    random.shuffle(data)
    return data

if __name__ == "__main__":
    training_data = generate_pixel_data(num_samples=200) #generate random pixel for training data

    stonix = perceptron.Perceptron(num_inputs=2, learning_rate=0.05) #initialize model

    #load parameters or train new
    if not load_weights_and_bias(stonix, filename="red_green_model"):
        print("Start training")
        stonix.train(training_data, epochs=100)
        print("Finished training")
        save_weights_and_bias(stonix, filename="red_green_model")
    else:
        print("Loaded model Parameters")

    print("\nStarting Test")

    #Test model on random pixels
    test_num = 5

    for _ in range(test_num):
        r = 0
        g = 0
        color = 0
        if random.randint(0, 1) == 1:
            g = random.randint(150, 255)
            r = random.randint(0, 100)
            color = 0
        else:
            r = random.randint(150, 255)
            g = random.randint(0, 100)
            color = 1

        pixel = (r / 255.0, g / 255.0)
        prediction = stonix.predict(pixel)
        print(f"Color: (R={pixel[0]}, G={pixel[1]}) > Prediction: {'Red' if prediction == 1 else 'Green'}  (Expected: {'Red' if color == 1 else 'Green'})")