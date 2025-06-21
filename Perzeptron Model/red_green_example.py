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
    for _ in range(num_samples // 2):
        g = random.randint(150, 255)
        r = random.randint(0, 100)
        data.append(((r, g), 0))

    for _ in range(num_samples // 2):
        r = random.randint(150, 255)
        g = random.randint(0, 100)
        data.append(((r, g), 1))

    random.shuffle(data)
    return data

if __name__ == "__main__":
    training_data = generate_pixel_data(num_samples=200)

    stonixVision = perceptron.Perceptron(num_inputs=2, learning_rate=0.05)

    if not load_weights_and_bias(stonixVision, filename="red_green_model"):
        print("Start training")
        stonixVision.train(training_data, epochs=1000)
        print("Finished training")
        save_weights_and_bias(stonixVision, filename="red_green_model")
    else:
        print("Loaded model Parameters")

    print("\nStarting Test")

    red_pixel = (200, 50)
    norm_red_pixel = (red_pixel[0] / 255.0, red_pixel[1] / 255.0)
    prediction_red = stonixVision.predict(norm_red_pixel)
    print(f"Pixel (R={red_pixel[0]}, G={red_pixel[1]}) -> Prediction: {'Red' if prediction_red == 1 else 'Green'} (Expected: Red)")

    green_pixel = (50, 200)
    norm_green_pixel = (green_pixel[0] / 255.0, green_pixel[1] / 255.0)
    prediction_green = stonixVision.predict(norm_green_pixel)
    print(f"Pixel (R={green_pixel[0]}, G={green_pixel[1]}) -> Prediction: {'Red' if prediction_green == 1 else 'Green'} (Expected: Green)")

    slightly_red_pixel = (180, 100)
    norm_slightly_red_pixel = (slightly_red_pixel[0] / 255.0, slightly_red_pixel[1] / 255.0)
    prediction_s_red = stonixVision.predict(norm_slightly_red_pixel)
    print(f"Pixel (R={slightly_red_pixel[0]}, G={slightly_red_pixel[1]}) -> Prediction: {'Red' if prediction_s_red == 1 else 'Green'} (Expected: Red)")

    slightly_green_pixel = (100, 180)
    norm_slightly_green_pixel = (slightly_green_pixel[0] / 255.0, slightly_green_pixel[1] / 255.0)
    prediction_s_green = stonixVision.predict(norm_slightly_green_pixel)
    print(f"Pixel (R={slightly_green_pixel[0]}, G={slightly_green_pixel[1]}) -> Prediction: {'Red' if prediction_s_green == 1 else 'Green'} (Expected: Green)")

    neutral_pixel = (128, 128)
    norm_neutral_pixel = (neutral_pixel[0] / 255.0, neutral_pixel[1] / 255.0)
    prediction_neutral = stonixVision.predict(norm_neutral_pixel)
    print(f"Pixel (R={neutral_pixel[0]}, G={neutral_pixel[1]}) -> Prediction: {'Red' if prediction_neutral == 1 else 'Green'} (Expected: Random)")