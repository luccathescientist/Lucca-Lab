import matplotlib.pyplot as plt
import json

def plot_results():
    with open("results.json", "r") as f:
        data = json.load(f)
    
    epochs = [d["epoch"] for d in data]
    loss = [d["loss"] for d in data]
    accuracy = [d["accuracy"] for d in data]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, marker='o', color='red')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, marker='s', color='green')
    plt.title("Logic Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig("distillation_metrics.png")
    print("Chart saved: distillation_metrics.png")

if __name__ == "__main__":
    plot_results()
