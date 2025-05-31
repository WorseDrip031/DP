import torch
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
from torchmetrics import Accuracy, F1Score, ConfusionMatrix

from tools.training import AGGC2022ClassificationDatamodule
from tools.training import Statistics


def evaluate_model(model:nn.Module,
                   datamodule:AGGC2022ClassificationDatamodule):
    
    # Device for neural network evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create metrics
    acc = Accuracy("multiclass", num_classes=model.num_classes)
    f1 = F1Score("multiclass", num_classes=model.num_classes)
    confmat = ConfusionMatrix(task="multiclass", num_classes=model.num_classes)

    # Evaluation
    stats = Statistics()
    all_preds = []
    all_labels = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        with tqdm(datamodule.dataloader_test, desc="Testing the trained model") as progress:

            print("Hello")

            for x, y in progress:



                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                probs = torch.softmax(y_hat, dim=1)
                preds = torch.argmax(probs, dim=1)
                labels = torch.argmax(y, dim=1)

                acc_test = acc(preds.cpu(), labels.cpu())
                f1_test = f1(preds.cpu(), labels.cpu())

                stats.step("acc", acc_test.item())
                stats.step("f1", f1_test.item())

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                progress.set_postfix(stats.get())

    all_preds_tensor = torch.cat(all_preds)
    all_labels_tensor = torch.cat(all_labels)

    conf_matrix = confmat(all_preds_tensor, all_labels_tensor)
    conf_matrix = conf_matrix.numpy()

    # Results
    print(f"Test Accuracy: {stats.get()['acc']}")
    print(f"Test F1-Score: {stats.get()['f1']}")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Save metrics to file
    with open("evaluation_metrics.txt", "w") as f:
        f.write(f"Test Accuracy: {stats.get()['acc']:.4f}\n")
        f.write(f"Test F1-Score: {stats.get()['f1']:.4f}\n")