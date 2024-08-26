import torch
from src.models import MAML_GNN_Model
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
import seaborn as sns

# Function to evaluate the model
def evaluate_model(model, data_loader, n_classes):
    model.eval()
    y_true = []
    y_pred = []
    y_pred_proba = []

    with torch.no_grad():
        for data in data_loader:
            ct_data, pet_data, fused_data, labels = data
            output = model(ct_data, pet_data, fused_data)
            pred_proba = torch.exp(output)
            pred = output.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_pred_proba.extend(pred_proba.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    return y_true, y_pred, y_pred_proba

# ROC Curve and AUC
def plot_roc_curve(y_true, y_pred_proba, n_classes, title="ROC Curve"):
    fpr = {}
    tpr = {}
    roc_auc = {}

    y_true_binarized = label_binarize(y_true, classes=np.arange(n_classes))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (area = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig('results/plots/roc_curve.png')
    plt.show()

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig('results/plots/confusion_matrix.png')
    plt.show()

# t-SNE Visualization
def plot_tsne(features, labels, title="t-SNE Visualization"):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(features)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=plt.cm.Spectral)
    plt.colorbar(scatter)
    plt.title(title)
    plt.savefig('results/plots/tsne_visualization.png')
    plt.show()

# Violin Plot
def plot_violin(data, labels, title="Violin Plot of AUC Values"):
    sns.violinplot(x=labels, y=data)
    plt.title(title)
    plt.savefig('results/plots/violin_plot.png')
    plt.show()

# T-Test
def perform_ttest(data1, data2, title="T-Test Results"):
    t_stat, p_value = ttest_ind(data1, data2)
    print(f"{title} - T-Statistic: {t_stat}, P-Value: {p_value}")

# Evaluate the model
data_loader = ...  # Replace with your DataLoader
model = MAML_GNN_Model(in_channels=7, out_channels=3)
y_true, y_pred, y_pred_proba = evaluate_model(model, data_loader, n_classes=3)

# Generate evaluation plots and metrics
plot_roc_curve(y_true, y_pred_proba, n_classes=3, title="ROC Curve for MAML-GNN")
plot_confusion_matrix(y_true, y_pred, labels=np.arange(3), title="Confusion Matrix for MAML-GNN")
plot_tsne(y_pred_proba, y_true, title="t-SNE Visualization for MAML-GNN")

# Example data for violin plot and t-test (replace with actual experimental results)
auc_values_maml_gnn = [0.92, 0.88, 0.91, 0.89, 0.93]  # Example AUC values for MAML-GNN
auc_values_baseline = [0.85, 0.82, 0.84, 0.83, 0.86]  # Example AUC values for baseline

# Generate statistical plots and analysis
plot_violin(auc_values_maml_gnn + auc_values_baseline, ['MAML-GNN'] * len(auc_values_maml_gnn) + ['Baseline'] * len(auc_values_baseline), title="Violin Plot of AUC Values")
perform_ttest(auc_values_maml_gnn, auc_values_baseline, title="T-Test for AUC Comparison")
