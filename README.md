# CIFAR-10 Image Classification using PyTorch, GPU with CUDA & AMP

This project demonstrates end-to-end image classification on the CIFAR-10 dataset using PyTorch, starting from a custom CNN model and extending to transfer learning with pretrained ResNet-18.

# 🚀 Features #

1. Data Preparation

- Applied standard CIFAR-10 normalization and augmentations (random crop, horizontal flip).
- Created train/test dataloaders with pinned memory and multiple workers for efficiency.

 # 2. Custom CNN #

- 5 convolutional blocks (Conv → BatchNorm → ReLU → MaxPool) with Dropout regularization.
- Fully connected layers for classification into 10 CIFAR-10 categories.
- Achieved ~82% test accuracy after 10 epochs


 # 3. Training Setup #

- Loss: CrossEntropyLoss with optional label smoothing.
- Optimizer: AdamW with weight decay.
- Scheduler: Cosine Annealing LR.
- Mixed Precision (AMP) enabled for faster GPU training on Tesla T4.

 # 4. Evaluation & Analysis #

- Accuracy and loss tracking for train/test sets.
- Per-class precision/recall/F1 using sklearn.
- Confusion matrix heatmap and per-class error analysis.
- Identified “most confused” class pairs (e.g., cat ↔ dog)

 # 5. Transfer Learning with ResNet-18 #

- Experiment A: Trained only the fully connected head → ~78% accuracy
- Experiment B: Fine-tuned layer4 + fc → ~91% accuracy
- Experiment C: Fine-tuned the full model (with warmup + cosine LR) → ~93% accuracy

# 📊 Results #
Model Variant	 vs Test Accuracy
- Custom SimpleCNN	~82%
- ResNet18 (head only)	~78%
- ResNet18 (layer4 + fc)	~91%
- ResNet18 (all layers)	~93%

# 🛠️ Requirements #

- Python 3.10+
- PyTorch ≥ 2.0
- torchvision
- matplotlib, seaborn, scikit-learn
