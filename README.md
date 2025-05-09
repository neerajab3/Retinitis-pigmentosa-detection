🧠 Retinitis Pigmentosa Detection using Deep Learning Models
This repository presents an image classification pipeline to detect Retinitis Pigmentosa (RP) from retinal images using three powerful deep learning models:
ResNet50, VGG16, and Vision Transformer (ViT). The project includes preprocessing, dataset balancing, model training, and evaluation steps.
📁 Repository Contents
File/Notebook	Description
RP_dataset_Balanced.ipynb	Loads the original dataset and applies image augmentation to balance classes (Normal vs RP).
RP_ResNet.ipynb	Builds and trains a ResNet50 model for RP detection.
RP_VGG.ipynb	Implements a VGG16 model for classification.
RP classification using VIT-ML_PC.ipynb	Applies a Vision Transformer (ViT) model to the same task.
🔍 Project Workflow
Data Preprocessing & Augmentation: Balanced the dataset using ImageDataGenerator.
Model Training: Fine-tuned ResNet50, VGG16, and ViT on the augmented dataset.
Evaluation: Measured accuracy, loss, and classification reports.
Compared results across the three models.
🛠️ Technologies Used
Python
TensorFlow / Keras
OpenCV
Matplotlib, Seaborn
Vision Transformer (tensorflow_hub or transformers)
| Model    | Accuracy | 
| CNN      | 92%      | 
| ResNet50 | 96%      | 
| VGG16    | 95%      | 
| ViT      | 98%      | 
