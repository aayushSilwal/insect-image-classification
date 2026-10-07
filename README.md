# Insect Image Classification with CNNs & Transfer Learning

Classifies images of insects/crop pests into **9 classes** using TensorFlow/Keras. Compares a CNN built from scratch, a deeper regularized CNN, an optimizer experiment (Adam vs SGD), and fine-tuning a pretrained VGG16.

**Classes:** aphids, armyworm, beetle, bollworm, grasshopper, mites, mosquito, sawfly, stem_borer

## Results

Evaluated on the held-out test set (384 images).

| Model | Optimizer | Test Accuracy | Test Loss |
|---|---|---|---|
| Baseline CNN (3 conv blocks) | Adam | 91.93% | 0.3885 |
| Deeper CNN (BatchNorm + Dropout) | Adam | 28.91% | 1.8902 |
| Baseline CNN | SGD (lr=0.001) | 24.48% | 2.0899 |
| **VGG16 transfer learning (frozen base)** | Adam | **97.14%** | 0.1974 |

**Takeaways**
- Transfer learning with pretrained ImageNet features gave the best accuracy and converged in only a few epochs.
- The baseline CNN trained from scratch worked well, but the deeper regularized CNN did not converge within 20 epochs. Heavy dropout plus a larger network likely needs a longer schedule or tuning.
- SGD at a low learning rate learned much more slowly than Adam over the same 20 epochs.

## Dataset

- 2,495 images across 9 classes: 2,111 train / 384 test
- Corrupted images were checked for and removed with OpenCV
- The dataset is not included in this repo; place it in `train/` and `test/` folders (one subfolder per class) and update the paths in the notebook.

## Pipeline

1. **Data checks**: class counts, sample image per class, corrupted-image removal
2. **Loading**: `image_dataset_from_directory` (128×128 for scratch models, 224×224 for VGG16), batch size 32
3. **Augmentation**: random horizontal flip, rotation, and zoom
4. **Models**
   - Baseline CNN: 3 Conv/MaxPool blocks → 3 dense layers (~6.6M params)
   - Deeper CNN: double-conv blocks with BatchNormalization and Dropout (~17.2M params)
   - Transfer model: frozen VGG16 base + dense head with dropout (~6.5M trainable params)
5. **Training & comparison**: Adam vs SGD, validation-loss and accuracy curves
6. **Evaluation**: test accuracy/loss, per-class precision/recall/F1, sample predictions

## Tech Stack

Python, TensorFlow/Keras, OpenCV, scikit-learn, NumPy, Matplotlib

## Getting Started

```bash
git clone https://github.com/aayush505/<repo-name>.git
cd <repo-name>
pip install tensorflow opencv-python scikit-learn numpy matplotlib
jupyter notebook insect-image-classification.ipynb
```

The notebook was developed in Google Colab (GPU recommended). Pretrained VGG16 weights download automatically on first run.

## Notes

The test set was also used as validation data during training, so reported accuracies may be slightly optimistic. A separate validation split would give a cleaner estimate.

## Author

**Aayush Silwal**: [GitHub](https://github.com/aayush505)
