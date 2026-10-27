# CIFAR-10 CNN: from 75% to 89%, one tested change at a time

A convolutional neural network trained from scratch on CIFAR-10 (60,000 color images, 10 classes), built in PyTorch.

The point of this project is not only the final number, but **how it got there**: every improvement was added one at a time, measured, and kept only if it helped.

**Final result: 89.07% test accuracy** (best epoch chosen on a separate validation set; the test set was used once).

## Results

| Version | Change | Accuracy | Kept? |
|---|---|---|---|
| v0 | Baseline: 3 conv layers, no augmentation | 75.06% (test) | |
| v1 | + data augmentation (random crop + horizontal flip) | 80.76% (test) | ✅ |
| v2 | + a second conv layer in each block | 83.64% (test) | ✅ |
| v3 | + batch normalization | 85.79% (test) | ✅ |
| v4 | + cosine learning-rate schedule | 88.07% (test) | ✅ |
| v5 | + dropout 0.5 (20 epochs) | 87.07% (test) | ❌ |
| v4 + val | Honest setup: 45k train / 5k validation, best epoch chosen on validation | 88.08% val, 87.60% test | ✅ |
| **v6** | **+ 40 epochs instead of 20** | **89.70% val, 89.07% test** | ✅ **Final** |
| v7 | + dropout 0.5 (40 epochs) | 89.52% val, 88.46% test | ❌ |

v0 to v5 were measured on the test set after the last epoch, before I added a validation set. From "v4 + val" on, every decision was made on validation and the test set was run only once per experiment.

## What each step taught me

- **Augmentation:** without it, training loss dropped to 0.053 while test accuracy peaked at epoch 9 and then fell, which is classic overfitting. Random shifts and flips stopped the model from memorizing the training images.
- **Deeper blocks:** a second conv layer before each pooling step lets the model combine patterns while the maps still have full detail.
- **Batch normalization:** re-centers each layer's outputs on every batch. The model reached 60% after epoch 1, compared with 48% without it.
- **Cosine learning-rate schedule:** with a fixed learning rate, accuracy bounced 1 to 2% from epoch to epoch. Shrinking the step size toward the end let the model settle, and the last epoch became the best epoch.
- **Validation set:** choosing the best epoch on the test set quietly inflates the result. Moving that choice to a separate validation split keeps the test number honest.
- **Dropout (rejected twice):** augmentation and batch norm had already removed the overfitting, so dropout only slowed learning. With 40 epochs the gap shrank to 0.18% on validation, which is within noise, so I kept the simpler model.
- **Reproducibility:** with a fixed seed, rerunning v6 gave exactly the same numbers.

## Model

```
3 blocks of:  Conv 3×3 → BatchNorm → ReLU → Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2
              (32 → 64 → 128 filters; maps shrink 32×32 → 16×16 → 8×8 → 4×4)
Then:         Flatten (2,048) → Linear 256 → ReLU → Linear 10
```

815,018 parameters.

## Training setup

- Data: 45,000 train / 5,000 validation (random split, seed 0) / 10,000 test
- Preprocessing: per-channel normalization; random crop (padding 4) and horizontal flip on training images only
- Loss: cross-entropy. Optimizer: Adam (lr 0.001) with cosine annealing over 40 epochs. Batch size 128.

## How to run

```bash
pip install -r requirements.txt
python cifar_train.py
```

The script downloads CIFAR-10 into `data/` on the first run.

The trained model is in `cifar_model_v6_40ep.pt`. Load it into the model defined in `cifar_train.py` with `model.load_state_dict(torch.load("cifar_model_v6_40ep.pt"))`.
