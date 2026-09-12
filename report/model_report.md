# AgriSmart AI — Model Report

## Task
Multi-class image classification of plant leaf diseases. Given a photo of a crop leaf,
the model predicts one of 38 classes, covering multiple crop species (Apple, Tomato,
Corn, Grape, Potato, Pepper, Strawberry, Soybean, and others) and their associated
diseases, or "healthy" for unaffected leaves.

## Dataset & Split
- **Source:** New Plant Diseases Dataset (Augmented) — a PlantVillage-derived dataset,
  publicly available on Kaggle:
  https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
- **License:** copyright-authors (as listed on the Kaggle dataset page)
- **Classes:** 38 crop-disease categories (including healthy classes)
- **Training set:** 70,295 images
- **Validation set:** 17,572 images
- **Split:** used the dataset's pre-existing train/valid folder split, no further
  splitting was performed
- **Note:** trained on the full public dataset (38 classes), as it was the
  training/validation source specified in the problem statement.

## Model / Approach
Architecture based on a standard ResNet50 transfer-learning pattern, adapted from a
public reference implementation (see README's Originality Declaration for details).

- **Architecture:** ResNet50 with transfer learning
  - Pretrained ImageNet weights, base layers frozen (`trainable=False`)
  - Custom classification head: `GlobalAveragePooling2D → Dense(128, relu) →
    Dense(64, relu) → Dense(38, softmax)`
  - Preprocessing (`preprocess_input`, ImageNet channel normalization) built directly
    into the model graph via a `Lambda` layer, so raw images can be passed to the
    model without separate preprocessing at inference time
- **Total parameters:** 23,860,710 (272,998 trainable, 23,587,712 frozen)
- **Training:**
  - Optimizer: Adam
  - Loss: categorical cross-entropy
  - Batch size: 32
  - Epochs: 30 (no early stopping triggered; validation accuracy kept improving
    intermittently through the full run)
  - Callbacks: `ModelCheckpoint` (saved best model by validation accuracy),
    `EarlyStopping` (patience 7, did not trigger), `ReduceLROnPlateau` (patience 5)
- **Data augmentation** (applied to training data only, to address the lab-to-field
  generalization gap the challenge is designed around): rotation (±20°), horizontal
  and vertical flips, brightness variation (0.7–1.3x), shear, zoom, width/height
  shifts. This goes beyond the augmentation in the baseline reference notebook this
  work was adapted from (see Originality Declaration in README), specifically to
  target generalization beyond clean, lab-condition images.

## Metric & Result
Evaluated on the full validation set (17,572 images):

| Metric | Score |
|---|---|
| Accuracy | 94.83% |
| Macro-F1 | 0.9487 |
| Weighted-F1 | 0.9490 |

Per-class precision/recall/F1 and the full confusion matrix were computed
(see `confusion_matrix.png` below, and the notebook in `/model` for the full
per-class breakdown).

## Baseline
No official baseline metric was published to our team for this challenge. Our
reported macro-F1 (0.9487) reflects performance on our own held-out validation split
of the public training dataset, not on the organizers' independent test set.

## Limitations
- **Validation, not true held-out testing:** our validation set is drawn from the
  same lab-condition source distribution (clean backgrounds, controlled lighting) as
  our training data. The problem statement explicitly notes that models trained this
  way often perform well on further lab-condition images but degrade on real
  field-condition photos (natural lighting, clutter, occlusion). We addressed this
  with augmentation targeting these conditions, but our reported metrics do not
  reflect true field-image performance, since no such test set was available to us.
- **Class confusion cluster (Tomato diseases):** the confusion matrix shows a
  consistent confusion cluster among Tomato Early Blight, Late Blight, Target Spot,
  and Bacterial Spot — likely due to visually overlapping symptoms (irregular
  brown/necrotic leaf lesions) across these categories. Tomato Target Spot in
  particular shows low precision (0.63) despite high recall (0.97), indicating the
  model over-predicts this class, pulling in genuine cases of the other three.
- **Secondary confusion (Corn diseases):** a smaller, similar pattern appears
  between Corn Cercospora Leaf Spot / Gray Leaf Spot and Corn Northern Leaf Blight.
- **Class list:** trained on all 38 classes present in the public dataset, since no
  official ~15–20 class shared list was provided during the competition window.

## Files
- Full training/evaluation notebook: `/model/training.ipynb`
- Trained model weights: hosted on Google Drive (see README for link and load
  instructions — `load_weights()`, not `load_model()`, due to a Keras
  Lambda-layer serialization limitation)
- Class-index mapping: `class_indices.json` (hosted alongside weights on Drive)

![Confusion Matrix](confusion_matrix.png)