import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import glob
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa
import seaborn as sns
from datetime import datetime
import json

# CONFIGURATION
BASE_DIR = '/gpfs0/bgu-rriemer/users/reifk/data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast'
CHECKPOINT_DIR = '/gpfs0/bgu-rriemer/users/reifk/efficientnet_checkpoints'

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join('outputs', f'experiment_{timestamp}')
os.makedirs(output_dir, exist_ok=True)

# Define subdirectories for benign and malignant
benign_dir = os.path.join(BASE_DIR, 'benign')
malignant_dir = os.path.join(BASE_DIR, 'malignant')

# Create directories for training and validation data
train_dir = os.path.join(BASE_DIR, 'train')
val_dir = os.path.join(BASE_DIR, 'validation')

# Clean up existing train and validation directories before starting fresh
print("\nCleaning up existing train/validation directories...")
for directory in [train_dir, val_dir]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"  Removed existing directory: {directory}")
    os.makedirs(directory, exist_ok=True)
    print(f"  Created fresh directory: {directory}")
    
# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Create subdirectories for benign and malignant in train and validation folders
for category in ['benign', 'malignant']:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)

print("BREAKHIS BREAST CANCER BINARY CLASSIFICATION")
print(f"Base directory: {BASE_DIR}")
print(f"Output directory: {output_dir}")
print(f"Checkpoint directory: {CHECKPOINT_DIR}")
print()

# STEP 1: DATA PREPARATION
print("STEP 1: Preparing data splits...")

# Initialize split info tracker
split_info = {}

def split_and_copy_images(source_dir, train_dest_dir, val_dest_dir, split_ratio=0.2):
    """Split images into train and validation sets"""
    image_files = glob.glob(os.path.join(source_dir, '**', '*.png'), recursive=True)
    train_files, val_files = train_test_split(image_files, test_size=split_ratio, random_state=42)
    
    print(f"  Found {len(image_files)} images in {os.path.basename(source_dir)}")
    print(f"  - Training: {len(train_files)}")
    print(f"  - Validation: {len(val_files)}")
    
    for file in train_files:
        shutil.copy(file, train_dest_dir)
    
    for file in val_files:
        shutil.copy(file, val_dest_dir)
    
    return len(image_files), len(train_files), len(val_files)

# Split and copy images for benign and malignant
print("\nSplitting benign images...")
total_b, train_b, val_b = split_and_copy_images(benign_dir, os.path.join(train_dir, 'benign'), os.path.join(val_dir, 'benign'))
split_info['benign'] = {'total': total_b, 'train': train_b, 'validation': val_b}

print("\nSplitting malignant images...")
total_m, train_m, val_m = split_and_copy_images(malignant_dir, os.path.join(train_dir, 'malignant'), os.path.join(val_dir, 'malignant'))
split_info['malignant'] = {'total': total_m, 'train': train_m, 'validation': val_m}

print("\nSplit Information Summary:")
print(f"Benign   - Total: {total_b}, Train: {train_b} ({train_b/total_b*100:.1f}%), Val: {val_b} ({val_b/total_b*100:.1f}%)")
print(f"Malignant - Total: {total_m}, Train: {train_m} ({train_m/total_m*100:.1f}%), Val: {val_m} ({val_m/total_m*100:.1f}%)")

# STEP 2: DATA AUGMENTATION SETUP
print("\n")
print("STEP 2: Setting up data augmentation...")

# General data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    brightness_range=[0.9, 1.1],
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.5)

# Create general data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

test_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# STEP 3: BALANCE TRAINING SET TO 50-50 RATIO
print("\n")
print("STEP 3: Balancing training set to 50-50 ratio...")

# Calculate the number of images in each class
class_counts = {category: len(glob.glob(os.path.join(train_dir, category, '*.png'))) 
                for category in ['benign', 'malignant']}
print(f"\nOriginal class distribution in training set:")
print(f"  Benign: {class_counts['benign']}")
print(f"  Malignant: {class_counts['malignant']}")

# Find majority and minority classes
majority_class = max(class_counts, key=class_counts.get)
minority_class = min(class_counts, key=class_counts.get)
majority_count = class_counts[majority_class]
minority_count = class_counts[minority_class]

print(f"\nMajority class: {majority_class} ({majority_count} images)")
print(f"Minority class: {minority_class} ({minority_count} images)")

# Calculate how many augmented images needed for minority class
# Target: same number as majority class for 50-50 balance
target_count = majority_count
augmentations_needed = target_count - minority_count

print(f"\nTarget count for 50-50 balance: {target_count} images per class")
print(f"Augmentations needed for {minority_class}: {augmentations_needed}")

# Custom augmentation pipeline
aug_pipeline = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.2),
    iaa.Affine(rotate=(-45, 45)),
    iaa.Multiply((0.8, 1.2)),
    iaa.GaussianBlur(sigma=(0.0, 3.0)),
    iaa.AdditiveGaussianNoise(scale=(0.01*255, 0.05*255))
])

# Augment minority class to reach target count
print(f"\nAugmenting {minority_class} class...")
minority_dir = os.path.join(train_dir, minority_class)
minority_images = glob.glob(os.path.join(minority_dir, '*.png'))

# Calculate how many augmentations per original image
augs_per_image = augmentations_needed // len(minority_images)
remaining_augs = augmentations_needed % len(minority_images)

print(f"  Original {minority_class} images: {len(minority_images)}")
print(f"  Augmentations per image: {augs_per_image}")
print(f"  Extra augmentations needed: {remaining_augs}")

augmented_count = 0
for idx, img_path in enumerate(minority_images):
    img = tf.keras.preprocessing.image.load_img(img_path)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Determine number of augmentations for this image
    num_augs = augs_per_image
    if idx < remaining_augs:
        num_augs += 1
    
    # Create augmented versions
    for i in range(num_augs):
        aug_images = aug_pipeline(images=img_array.astype(np.uint8))
        for aug_img in aug_images:
            aug_img_path = os.path.join(minority_dir, f"aug_{augmented_count:05d}_{os.path.basename(img_path)}")
            aug_img_pil = tf.keras.preprocessing.image.array_to_img(aug_img)
            aug_img_pil.save(aug_img_path)
            augmented_count += 1
    
    # Progress indicator every 100 images
    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx + 1}/{len(minority_images)} images, created {augmented_count} augmentations so far...")

print(f"  Total augmented images created: {augmented_count}")

# Verify final counts
final_class_counts = {category: len(glob.glob(os.path.join(train_dir, category, '*.png'))) 
                      for category in ['benign', 'malignant']}
print(f"\nFinal balanced class distribution in training set:")
print(f"  Benign: {final_class_counts['benign']}")
print(f"  Malignant: {final_class_counts['malignant']}")
print(f"  Ratio: {final_class_counts['benign']}/{final_class_counts['malignant']} = {final_class_counts['benign']/final_class_counts['malignant']:.2f}")

# Verify validation and test sets remain unchanged
val_class_counts = {category: len(glob.glob(os.path.join(val_dir, category, '*.png'))) 
                    for category in ['benign', 'malignant']}
print(f"\nValidation set (UNCHANGED):")
print(f"  Benign: {val_class_counts['benign']}")
print(f"  Malignant: {val_class_counts['malignant']}")

# Re-create generators after augmentation
print("\nRecreating data generators after augmentation...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

test_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# STEP 4: BUILD MODEL
print("\n")
print("STEP 4: Building model architecture...")

model = Sequential([
    EfficientNetB5(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    GlobalAveragePooling2D(),
    Dropout(0.8),
    Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel architecture:")
model.summary()

# STEP 5: SETUP CALLBACKS
print("\n")
print("STEP 5: Setting up callbacks...")

checkpoint_filepath = os.path.join(CHECKPOINT_DIR, 'best_model.keras')
checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

def scheduler(epoch, lr):
    if epoch < 10:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.1))

lr_callback = LearningRateScheduler(scheduler, verbose=1)

print("Callbacks configured:")
print("  - ModelCheckpoint: saves best model")
print("  - EarlyStopping: patience=15")
print("  - ReduceLROnPlateau: patience=5, factor=0.5")
print("  - LearningRateScheduler: exponential decay after epoch 10")

# STEP 6: COMPUTE CLASS WEIGHTS
print("STEP 6: Computing class weights...")

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

print(f"Class weights: {class_weights}")

# STEP 7: TRAIN MODEL
print("\n")
print("STEP 7: Training model for 37 epochs...")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=37,
    #class_weight=class_weights,
    callbacks=[checkpoint, early_stopping, reduce_lr, lr_callback],
    verbose=1
)

# STEP 8: LOAD BEST MODEL AND EVALUATE
print("\n")
print("STEP 8: Loading best model and evaluating...")

# Load the best saved model
best_model = tf.keras.models.load_model(checkpoint_filepath)

# Evaluate on validation set
val_loss, val_accuracy = best_model.evaluate(validation_generator, verbose=0)
print(f'\nValidation loss: {val_loss:.4f}')
print(f'Validation accuracy: {val_accuracy * 100:.2f}%')

# Evaluate on test set
test_loss, test_accuracy = best_model.evaluate(test_generator, verbose=0)
print(f'\nTest loss: {test_loss:.4f}')
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# STEP 9: GENERATE PREDICTIONS AND METRICS
print("\n")
print("STEP 9: Generating predictions and metrics...")

# Generate predictions on test set
y_pred = best_model.predict(test_generator, verbose=1)
y_pred_labels = (y_pred > 0.5).astype(int).flatten()
y_true = test_generator.classes[:len(y_pred_labels)]

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_labels)
class_names = ['benign', 'malignant']

print("\nConfusion Matrix:")
print(conf_matrix)

# Print classification report
print("\nClassification Report:")
report = classification_report(y_true, y_pred_labels, target_names=class_names)
print(report)

# STEP 10: PLOT RESULTS
print("\n")
print("STEP 10: Plotting results...")

# Plot training & validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'), dpi=300, bbox_inches='tight')
print(f"Saved accuracy plot to: {os.path.join(output_dir, 'accuracy_plot.png')}")
plt.close()

# Plot training & validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'loss_plot.png'), dpi=300, bbox_inches='tight')
print(f"Saved loss plot to: {os.path.join(output_dir, 'loss_plot.png')}")
plt.close()

# Plot confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, 
            annot_kws={"size": 34})
plt.xlabel('Predicted Label', fontsize=18)
plt.ylabel('True Label', fontsize=18)
plt.title('Confusion Matrix - Test Set', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print(f"Saved confusion matrix to: {os.path.join(output_dir, 'confusion_matrix.png')}")
plt.close()

# STEP 11: SAVE ARTIFACTS
print("\n" + "=" * 80)
print("STEP 11: Saving all artifacts and configurations...")

# Save best model to output directory
output_model_path = os.path.join(output_dir, 'best_model.keras')
shutil.copy(checkpoint_filepath, output_model_path)
print(f"Model copied to: {output_model_path}")

# Save configuration
config = {
    'timestamp': timestamp,
    'base_dir': BASE_DIR,
    'img_size': (224, 224),
    'batch_size': 32,
    'epochs': 28,
    'train_split': 0.8,
    'val_split': 0.2,
    'random_state': 42,
    'model': 'EfficientNetB5',
    'learning_rate': 1e-4,
    'optimizer': 'Adam',
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy']
}

config_path = os.path.join(output_dir, 'config.json')
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"Configuration saved to: {config_path}")

# Save split information
split_path = os.path.join(output_dir, 'split_info.json')
with open(split_path, 'w') as f:
    json.dump(split_info, f, indent=2)
print(f"Split info saved to: {split_path}")

# Save metrics summary
metrics_summary = {
    'test': {
        'accuracy': float(test_accuracy),
        'loss': float(test_loss),
    }
}

metrics_path = os.path.join(output_dir, 'metrics_summary.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics_summary, f, indent=2)
print(f"Metrics summary saved to: {metrics_path}")

# Save detailed classification report as JSON
report_dict = classification_report(y_true, y_pred_labels, target_names=class_names, output_dict=True)

test_report_path = os.path.join(output_dir, 'test_report.json')
with open(test_report_path, 'w') as f:
    json.dump(report_dict, f, indent=2)
print(f"Test report saved to: {test_report_path}")

# Save validation report
val_loss, val_accuracy = best_model.evaluate(validation_generator, verbose=0)
y_val_pred = best_model.predict(validation_generator, verbose=0)
y_val_pred_labels = (y_val_pred > 0.5).astype(int).flatten()
y_val_true = validation_generator.classes[:len(y_val_pred_labels)]

val_report_dict = classification_report(y_val_true, y_val_pred_labels, target_names=class_names, output_dict=True)

val_report_path = os.path.join(output_dir, 'validation_report.json')
with open(val_report_path, 'w') as f:
    json.dump(val_report_dict, f, indent=2)
print(f"Validation report saved to: {val_report_path}")

# Save training history
history_dict = {
    'accuracy': [float(x) for x in history.history['accuracy']],
    'loss': [float(x) for x in history.history['loss']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    'val_loss': [float(x) for x in history.history['val_loss']]
}

history_path = os.path.join(output_dir, 'training_history.json')
with open(history_path, 'w') as f:
    json.dump(history_dict, f, indent=2)
print(f"Training history saved to: {history_path}")

# TRAINING COMPLETE
# Print final summary of results and artifacts
print("TRAINING COMPLETE!")
print(f"\nExperiment timestamp: {timestamp}")
print(f"Output directory: {output_dir}")
print(f"\nBest model saved to: {output_model_path}")
print(f"Plots saved to: {output_dir}")
print(f"Configuration saved to: {output_dir}")
print(f"\nFinal Results:")
print(f"  - Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"  - Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"  - Test Loss: {test_loss:.4f}")
print(f"\nArtifacts saved:")
print(f"  - best_model.keras")
print(f"  - config.json")
print(f"  - split_info.json")
print(f"  - metrics_summary.json")
print(f"  - test_report.json")
print(f"  - validation_report.json")
print(f"  - training_history.json")
print(f"  - accuracy_plot.png")
print(f"  - loss_plot.png")
print(f"  - confusion_matrix.png")