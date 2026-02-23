import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
import os
import shutil
import glob
import seaborn as sns
from collections import defaultdict
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

# Create directories for training, validation, and test data
train_dir = os.path.join(BASE_DIR, 'train')
validation_dir = os.path.join(BASE_DIR, 'validation')
test_dir = os.path.join(BASE_DIR, 'test')

# Clean up existing directories before starting fresh
print("\nCleaning up existing train/validation/test directories...")
for directory in [train_dir, validation_dir, test_dir]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"  Removed existing directory: {directory}")
    os.makedirs(directory, exist_ok=True)
    print(f"  Created fresh directory: {directory}")

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Create subdirectories for benign and malignant in all directories
for directory in [train_dir, validation_dir, test_dir]:
    for category in ['benign', 'malignant']:
        os.makedirs(os.path.join(directory, category), exist_ok=True)

print("=" * 80)
print("BREAKHIS BREAST CANCER BINARY CLASSIFICATION")
print("Patient-Level Split (No Data Leakage)")
print("=" * 80)
print(f"Base directory: {BASE_DIR}")
print(f"Checkpoint directory: {CHECKPOINT_DIR}")
print()

# STEP 1: EXTRACT PATIENT IDs FROM FILENAMES
print("STEP 1: Extracting patient IDs from filenames...")
print("\nBreakHis filename formats supported:")
print("  Format: SOB_[type]_[subtype]-[code]-[patient_id]-[magnification]-[sequence].png")
print("  Examples:")
print("    - SOB_B_A-14-22549AB-100-001.png (Benign Adenosis)")
print("    - SOB_M_DC-14-10926-100-001.png (Malignant Ductal Carcinoma)")
print("\nKey components:")
print("  - SOB = Surgical Open Biopsy")
print("  - M/B = Malignant/Benign")
print("  - A/F/TA/PT/DC/LC/MC/PC = Tumor subtype")
print("  - code = Version code (14 or other)")
print("  - patient_id = Unique identifier for the slide/patient (numeric or alphanumeric)")
print("  - 40/100/200/400 = Magnification (40X, 100X, 200X, 400X)")
print("  - sequence = Image sequence number")

def extract_patient_id(filepath):
    """
    Extract patient ID (slide ID) from BreakHis filename.
    Format: SOB_[type]_[subtype]-[code]-[patient_id]-[magnification]-[sequence].png
    Examples:
    - SOB_B_A-14-22549AB-100-001.png → patient_id = '22549AB'
    - SOB_M_DC-14-10926-100-001.png → patient_id = '10926'
    """
    filename = os.path.basename(filepath)
    try:
        # Remove .png extension
        name_without_ext = filename.replace('.png', '')
        
        # Split by '-'
        parts = name_without_ext.split('-')
        
        if len(parts) >= 5:
            # Format: SOB_[type]_[subtype]-[code]-[patient_id]-[magnification]-[sequence]
            # After split by '-': ['SOB_B_A', 'code', 'patient_id', 'magnification', 'sequence']
            # Patient ID is at index 2 (third element)
            patient_id = parts[2]
            return patient_id
        else:
            print(f"Warning: Unexpected filename format: {filename}")
            print(f"  Parts: {parts}")
            # Fallback: return the whole filename
            return filename
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return filename

def get_magnification(filepath):
    """
    Extract magnification level from filename.
    Format: SOB_[type]_[subtype]-[code]-[patient_id]-[magnification]-[sequence].png
    Examples:
    - SOB_B_A-14-22549AB-100-001.png → '100'
    - SOB_M_DC-14-10926-40-001.png → '40'
    """
    filename = os.path.basename(filepath)
    try:
        # Remove .png extension
        name_without_ext = filename.replace('.png', '')
        parts = name_without_ext.split('-')
        
        if len(parts) >= 5:
            # Format: SOB_[type]_[subtype]-[code]-[patient_id]-[magnification]-[sequence]
            # After split by '-': ['SOB_B_A', 'code', 'patient_id', 'magnification', 'sequence']
            # Magnification is at index 3 (fourth element)
            mag = parts[3]
            # Verify it's numeric (valid magnifications: 40, 100, 200, 400)
            if mag.isdigit():
                return mag
            else:
                print(f"Warning: Magnification is not numeric: {mag} in {filename}")
                return "unknown"
        else:
            print(f"Warning: Cannot extract magnification from {filename}")
            return "unknown"
    except Exception as e:
        print(f"Error extracting magnification from {filename}: {e}")
        return "unknown"

def get_tumor_type(filepath):
    """Extract tumor type (benign/malignant) from path."""
    if 'benign' in filepath:
        return 'benign'
    elif 'malignant' in filepath:
        return 'malignant'
    else:
        return 'unknown'

# STEP 2: COLLECT ALL IMAGES AND GROUP BY PATIENT
print("\n" + "=" * 80)
print("STEP 2: Collecting images and grouping by patient...")

# Collect all images
all_benign_images = glob.glob(os.path.join(benign_dir, '**', '*.png'), recursive=True)
all_malignant_images = glob.glob(os.path.join(malignant_dir, '**', '*.png'), recursive=True)
all_images = all_benign_images + all_malignant_images

print(f"\nTotal images found: {len(all_images)}")
print(f"  Benign: {len(all_benign_images)}")
print(f"  Malignant: {len(all_malignant_images)}")

# Group images by patient ID
patient_groups = defaultdict(list)
patient_labels = {}  # Store the label (benign/malignant) for each patient

for img_path in all_images:
    patient_id = extract_patient_id(img_path)
    tumor_type = get_tumor_type(img_path)
    
    patient_groups[patient_id].append(img_path)
    patient_labels[patient_id] = tumor_type

print(f"\nUnique patients found: {len(patient_groups)}")

# Analyze patient distribution
benign_patients = [pid for pid, label in patient_labels.items() if label == 'benign']
malignant_patients = [pid for pid, label in patient_labels.items() if label == 'malignant']

print(f"  Benign patients: {len(benign_patients)}")
print(f"  Malignant patients: {len(malignant_patients)}")

# Show example patient data
print("\nExample patient data (first 3 patients):")
for i, (patient_id, images) in enumerate(list(patient_groups.items())[:3]):
    mags = [get_magnification(img) for img in images]
    mag_counts = {mag: mags.count(mag) for mag in set(mags)}
    print(f"  Patient {patient_id} ({patient_labels[patient_id]}): {len(images)} images")
    print(f"    Magnifications: {mag_counts}")

# STEP 3: PATIENT-LEVEL TRAIN/VALIDATION/TEST SPLIT (WITH STRATIFICATION)
print("\n" + "=" * 80)
print("STEP 3: Performing stratified patient-level train/validation/test split...")
print("\nUsing StratifiedGroupKFold (2-stage split) to ensure:")
print("  1. No patient appears in multiple sets (no data leakage)")
print("  2. Class distribution is maintained in all sets (stratification)")
print("  3. Validation and test are separate at patient level")

# Initialize split info tracker
split_info = {}

# Prepare data for StratifiedGroupKFold
patient_ids = list(patient_groups.keys())
labels = [patient_labels[pid] for pid in patient_ids]

# Convert labels to binary (0 = benign, 1 = malignant) for stratification
label_map = {'benign': 0, 'malignant': 1}
binary_labels = np.array([label_map[label] for label in labels])

# STAGE 1: Split into train (80%) and temp_val (20%)
sgkf_stage1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, temp_val_idx = next(sgkf_stage1.split(X=np.array(patient_ids).reshape(-1, 1), 
                                                   y=binary_labels, 
                                                   groups=np.array(patient_ids)))

train_patients = [patient_ids[i] for i in train_idx]
temp_val_patients = [patient_ids[i] for i in temp_val_idx]

# STAGE 2: Split temp_val (20%) into validation (10%) and test (10%)
temp_val_labels = np.array([label_map[patient_labels[pid]] for pid in temp_val_patients])
sgkf_stage2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
val_idx, test_idx = next(sgkf_stage2.split(X=np.array(temp_val_patients).reshape(-1, 1), 
                                            y=temp_val_labels, 
                                            groups=np.array(temp_val_patients)))

validation_patients = [temp_val_patients[i] for i in val_idx]
test_patients = [temp_val_patients[i] for i in test_idx]

print(f"\nStratified split results:")
print(f"  Training patients: {len(train_patients)} (~80%)")
print(f"  Validation patients: {len(validation_patients)} (~10%)")
print(f"  Test patients: {len(test_patients)} (~10%)")

# Calculate image counts
train_images = [img for pid in train_patients for img in patient_groups[pid]]
validation_images = [img for pid in validation_patients for img in patient_groups[pid]]
test_images = [img for pid in test_patients for img in patient_groups[pid]]

print(f"\nImage distribution:")
print(f"  Training images: {len(train_images)}")
print(f"  Validation images: {len(validation_images)}")
print(f"  Test images: {len(test_images)}")

# Show distribution by class
train_benign = sum(1 for pid in train_patients if patient_labels[pid] == 'benign')
train_malignant = sum(1 for pid in train_patients if patient_labels[pid] == 'malignant')
val_benign = sum(1 for pid in validation_patients if patient_labels[pid] == 'benign')
val_malignant = sum(1 for pid in validation_patients if patient_labels[pid] == 'malignant')
test_benign = sum(1 for pid in test_patients if patient_labels[pid] == 'benign')
test_malignant = sum(1 for pid in test_patients if patient_labels[pid] == 'malignant')

print(f"\nPatient distribution by class:")
print(f"  Training - Benign: {train_benign}, Malignant: {train_malignant}")
print(f"  Validation - Benign: {val_benign}, Malignant: {val_malignant}")
print(f"  Test - Benign: {test_benign}, Malignant: {test_malignant}")

# Count images by class
train_benign_imgs = sum(len(patient_groups[pid]) for pid in train_patients if patient_labels[pid] == 'benign')
train_malignant_imgs = sum(len(patient_groups[pid]) for pid in train_patients if patient_labels[pid] == 'malignant')
val_benign_imgs = sum(len(patient_groups[pid]) for pid in validation_patients if patient_labels[pid] == 'benign')
val_malignant_imgs = sum(len(patient_groups[pid]) for pid in validation_patients if patient_labels[pid] == 'malignant')
test_benign_imgs = sum(len(patient_groups[pid]) for pid in test_patients if patient_labels[pid] == 'benign')
test_malignant_imgs = sum(len(patient_groups[pid]) for pid in test_patients if patient_labels[pid] == 'malignant')

print(f"\nImage distribution by class:")
print(f"  Training - Benign: {train_benign_imgs}, Malignant: {train_malignant_imgs}")
print(f"  Validation - Benign: {val_benign_imgs}, Malignant: {val_malignant_imgs}")
print(f"  Test - Benign: {test_benign_imgs}, Malignant: {test_malignant_imgs}")

# Populate split_info with stratified 3-way patient-level split details
split_info = {
    'split_type': 'stratified_3way_patient_level',
    'split_method': 'Two-stage StratifiedGroupKFold (no patient overlap)',
    'random_state': 42,
    'total_patients': len(patient_groups),
    'total_images': len(all_images),
    'stratified': True,
    'note': 'Validation and test are completely separate at patient level - no data leakage',
    'patients': {
        'total_benign': len(benign_patients),
        'total_malignant': len(malignant_patients),
        'training_benign': train_benign,
        'training_malignant': train_malignant,
        'training_ratio': f"{train_benign}:{train_malignant}" if train_benign > 0 else "N/A",
        'validation_benign': val_benign,
        'validation_malignant': val_malignant,
        'validation_ratio': f"{val_benign}:{val_malignant}" if val_benign > 0 else "N/A",
        'test_benign': test_benign,
        'test_malignant': test_malignant,
        'test_ratio': f"{test_benign}:{test_malignant}" if test_benign > 0 else "N/A",
    },
    'images': {
        'total_benign': len(all_benign_images),
        'total_malignant': len(all_malignant_images),
        'training_benign': train_benign_imgs,
        'training_malignant': train_malignant_imgs,
        'training_ratio': f"{train_benign_imgs}:{train_malignant_imgs}" if train_benign_imgs > 0 else "N/A",
        'validation_benign': val_benign_imgs,
        'validation_malignant': val_malignant_imgs,
        'validation_ratio': f"{val_benign_imgs}:{val_malignant_imgs}" if val_benign_imgs > 0 else "N/A",
        'test_benign': test_benign_imgs,
        'test_malignant': test_malignant_imgs,
        'test_ratio': f"{test_benign_imgs}:{test_malignant_imgs}" if test_benign_imgs > 0 else "N/A",
    }
}

# STEP 4: COPY IMAGES TO TRAIN/VALIDATION/TEST DIRECTORIES
print("\n" + "=" * 80)
print("STEP 4: Copying images to train/validation/test directories...")

def copy_images_by_patient(patient_list, dest_dir):
    """Copy all images for given patients to destination directory."""
    copied_count = {'benign': 0, 'malignant': 0}
    
    for patient_id in patient_list:
        label = patient_labels[patient_id]
        dest_category_dir = os.path.join(dest_dir, label)
        
        for img_path in patient_groups[patient_id]:
            dest_path = os.path.join(dest_category_dir, os.path.basename(img_path))
            shutil.copy(img_path, dest_path)
            copied_count[label] += 1
    
    return copied_count

print("\nCopying training images...")
train_copied = copy_images_by_patient(train_patients, train_dir)
print(f"  Benign: {train_copied['benign']} images")
print(f"  Malignant: {train_copied['malignant']} images")

print("\nCopying validation images...")
val_copied = copy_images_by_patient(validation_patients, validation_dir)
print(f"  Benign: {val_copied['benign']} images")
print(f"  Malignant: {val_copied['malignant']} images")

print("\nCopying test images...")
test_copied = copy_images_by_patient(test_patients, test_dir)
print(f"  Benign: {test_copied['benign']} images")
print(f"  Malignant: {test_copied['malignant']} images")

# Verify no patient overlap between any sets
train_patient_set = set(train_patients)
val_patient_set = set(validation_patients)
test_patient_set = set(test_patients)

train_val_overlap = train_patient_set.intersection(val_patient_set)
train_test_overlap = train_patient_set.intersection(test_patient_set)
val_test_overlap = val_patient_set.intersection(test_patient_set)

print(f"\n✓ Verification: Patient overlap between sets:")
print(f"  Train/Validation overlap: {len(train_val_overlap)}")
print(f"  Train/Test overlap: {len(train_test_overlap)}")
print(f"  Validation/Test overlap: {len(val_test_overlap)}")

if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
    print("  ✓ SUCCESS! No data leakage - all patients are properly separated across all sets")
else:
    overlaps = []
    if len(train_val_overlap) > 0:
        overlaps.append(f"Train/Val: {len(train_val_overlap)}")
    if len(train_test_overlap) > 0:
        overlaps.append(f"Train/Test: {len(train_test_overlap)}")
    if len(val_test_overlap) > 0:
        overlaps.append(f"Val/Test: {len(val_test_overlap)}")
    print(f"  ⚠ WARNING! Found overlaps: {', '.join(overlaps)}")

# STEP 5: HANDLE CLASS IMBALANCE WITH AUGMENTATION
print("\n" + "=" * 80)
print("STEP 5: Validating training images and handling class imbalance...")

# First, validate all training images and remove corrupted ones
print("\nValidating training images...")
valid_images_count = 0
corrupted_files_to_remove = []

for category in ['benign', 'malignant']:
    category_dir = os.path.join(train_dir, category)
    images = glob.glob(os.path.join(category_dir, '*.png'))
    
    for img_path in images:
        try:
            img = tf.keras.preprocessing.image.load_img(img_path)
            valid_images_count += 1
        except Exception as e:
            print(f"⚠ Corrupted image found: {os.path.basename(img_path)}")
            print(f"  Path: {img_path}")
            corrupted_files_to_remove.append(img_path)

print(f"Valid images: {valid_images_count}")
if corrupted_files_to_remove:
    print(f"Corrupted images found: {len(corrupted_files_to_remove)}")
    print("Removing corrupted files...")
    for img_path in corrupted_files_to_remove:
        try:
            os.remove(img_path)
            print(f"  Removed: {os.path.basename(img_path)}")
        except Exception as e:
            print(f"  Could not remove {img_path}: {e}")
else:
    print("No corrupted images found.")

print("\nProceeding with class balance adjustment...")

# Calculate the number of images in each class AFTER patient-level split
class_counts = {category: len(glob.glob(os.path.join(train_dir, category, '*.png'))) 
                for category in ['benign', 'malignant']}
print(f"\nClass distribution in training set (after patient-level split):")
print(f"  Benign: {class_counts['benign']}")
print(f"  Malignant: {class_counts['malignant']}")

# Determine underrepresented class
mean_count = np.mean(list(class_counts.values()))
underrepresented_class = [category for category, count in class_counts.items() if count < mean_count]
print(f"  Underrepresented class: {underrepresented_class}")
print(f"  Mean count: {mean_count:.0f}")

# Custom augmentation pipeline for underrepresented class

aug_pipeline = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.2),
    iaa.Affine(rotate=(-45, 45)),
    iaa.Multiply((0.8, 1.2)),
    iaa.GaussianBlur(sigma=(0.0, 3.0)),
    iaa.AdditiveGaussianNoise(scale=(0.01*255, 0.05*255))
])

print("\nAugmenting underrepresented class...")
# Augment and add images for underrepresented class
corrupted_images = []
for category in underrepresented_class:
    category_dir = os.path.join(train_dir, category)
    images = glob.glob(os.path.join(category_dir, '*.png'))
    
    augmented_count = 0
    augs_per_image = int(mean_count / len(images))
    
    print(f"\n  Processing {category} class:")
    print(f"    Original images: {len(images)}")
    print(f"    Augmentations per image: {augs_per_image}")
    
    for idx, img_path in enumerate(images):
        try:
            img = tf.keras.preprocessing.image.load_img(img_path)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            for i in range(augs_per_image):
                aug_images = aug_pipeline(images=img_array.astype(np.uint8))
                for aug_img in aug_images:
                    aug_img_path = os.path.join(category_dir, f"aug_{augmented_count:05d}_{os.path.basename(img_path)}")
                    aug_img_pil = tf.keras.preprocessing.image.array_to_img(aug_img)
                    aug_img_pil.save(aug_img_path)
                    augmented_count += 1
        except Exception as e:
            print(f"\n    ⚠ WARNING: Could not load image: {img_path}")
            print(f"      Error: {type(e).__name__}: {str(e)}")
            corrupted_images.append(img_path)
        
        # Progress indicator every 100 images
        if (idx + 1) % 100 == 0:
            print(f"    Processed {idx + 1}/{len(images)} images, created {augmented_count} augmentations...")
    
    print(f"    Total augmented images created: {augmented_count}")

if corrupted_images:
    print(f"\n⚠ Found {len(corrupted_images)} corrupted or invalid image files:")
    for img_path in corrupted_images:
        print(f"  - {img_path}")
    print("\nThese images will be skipped during augmentation.")

# Verify final counts
final_class_counts = {category: len(glob.glob(os.path.join(train_dir, category, '*.png'))) 
                      for category in ['benign', 'malignant']}
print(f"\nFinal class distribution in training set (after augmentation):")
print(f"  Benign: {final_class_counts['benign']}")
print(f"  Malignant: {final_class_counts['malignant']}")
ratio = final_class_counts['benign'] / final_class_counts['malignant']
print(f"  Ratio (Benign/Malignant): {ratio:.2f}")

# STEP 6: DATA AUGMENTATION SETUP (RE-CREATE GENERATORS)
print("\n" + "=" * 80)
print("STEP 6: Setting up data augmentation and recreating generators...")

# General data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.9, 1.1],
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators (after augmentation)
print("\nCreating data generators with augmented training set...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = val_test_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# STEP 7: BUILD MODEL
print("\n" + "=" * 80)
print("STEP 7: Building model architecture...")

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

# STEP 7: SETUP CALLBACKS
print("\n" + "=" * 80)
print("STEP 7: Setting up callbacks...")

checkpoint_filepath = os.path.join(CHECKPOINT_DIR, 'best_model_patient_split.keras')
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

# STEP 8: COMPUTE CLASS WEIGHTS
print("\n" + "=" * 80)
print("STEP 8: Computing class weights...")

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

print(f"Class weights: {class_weights}")

# STEP 9: TRAIN MODEL
print("\n" + "=" * 80)
print("STEP 9: Training model for 37 epochs...")
print("=" * 80)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=37,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stopping, reduce_lr, lr_callback],
    verbose=1
)

# STEP 10: LOoade best model and evaluate
print("\n" + "=" * 80)
print("STEP 10: Loading best model and evaluating...")

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

# STEP 11: GENERATE PREDICTIONS AND METRICS
print("\n" + "=" * 80)
print("STEP 11: Generating predictions and metrics...")

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

# STEP 12: PLOT RESULTS
print("\n" + "=" * 80)
print("STEP 12: Plotting results...")

# Plot training & validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training and Validation Accuracy (Patient-Level Split)')
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
plt.title('Training and Validation Loss (Patient-Level Split)')
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
plt.title('Confusion Matrix - Test Set (Patient-Level Split)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print(f"Saved confusion matrix to: {os.path.join(output_dir, 'confusion_matrix.png')}")
plt.close()

# STEP 13: SAVE ARTIFACTS
print("\n" + "=" * 80)
print("STEP 13: Saving all artifacts and configurations...")

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
    'epochs': 37,
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
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\nExperiment timestamp: {timestamp}")
print(f"Output directory: {output_dir}")
print(f"\nBest model saved to: {output_model_path}")
print(f"Plots saved to: {output_dir}")
print(f"Configuration saved to: {output_dir}")
print(f"\nFinal Results (Patient-Level Split - No Data Leakage):")
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
print("=" * 80)