import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedGroupKFold
import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import glob
import seaborn as sns
from collections import defaultdict
from datetime import datetime
import json
import time
import cv2

# GRID SEARCH CONFIGURATION
print("=" * 80)
print("BREAKHIS GRID SEARCH - COMPREHENSIVE HYPERPARAMETER SEARCH")
print("=" * 80)

# Base configuration
BASE_DIR = '/gpfs0/bgu-rriemer/users/reifk/data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast'
CHECKPOINT_DIR = '/gpfs0/bgu-rriemer/users/reifk/efficientnet_checkpoints'
GRID_SEARCH_OUTPUT_DIR = 'grid_search_results'

# Training configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 37
LEARNING_RATE = 1e-4
RANDOM_STATE = 42

# Grid search parameters
GRID_PARAMS = {
    'class_balance': ['balanced_50_50', 'imbalanced'],
    'dropout_rate': [0.3, 0.5, 0.8],  
    'loss_function': ['binary_crossentropy', 'focal_loss'],  
}

# Calculate total experiments
total_experiments = (len(GRID_PARAMS['class_balance']) * 
                    len(GRID_PARAMS['dropout_rate']) * 
                    len(GRID_PARAMS['loss_function']))

print(f"\nGrid Search Configuration:")
print(f"  Class Balance: {GRID_PARAMS['class_balance']}")
print(f"  Dropout Rates: {GRID_PARAMS['dropout_rate']}")
print(f"  Loss Functions: {GRID_PARAMS['loss_function']}")
print(f"\nTotal experiments: {total_experiments}")
print(f"Estimated time: {total_experiments * 1.5:.1f} hours ({total_experiments * 90:.0f} minutes)")
print("=" * 80)

# Set random seeds
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Create main grid search output directory
os.makedirs(GRID_SEARCH_OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# FOCAL LOSS IMPLEMENTATION
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for addressing class imbalance
    gamma: focusing parameter (higher = focus more on hard examples)
    alpha: weighting factor (balance between classes)
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.math.pow(1 - y_pred, gamma)
        
        cross_entropy_neg = -(1 - y_true) * tf.math.log(1 - y_pred)
        weight_neg = (1 - alpha) * (1 - y_true) * tf.math.pow(y_pred, gamma)
        
        loss = weight * cross_entropy + weight_neg * cross_entropy_neg
        return tf.reduce_mean(loss)
    
    return focal_loss_fixed

def extract_patient_id(filepath):
    """Extract patient ID from BreakHis filename"""
    filename = os.path.basename(filepath)
    try:
        name_without_ext = filename.replace('.png', '')
        parts = name_without_ext.split('-')
        if len(parts) >= 5:
            return parts[2]
        else:
            return filename
    except Exception as e:
        return filename

def get_tumor_type(filepath):
    """Extract tumor type from path"""
    if 'benign' in filepath:
        return 'benign'
    elif 'malignant' in filepath:
        return 'malignant'
    else:
        return 'unknown'

def copy_images_by_patient(patient_list, patient_groups, patient_labels, dest_dir):
    """Copy all images for given patients to destination directory"""
    copied_count = {'benign': 0, 'malignant': 0}
    
    for patient_id in patient_list:
        label = patient_labels[patient_id]
        dest_category_dir = os.path.join(dest_dir, label)
        
        for img_path in patient_groups[patient_id]:
            dest_path = os.path.join(dest_category_dir, os.path.basename(img_path))
            shutil.copy(img_path, dest_path)
            copied_count[label] += 1
    
    return copied_count

def create_confusion_matrix_plot(conf_matrix, class_names, title, output_path):
    """Create and save confusion matrix plot"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, 
                annot_kws={"size": 34})
    plt.xlabel('Predicted Label', fontsize=18)
    plt.ylabel('True Label', fontsize=18)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_training_plots(history, output_dir, experiment_name):
    """Create and save training history plots"""
    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Training and Validation Accuracy\n{experiment_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'])
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Training and Validation Loss\n{experiment_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

# MAIN GRID SEARCH LOOP
def run_experiment(exp_num, class_balance, dropout_rate, loss_function):
    """Run a single experiment with given hyperparameters"""
    
    # Create experiment name and directory
    exp_name = f"exp{exp_num:02d}_balance-{class_balance}_dropout-{dropout_rate}_loss-{loss_function}"
    exp_dir = os.path.join(GRID_SEARCH_OUTPUT_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "=" * 80)
    print(f"EXPERIMENT {exp_num}/{total_experiments}: {exp_name}")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    experiment_start_time = time.time()
    
    # Define directories
    benign_dir = os.path.join(BASE_DIR, 'benign')
    malignant_dir = os.path.join(BASE_DIR, 'malignant')
    
    train_dir = os.path.join(BASE_DIR, f'train_{exp_name}')
    validation_dir = os.path.join(BASE_DIR, f'validation_{exp_name}')
    test_dir = os.path.join(BASE_DIR, f'test_{exp_name}')
    
    # Clean up existing directories
    for directory in [train_dir, validation_dir, test_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)
        for category in ['benign', 'malignant']:
            os.makedirs(os.path.join(directory, category), exist_ok=True)
    
    # STEP 1: COLLECT AND SPLIT DATA
    print("\nSTEP 1: Collecting images and creating patient-level split...")
    
    all_benign_images = glob.glob(os.path.join(benign_dir, '**', '*.png'), recursive=True)
    all_malignant_images = glob.glob(os.path.join(malignant_dir, '**', '*.png'), recursive=True)
    all_images = all_benign_images + all_malignant_images
    
    # Group by patient
    patient_groups = defaultdict(list)
    patient_labels = {}
    
    for img_path in all_images:
        patient_id = extract_patient_id(img_path)
        tumor_type = get_tumor_type(img_path)
        patient_groups[patient_id].append(img_path)
        patient_labels[patient_id] = tumor_type
    
    # Patient-level stratified split
    patient_ids = list(patient_groups.keys())
    labels = [patient_labels[pid] for pid in patient_ids]
    label_map = {'benign': 0, 'malignant': 1}
    binary_labels = np.array([label_map[label] for label in labels])
    
    # Stage 1: 80/20 split
    sgkf_stage1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, temp_val_idx = next(sgkf_stage1.split(X=np.array(patient_ids).reshape(-1, 1), 
                                                       y=binary_labels, 
                                                       groups=np.array(patient_ids)))
    
    train_patients = [patient_ids[i] for i in train_idx]
    temp_val_patients = [patient_ids[i] for i in temp_val_idx]
    
    # Stage 2: 10/10 split
    temp_val_labels = np.array([label_map[patient_labels[pid]] for pid in temp_val_patients])
    sgkf_stage2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    val_idx, test_idx = next(sgkf_stage2.split(X=np.array(temp_val_patients).reshape(-1, 1), 
                                                y=temp_val_labels, 
                                                groups=np.array(temp_val_patients)))
    
    validation_patients = [temp_val_patients[i] for i in val_idx]
    test_patients = [temp_val_patients[i] for i in test_idx]
    
    print(f"  Training patients: {len(train_patients)}")
    print(f"  Validation patients: {len(validation_patients)}")
    print(f"  Test patients: {len(test_patients)}")
    
    # Copy images
    print("\n  Copying images...")
    train_copied = copy_images_by_patient(train_patients, patient_groups, patient_labels, train_dir)
    val_copied = copy_images_by_patient(validation_patients, patient_groups, patient_labels, validation_dir)
    test_copied = copy_images_by_patient(test_patients, patient_groups, patient_labels, test_dir)
    
    print(f"  Train - Benign: {train_copied['benign']}, Malignant: {train_copied['malignant']}")
    print(f"  Val - Benign: {val_copied['benign']}, Malignant: {val_copied['malignant']}")
    print(f"  Test - Benign: {test_copied['benign']}, Malignant: {test_copied['malignant']}")
    
    # STEP 2: BALANCE TRAINING SET (if 50-50 balance enabled)
    if class_balance == 'balanced_50_50':
        print("\nSTEP 3: Balancing training set to 50-50...")
        
        class_counts = {category: len(glob.glob(os.path.join(train_dir, category, '*.png'))) 
                        for category in ['benign', 'malignant']}
        
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        majority_count = class_counts[majority_class]
        minority_count = class_counts[minority_class]
        
        target_count = majority_count
        augmentations_needed = target_count - minority_count
        
        print(f"  Majority: {majority_class} ({majority_count})")
        print(f"  Minority: {minority_class} ({minority_count})")
        print(f"  Augmentations needed: {augmentations_needed}")
        
        # Augmentation pipeline
        aug_pipeline = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.2),
            iaa.Affine(rotate=(-45, 45)),
            iaa.Multiply((0.8, 1.2)),
            iaa.GaussianBlur(sigma=(0.0, 3.0)),
            iaa.AdditiveGaussianNoise(scale=(0.01*255, 0.05*255))
        ], random_state=RANDOM_STATE)
        
        minority_dir = os.path.join(train_dir, minority_class)
        minority_images = glob.glob(os.path.join(minority_dir, '*.png'))
        
        augs_per_image = augmentations_needed // len(minority_images)
        remaining_augs = augmentations_needed % len(minority_images)
        
        augmented_count = 0
        for idx, img_path in enumerate(minority_images):
            img = tf.keras.preprocessing.image.load_img(img_path)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            num_augs = augs_per_image
            if idx < remaining_augs:
                num_augs += 1
            
            for i in range(num_augs):
                aug_images = aug_pipeline(images=img_array.astype(np.uint8))
                for aug_img in aug_images:
                    aug_img_path = os.path.join(minority_dir, f"aug_{augmented_count:05d}_{os.path.basename(img_path)}")
                    aug_img_pil = tf.keras.preprocessing.image.array_to_img(aug_img)
                    aug_img_pil.save(aug_img_path)
                    augmented_count += 1
        
        final_counts = {category: len(glob.glob(os.path.join(train_dir, category, '*.png'))) 
                        for category in ['benign', 'malignant']}
        print(f"  Final - Benign: {final_counts['benign']}, Malignant: {final_counts['malignant']}")
    else:
        # THIS IS THE "IMBALANCED" MODE
        print("\nSTEP 3: Validating images and applying Mean-Based Augmentation...")

        # 1. Validate and remove corrupted images
        print("\nValidating training images...")
        valid_images_count = 0
        corrupted_files_to_remove = []

        for category in ['benign', 'malignant']:
            category_dir = os.path.join(train_dir, category)
            images = glob.glob(os.path.join(category_dir, '*.png'))
            
            for img_path in images:
                try:
                    # Try to load the image to check for corruption
                    img = tf.keras.preprocessing.image.load_img(img_path)
                    valid_images_count += 1
                except Exception as e:
                    print(f"⚠ Corrupted image found: {os.path.basename(img_path)}")
                    corrupted_files_to_remove.append(img_path)

        if corrupted_files_to_remove:
            print(f"Corrupted images found: {len(corrupted_files_to_remove)}")
            print("Removing corrupted files...")
            for img_path in corrupted_files_to_remove:
                try:
                    os.remove(img_path)
                except Exception as e:
                    print(f"  Could not remove {img_path}: {e}")
        else:
            print("No corrupted images found.")

        # 2. Mean-based Augmentation Logic
        class_counts = {category: len(glob.glob(os.path.join(train_dir, category, '*.png'))) 
                        for category in ['benign', 'malignant']}
        
        mean_count = np.mean(list(class_counts.values()))
        underrepresented_class = [category for category, count in class_counts.items() if count < mean_count]
        
        print(f"  Mean count: {mean_count:.0f}")
        print(f"  Underrepresented class: {underrepresented_class}")

        aug_pipeline = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.2),
            iaa.Affine(rotate=(-45, 45)),
            iaa.Multiply((0.8, 1.2)),
            iaa.GaussianBlur(sigma=(0.0, 3.0)),
            iaa.AdditiveGaussianNoise(scale=(0.01*255, 0.05*255))
        ], random_state=RANDOM_STATE)

        for category in underrepresented_class:
            category_dir = os.path.join(train_dir, category)
            images = glob.glob(os.path.join(category_dir, '*.png'))
            
            if len(images) > 0:
                augmented_count = 0
                augs_per_image = int(mean_count / len(images))
                
                print(f"  Augmenting {category}: {len(images)} original images")
                print(f"  Augmentations per image: {augs_per_image}")
                
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
                         print(f"  Skipping {img_path}: {e}")
                
                print(f"  Created {augmented_count} new images for {category}")

        final_counts = {category: len(glob.glob(os.path.join(train_dir, category, '*.png'))) 
                        for category in ['benign', 'malignant']}
        print(f"  Final Imbalanced Distribution: {final_counts}")
    
    # STEP 3: CREATE DATA GENERATORS
    print("\nSTEP 3: Creating data generators...")
    
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
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=RANDOM_STATE
    )
    
    validation_generator = val_test_datagen.flow_from_directory(
        validation_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
        seed=RANDOM_STATE
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    # STEP 4: BUILD MODEL
    print("\nSTEP 4: Building model...")
    
    model = Sequential([
        EfficientNetB5(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights='imagenet'),
        GlobalAveragePooling2D(),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    ])
    
    # Select loss function
    if loss_function == 'focal_loss':
        loss_fn = focal_loss(gamma=2.0, alpha=0.25)
    else:
        loss_fn = 'binary_crossentropy'
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=['accuracy']
    )

    # STEP 4.5: COMPUTE CLASS WEIGHTS (for imbalanced mode only)
    class_weights = None
    if class_balance == 'imbalanced':
        print("\nSTEP 4.5: Computing class weights for imbalanced mode...")
        
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_generator.classes),
            y=train_generator.classes
        )
        class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
        print(f"  Class weights: {class_weights}")
        print("  → These weights will compensate for class imbalance in loss function")
    else:
        print("\nSTEP 4.5: Skipping class weights (balanced 50-50 mode - not needed)")
        
    
    # STEP 5: SETUP CALLBACKS
    print("\nSTEP 5: Setting up callbacks...")
    
    checkpoint_filepath = os.path.join(CHECKPOINT_DIR, f'{exp_name}_{timestamp}.keras')
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=0
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=0
    )
    
    # STEP 6: TRAIN MODEL
    print(f"\nSTEP 6: Training model for {EPOCHS} epochs...")
    
    # Apply class weights only for imbalanced mode
    if class_balance == 'imbalanced':
        print("  → Using weighted loss (class_weight applied)")
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=EPOCHS,
            class_weight=class_weights,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            verbose=2
        )
    else:
        print("  → Using unweighted loss (50-50 balanced data)")
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=EPOCHS,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            verbose=2
    )
    
    # STEP 7: EVALUATE MODEL
    print("\nSTEP 7: Evaluating model...")
    
    best_model = tf.keras.models.load_model(checkpoint_filepath, custom_objects={'focal_loss_fixed': focal_loss()} if loss_function == 'focal_loss' else None)
    
    # Validation metrics
    val_loss, val_accuracy = best_model.evaluate(validation_generator, verbose=0)
    
    # Test metrics
    test_loss, test_accuracy = best_model.evaluate(test_generator, verbose=0)
    
    # Predictions
    y_pred = best_model.predict(test_generator, verbose=0)
    y_pred_labels = (y_pred > 0.5).astype(int).flatten()
    y_true = test_generator.classes[:len(y_pred_labels)]
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred_labels)
    class_names = ['benign', 'malignant']
    
    # Classification report
    report_dict = classification_report(y_true, y_pred_labels, target_names=class_names, output_dict=True)
    
    print(f"\n  Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"  Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}")
    
    # STEP 8: SAVE RESULTS
    print("\nSTEP 8: Saving results...")
    
    # Save model
    output_model_path = os.path.join(exp_dir, 'best_model.keras')
    shutil.copy(checkpoint_filepath, output_model_path)
    
    # Save configuration
    config = {
    'experiment_number': exp_num,
    'experiment_name': exp_name,
    'timestamp': timestamp,
    'class_balance': class_balance,
    'dropout_rate': dropout_rate,
    'loss_function': loss_function,
    'class_weights_used': class_weights is not None,
    'class_weights': {str(k): float(v) for k, v in class_weights.items()} if class_weights else None,
    'epochs': EPOCHS,
    'learning_rate': LEARNING_RATE,
    'batch_size': BATCH_SIZE,
    'random_state': RANDOM_STATE,
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss),
    'val_accuracy': float(val_accuracy),
    'val_loss': float(val_loss)
}
    
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save metrics
    metrics = {
        'test': {
            'accuracy': float(test_accuracy),
            'loss': float(test_loss),
            'classification_report': report_dict
        },
        'validation': {
            'accuracy': float(val_accuracy),
            'loss': float(val_loss)
        }
    }
    
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save training history
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
    
    with open(os.path.join(exp_dir, 'training_history.json'), 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Create plots
    create_training_plots(history, exp_dir, exp_name)
    create_confusion_matrix_plot(conf_matrix, class_names, 
                                  f'Confusion Matrix - {exp_name}',
                                  os.path.join(exp_dir, 'confusion_matrix.png'))
    
    # Save confusion matrix data
    with open(os.path.join(exp_dir, 'confusion_matrix.json'), 'w') as f:
        json.dump({
            'confusion_matrix': conf_matrix.tolist(),
            'class_names': class_names
        }, f, indent=2)
    
    # Clean up temporary directories
    print("\nCleaning up temporary directories...")
    for directory in [train_dir, validation_dir, test_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    
    experiment_time = time.time() - experiment_start_time
    print(f"\nExperiment completed in {experiment_time/60:.1f} minutes")
    print(f"Results saved to: {exp_dir}")
    
    return {
        'experiment_number': exp_num,
        'experiment_name': exp_name,
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'val_accuracy': float(val_accuracy),
        'val_loss': float(val_loss),
        'time_minutes': experiment_time/60,
        'config': config
    }

# RUN ALL EXPERIMENTS
print("\n" + "=" * 80)
print("STARTING GRID SEARCH")
print("=" * 80)

grid_search_start_time = time.time()
all_results = []
exp_num = 1

for balance in GRID_PARAMS['class_balance']:
    for dropout in GRID_PARAMS['dropout_rate']:
        for loss_fn in GRID_PARAMS['loss_function']:
            try:
                result = run_experiment(exp_num, balance, dropout, loss_fn)
                all_results.append(result)
            except Exception as e:
                print(f"\n!!! ERROR in experiment {exp_num}: {e}")
                print("Continuing with next experiment...")
                
            exp_num += 1

# SUMMARIZE RESULTS
total_time = time.time() - grid_search_start_time

print("\n" + "=" * 80)
print("GRID SEARCH COMPLETE!")
print("=" * 80)
print(f"\nTotal time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
print(f"Experiments completed: {len(all_results)}/{total_experiments}")

# Save summary
summary = {
    'total_experiments': total_experiments,
    'completed_experiments': len(all_results),
    'total_time_hours': total_time/3600,
    'grid_parameters': GRID_PARAMS,
    'results': all_results
}

with open(os.path.join(GRID_SEARCH_OUTPUT_DIR, 'grid_search_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

# Print top 5 results by test accuracy
print("\nTop 5 Experiments by Test Accuracy:")
print("-" * 80)
sorted_results = sorted(all_results, key=lambda x: x['test_accuracy'], reverse=True)
for i, result in enumerate(sorted_results[:5], 1):
    print(f"{i}. {result['experiment_name']}")
    print(f"   Test Accuracy: {result['test_accuracy']*100:.2f}%")
    print(f"   Test Loss: {result['test_loss']:.4f}")
    print(f"   Val Accuracy: {result['val_accuracy']*100:.2f}%")
    print()

print(f"\nFull results saved to: {GRID_SEARCH_OUTPUT_DIR}/grid_search_summary.json")
print("=" * 80)
