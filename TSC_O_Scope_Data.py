"""
Time Series Classification for Electrode Identification using dl-4-tsc

Predicts which electrode is actuated based on EMF probe readings.
Uses the ResNet classifier from: https://github.com/hfawaz/dl-4-tsc

Usage:
    1. Prepare CSV: first column = label, remaining columns = time series EMF data
    2. Run: python TSC_O_Scope_Data.py --train data.csv --output ./results
    3. Predict: python TSC_O_Scope_Data.py --predict new_data.csv --model ./results
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure dl-4-tsc is available
try:
    from classifiers.resnet import Classifier_RESNET
except ImportError:
    print("dl-4-tsc not found. Clone it with:")
    print("  git clone https://github.com/hfawaz/dl-4-tsc.git")
    print("  cd dl-4-tsc && pip install -r requirements.txt")
    sys.exit(1)

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_csv_dataset(csv_path, test_size=0.2):
    """
    Load CSV dataset and split into train/test.
    
    CSV format: first column = label, remaining columns = EMF time series
    
    Returns:
        x_train, y_train, x_test, y_test, label_encoder
    """
    df = pd.read_csv(csv_path)
    
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.astype(np.float32)
    
    # Encode labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=SEED, stratify=y_encoded
    )
    
    return x_train, y_train, x_test, y_test, le


def normalize(x_train, x_test):
    """Z-normalize each time series independently."""
    def znorm(x):
        std = x.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (x - x.mean(axis=1, keepdims=True)) / std
    
    return znorm(x_train), znorm(x_test)


def prepare_for_resnet(x_train, y_train, x_test, y_test):
    """
    Prepare data for dl-4-tsc ResNet:
    - Reshape to (samples, timesteps, 1) for Conv1D
    - One-hot encode labels
    """
    # Reshape for Conv1D
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    
    # One-hot encode
    nb_classes = len(np.unique(np.concatenate([y_train, y_test])))
    enc = OneHotEncoder(sparse_output=False, categories='auto')
    y_all = np.concatenate([y_train, y_test]).reshape(-1, 1)
    enc.fit(y_all)
    
    y_train_onehot = enc.transform(y_train.reshape(-1, 1))
    y_test_onehot = enc.transform(y_test.reshape(-1, 1))
    
    return x_train, y_train_onehot, x_test, y_test_onehot, nb_classes


def train_resnet(x_train, y_train, x_test, y_test, output_dir, nb_epochs=500):
    """
    Train ResNet classifier from dl-4-tsc.
    
    Args:
        x_train: Training data (samples, timesteps, 1)
        y_train: One-hot encoded training labels
        x_test: Test data
        y_test: One-hot encoded test labels
        output_dir: Directory to save model
        nb_epochs: Number of training epochs
        
    Returns:
        Trained classifier, metrics DataFrame
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_shape = x_train.shape[1:]
    nb_classes = y_train.shape[1]
    
    print(f"\nTraining ResNet classifier")
    print(f"  Input shape: {input_shape}")
    print(f"  Classes: {nb_classes}")
    print(f"  Train samples: {len(x_train)}")
    print(f"  Test samples: {len(x_test)}")
    print(f"  Epochs: {nb_epochs}")
    
    # Initialize classifier
    classifier = Classifier_RESNET(
        output_directory=str(output_dir) + '/',
        input_shape=input_shape,
        nb_classes=nb_classes,
        verbose=True
    )
    
    # Train
    y_true = np.argmax(y_test, axis=1)
    df_metrics = classifier.fit(x_train, y_train, x_test, y_test, y_true)
    
    print(f"\nResults saved to: {output_dir}")
    return classifier, df_metrics


def predict(model_dir, x_new, label_encoder=None):
    """
    Predict electrode class for new EMF measurements.
    
    Args:
        model_dir: Directory containing trained model
        x_new: New data (samples, timesteps) or (samples, timesteps, 1)
        label_encoder: Optional LabelEncoder to decode predictions
        
    Returns:
        Predicted class labels
    """
    import tensorflow.keras as keras
    
    model_path = Path(model_dir) / 'best_model.hdf5'
    model = keras.models.load_model(str(model_path))
    
    # Ensure correct shape
    if len(x_new.shape) == 2:
        x_new = x_new.reshape(x_new.shape[0], x_new.shape[1], 1)
    
    # Z-normalize
    std = x_new.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    x_new = (x_new - x_new.mean(axis=1, keepdims=True)) / std
    
    # Predict
    y_pred_proba = model.predict(x_new)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred)
    
    return y_pred, y_pred_proba


def main():
    parser = argparse.ArgumentParser(description='EMF Time Series Classification')
    parser.add_argument('--train', type=str, help='Path to training CSV')
    parser.add_argument('--predict', type=str, help='Path to CSV for prediction')
    parser.add_argument('--model', type=str, default='./results', help='Model directory')
    parser.add_argument('--output', type=str, default='./results', help='Output directory')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    args = parser.parse_args()
    
    if args.train:
        # Load and prepare data
        print(f"Loading dataset: {args.train}")
        x_train, y_train, x_test, y_test, le = load_csv_dataset(args.train, args.test_size)
        
        # Normalize
        x_train, x_test = normalize(x_train, x_test)
        
        # Prepare for ResNet
        x_train, y_train_oh, x_test, y_test_oh, nb_classes = prepare_for_resnet(
            x_train, y_train, x_test, y_test
        )
        
        # Train
        classifier, metrics = train_resnet(
            x_train, y_train_oh, x_test, y_test_oh,
            args.output, nb_epochs=args.epochs
        )
        
        # Save label encoder
        np.save(Path(args.output) / 'label_classes.npy', le.classes_)
        
        print(f"\nTest Accuracy: {metrics['accuracy'].values[0]:.4f}")
        
    elif args.predict:
        # Load new data for prediction
        df = pd.read_csv(args.predict)
        
        # Check if first column is label or data
        if df.iloc[:, 0].dtype == 'object' or df.iloc[:, 0].nunique() < 100:
            x_new = df.iloc[:, 1:].values.astype(np.float32)
            true_labels = df.iloc[:, 0].values
        else:
            x_new = df.values.astype(np.float32)
            true_labels = None
        
        # Load label encoder
        le_path = Path(args.model) / 'label_classes.npy'
        le = LabelEncoder()
        if le_path.exists():
            le.classes_ = np.load(le_path, allow_pickle=True)
        else:
            le = None
        
        # Predict
        predictions, probabilities = predict(args.model, x_new, le)
        
        print("\nPredictions:")
        for i, pred in enumerate(predictions):
            conf = probabilities[i].max() * 100
            print(f"  Sample {i+1}: Electrode {pred} (confidence: {conf:.1f}%)")
            
        if true_labels is not None:
            acc = accuracy_score(true_labels, predictions)
            print(f"\nAccuracy: {acc:.4f}")
    
    else:
        parser.print_help()


# Example programmatic usage
def example_usage():
    """
    Example: Train and predict programmatically without CLI.
    """
    # Simulated data: 100 samples, 500 timesteps, 10 electrode classes
    np.random.seed(42)
    n_samples, n_timesteps, n_classes = 100, 500, 10
    
    X = np.random.randn(n_samples, n_timesteps).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize
    x_train, x_test = normalize(x_train, x_test)
    
    # Prepare
    x_train, y_train_oh, x_test, y_test_oh, nb_classes = prepare_for_resnet(
        x_train, y_train, x_test, y_test
    )
    
    # Train (reduced epochs for demo)
    classifier, metrics = train_resnet(
        x_train, y_train_oh, x_test, y_test_oh,
        output_dir='./demo_results',
        nb_epochs=10
    )
    
    # Predict new sample
    x_new = np.random.randn(1, n_timesteps).astype(np.float32)
    predictions, probs = predict('./demo_results', x_new)
    print(f"Predicted class: {predictions[0]}")


if __name__ == "__main__":
    main()
