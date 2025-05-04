# Anomaly-Based Intrusion Detection Using Machine Learning on the CICIDS2017 Dataset


## 1. Setting Up the Development Environment

 Python environment with all the necessary libraries:

```python
# implementation_setup.py
import sys
import subprocess

def install_required_packages():
    """Install required packages for the project."""
    required_packages = [
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tensorflow',
        'keras',
        'imbalanced-learn',  # For SMOTE and other imbalanced data techniques
        'xgboost',
        'lightgbm',
        'shap'  # For model explainability
    ]
    
    for package in required_packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    print("All required packages installed successfully!")

if __name__ == "__main__":
    install_required_packages()
```

## 2. Downloading and Loading the CICIDS2017 Dataset

The CICIDS2017 dataset is large and typically split into multiple CSV files. Here's how to download and load it:

```python
# data_loader.py
import os
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from zipfile import ZipFile

def download_cicids2017():
    """
    Download the CICIDS2017 dataset.
    Note: This is a placeholder. You may need to adjust the URL or download method.
    """
    # The dataset is typically available from the Canadian Institute for Cybersecurity
    # You may need to register or use a different source
    print("Please download the CICIDS2017 dataset from the Canadian Institute for Cybersecurity")
    print("URL: https://www.unb.ca/cic/datasets/ids-2017.html")
    print("After downloading, extract the files to a folder named 'data' in your project directory")

def load_cicids2017(data_path='./data'):
    """
    Load the CICIDS2017 dataset from CSV files.
    
    Args:
        data_path: Path to the directory containing the dataset files
        
    Returns:
        A pandas DataFrame containing the combined dataset
    """
    # List of CSV files in the dataset
    csv_files = [
        'Monday-WorkingHours.pcap_ISCX.csv',
        'Tuesday-WorkingHours.pcap_ISCX.csv',
        'Wednesday-workingHours.pcap_ISCX.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    ]
    
    # Initialize an empty list to store DataFrames
    dfs = []
    
    # Load each CSV file
    for file in csv_files:
        file_path = os.path.join(data_path, file)
        if os.path.exists(file_path):
            print(f"Loading {file}...")
            # Handle potential encoding issues and inconsistent column names
            try:
                df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            except:
                df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
            
            # Standardize column names
            df.columns = [col.strip() for col in df.columns]
            
            dfs.append(df)
        else:
            print(f"Warning: {file_path} not found.")
    
    # Combine all DataFrames
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Dataset loaded successfully with {combined_df.shape[0]} rows and {combined_df.shape[1]} columns.")
        return combined_df
    else:
        print("No data files were found. Please check the data path.")
        return None

if __name__ == "__main__":
    download_cicids2017()
    # Uncomment the line below after downloading the dataset
    # df = load_cicids2017()
```

## 3. Exploratory Data Analysis (EDA)


```python
# exploratory_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_dataset(df):
    """
    Perform exploratory data analysis on the CICIDS2017 dataset.
    
    Args:
        df: Pandas DataFrame containing the dataset
    """
    # Basic information
    print("Dataset shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    
    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Check for infinite values
    print("\nInfinite values per column:")
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum()
    print(inf_count[inf_count > 0])
    
    # Class distribution
    print("\nClass distribution:")
    print(df['Label'].value_counts())
    
    # Visualize class distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Label', data=df)
    plt.title('Class Distribution')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    
    # Feature correlation
    plt.figure(figsize=(20, 16))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=False, cmap='coolwarm')
    plt.title('Feature Correlation')
    plt.tight_layout()
    plt.savefig('feature_correlation.png')
    
    # Feature distributions
    numeric_columns = numeric_df.columns[:10]  # First 10 numeric columns for brevity
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(2, 5, i)
        sns.histplot(df[column], kde=True)
        plt.title(column)
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    
    print("EDA completed. Visualizations saved as PNG files.")

if __name__ == "__main__":
    # Load the dataset
    # Assuming you've already run data_loader.py
    df = pd.read_csv('./data/combined_dataset.csv')  # Adjust path as needed
    explore_dataset(df)
```

## 4. Data Preprocessing


```python
# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess_data(df, binary_classification=True):
    """
    Preprocess the CICIDS2017 dataset for machine learning.
    
    Args:
        df: Pandas DataFrame containing the dataset
        binary_classification: If True, convert labels to binary (Normal vs. Attack)
        
    Returns:
        X_train, X_test, y_train, y_test: Preprocessed data splits
    """
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Handle missing values
    print("Handling missing values...")
    for column in data.columns:
        if data[column].dtype in [np.float64, np.int64]:
            data[column] = data[column].fillna(data[column].median())
        else:
            data[column] = data[column].fillna(data[column].mode()[0])
    
    # Handle infinite values
    print("Handling infinite values...")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        data[column] = data[column].replace([np.inf, -np.inf], np.nan)
        data[column] = data[column].fillna(data[column].median())
    
    # Convert labels
    print("Processing labels...")
    if binary_classification:
        # Convert to binary classification (Normal vs. Attack)
        data['Label'] = data['Label'].apply(lambda x: 'Normal' if x == 'BENIGN' else 'Attack')
    
    # Encode labels
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'])
    
    # Split features and target
    X = data.drop('Label', axis=1)
    y = data['Label']
    
    # Handle categorical features if any
    X = pd.get_dummies(X)
    
    # Split data into training and testing sets
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Handle class imbalance using SMOTE
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print("Data preprocessing completed.")
    return X_train_resampled, X_test, y_train_resampled, y_test, label_encoder.classes_

if __name__ == "__main__":
    # Load the dataset
    # Assuming you've already run data_loader.py
    df = pd.read_csv('./data/combined_dataset.csv')  # Adjust path as needed
    X_train, X_test, y_train, y_test, classes = preprocess_data(df)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    print(f"Class labels: {classes}")
```

## 5. Feature Selection

 feature selection to improve model performance:

```python
# feature_selection.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

def select_features(X_train, X_test, y_train, feature_names, method='random_forest', k=20):
    """
    Perform feature selection on the dataset.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        feature_names: Names of the features
        method: Feature selection method ('random_forest' or 'anova')
        k: Number of top features to select
        
    Returns:
        X_train_selected, X_test_selected: Datasets with selected features
        selected_features: Names of selected features
    """
    if method == 'random_forest':
        print("Performing feature selection using Random Forest importance...")
        # Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Get feature importances
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Select top k features
        top_indices = indices[:k]
        X_train_selected = X_train[:, top_indices]
        X_test_selected = X_test[:, top_indices]
        
        # Get names of selected features
        selected_features = [feature_names[i] for i in top_indices]
        
        # Visualize feature importances
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.bar(range(k), importances[top_indices], align='center')
        plt.xticks(range(k), selected_features, rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importances.png')
        
    elif method == 'anova':
        print("Performing feature selection using ANOVA F-value...")
        # ANOVA F-value for feature selection
        selector = SelectKBest(f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get indices of selected features
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        
        # Visualize F-scores
        plt.figure(figsize=(12, 8))
        plt.title('Feature F-scores')
        plt.bar(range(k), selector.scores_[selected_indices], align='center')
        plt.xticks(range(k), selected_features, rotation=90)
        plt.tight_layout()
        plt.savefig('feature_fscores.png')
    
    print(f"Selected {k} features: {', '.join(selected_features[:5])}...")
    return X_train_selected, X_test_selected, selected_features

if __name__ == "__main__":
    # Load preprocessed data
    # Assuming you've already run data_preprocessing.py
    X_train = np.load('./data/X_train.npy')
    X_test = np.load('./data/X_test.npy')
    y_train = np.load('./data/y_train.npy')
    
    # Load feature names
    feature_names = pd.read_csv('./data/feature_names.csv')['Feature'].tolist()
    
    # Select features
    X_train_selected, X_test_selected, selected_features = select_features(
        X_train, X_test, y_train, feature_names, method='random_forest', k=20
    )
    
    print(f"Original feature count: {X_train.shape[1]}")
    print(f"Selected feature count: {X_train_selected.shape[1]}")
```

## 6. Model Training and Evaluation

 implement and evaluate multiple machine learning models:

```python
# model_training.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import time
import joblib

def train_and_evaluate_models(X_train, X_test, y_train, y_test, class_names):
    """
    Train and evaluate multiple machine learning models.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        class_names: Names of the classes
        
    Returns:
        results: Dictionary containing model performances
    """
    # Define models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    }
    
    # Dictionary to store results
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate ROC curve and AUC (for binary classification)
        if len(class_names) == 2:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = None, None, None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Training time
        training_time = time.time() - start_time
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'training_time': training_time,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc
        }
        
        # Print results
        print(f"{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if roc_auc is not None:
            print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")
        
        # Save model
        joblib.dump(model, f'./models/{name.replace(" ", "_").lower()}.pkl')
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'./plots/confusion_matrix_{name.replace(" ", "_").lower()}.png')
    
    # Plot ROC curves (for binary classification)
    if len(class_names) == 2:
        plt.figure(figsize=(10, 8))
        for name, result in results.items():
            plt.plot(result['fpr'], result['tpr'], label=f'{name} (AUC = {result["roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.savefig('./plots/roc_curves.png')
    
    # Compare model performances
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    precisions = [results[name]['precision'] for name in model_names]
    recalls = [results[name]['recall'] for name in model_names]
    f1_scores = [results[name]['f1_score'] for name in model_names]
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(model_names))
    width = 0.2
    
    plt.bar(x - width*1.5, accuracies, width, label='Accuracy')
    plt.bar(x - width/2, precisions, width, label='Precision')
    plt.bar(x + width/2, recalls, width, label='Recall')
    plt.bar(x + width*1.5, f1_scores, width, label='F1 Score')
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/model_comparison.png')
    
    print("\nModel evaluation completed. Results saved to files.")
    return results

if __name__ == "__main__":
    # Create directories for outputs
    import os
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./plots', exist_ok=True)
    
    # Load preprocessed data
    # Assuming you've already run feature_selection.py
    X_train = np.load('./data/X_train_selected.npy')
    X_test = np.load('./data/X_test_selected.npy')
    y_train = np.load('./data/y_train.npy')
    y_test = np.load('./data/y_test.npy')
    
    # Load class names
    class_names = np.load('./data/class_names.npy')
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, class_names)
```

## 7. Deep Learning Model Implementation

 deep learning model for anomaly detection:

```python
# deep_learning_model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import time

def build_ann_model(input_shape, num_classes):
    """
    Build an Artificial Neural Network model.
    
    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
        
    Returns:
        model: Compiled Keras model
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_evaluate_ann(X_train, X_test, y_train, y_test, class_names):
    """
    Train and evaluate an Artificial Neural Network model.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        class_names: Names of the classes
        
    Returns:
        results: Dictionary containing model performance
    """
    print("\nTraining Artificial Neural Network...")
    start_time = time.time()
    
    # Get input shape and number of classes
    input_shape = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    # Build the model
    model = build_ann_model(input_shape, num_classes)
    
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('./models/ann_model.h5', save_best_only=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Make predictions
    y_pred_proba = model.predict(X_test)
    if num_classes > 2:
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Calculate ROC curve and AUC (for binary classification)
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr, roc_auc = None, None, None
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Training time
    training_time = time.time() - start_time
    
    # Store results
    results = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'training_time': training_time,
        'history': history.history,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }
    
    # Print results
    print("Artificial Neural Network Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('./plots/ann_training_history.png')
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Artificial Neural Network')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('./plots/confusion_matrix_ann.png')
    
    # Plot ROC curve (for binary classification)
    if num_classes == 2:
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ANN (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Artificial Neural Network')
        plt.legend(loc='lower right')
        plt.savefig('./plots/roc_curve_ann.png')
    
    print("ANN evaluation completed. Results saved to files.")
    return results

if __name__ == "__main__":
    # Create directories for outputs
    import os
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./plots', exist_ok=True)
    
    # Load preprocessed data
    # Assuming you've already run feature_selection.py
    X_train = np.load('./data/X_train_selected.npy')
    X_test = np.load('./data/X_test_selected.npy')
    y_train = np.load('./data/y_train.npy')
    y_test = np.load('./data/y_test.npy')
    
    # Load class names
    class_names = np.load('./data/class_names.npy')
    
    # Train and evaluate ANN model
    results = train_and_evaluate_ann(X_train, X_test, y_train, y_test, class_names)
```

## 8. Model Deployment and Real-time Detection

 simple system for real-time anomaly detection:

```python
# anomaly_detection_system.py
import numpy as np
import pandas as pd
import joblib
import time
import os
from sklearn.preprocessing import StandardScaler

class AnomalyDetectionSystem:
    """
    A system for real-time anomaly-based intrusion detection.
    """
    
    def __init__(self, model_path, scaler_path, feature_names_path, selected_features_path=None):
        """
        Initialize the anomaly detection system.
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the feature scaler
            feature_names_path: Path to the feature names
            selected_features_path: Path to the selected features (optional)
        """
        # Load the model
        self.model = joblib.load(model_path)
        
        # Load the scaler
        self.scaler = joblib.load(scaler_path)
        
        # Load feature names
        self.feature_names = pd.read_csv(feature_names_path)['Feature'].tolist()
        
        # Load selected features if provided
        if selected_features_path and os.path.exists(selected_features_path):
            with open(selected_features_path, 'r') as f:
                self.selected_features = [line.strip() for line in f.readlines()]
            print(f"Using {len(self.selected_features)} selected features.")
        else:
            self.selected_features = self.feature_names
            print(f"Using all {len(self.feature_names)} features.")
        
        print("Anomaly Detection System initialized successfully.")
    
    def preprocess_data(self, data):
        """
        Preprocess the input data for prediction.
        
        Args:
            data: Input data as a dictionary or pandas DataFrame
            
        Returns:
            processed_data: Preprocessed data ready for prediction
        """
        # Convert to DataFrame if it's a dictionary
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Ensure all required features are present
        for feature in self.selected_features:
            if feature not in data.columns:
                data[feature] = 0  # Default value for missing features
        
        # Select only the required features
        data = data[self.selected_features]
        
        # Scale the features
        processed_data = self.scaler.transform(data)
        
        return processed_data
    
    def detect_anomaly(self, data):
        """
        Detect if the input data represents an anomaly.
        
        Args:
            data: Input data as a dictionary or pandas DataFrame
            
        Returns:
            result: Dictionary containing prediction results
        """
        # Preprocess the data
        processed_data = self.preprocess_data(data)
        
        # Make prediction
        start_time = time.time()
        
        # Get prediction and probability
        if hasattr(self.model, 'predict_proba'):
            prediction = self.model.predict(processed_data)[0]
            probabilities = self.model.predict_proba(processed_data)[0]
        else:
            prediction = self.model.predict(processed_data)[0]
            probabilities = None
        
        # Calculate detection time
        detection_time = time.time() - start_time
        
        # Prepare result
        result = {
            'prediction': int(prediction),
            'is_anomaly': bool(prediction),  # Assuming 0 is normal, 1 is anomaly
            'confidence': float(max(probabilities)) if probabilities is not None else None,
            'detection_time': detection_time
        }
        
        return result
    
    def batch_detect(self, data_batch):
        """
        Perform batch anomaly detection on multiple data points.
        
        Args:
            data_batch: Batch of data as a pandas DataFrame
            
        Returns:
            results: List of detection results
        """
        # Preprocess the batch
        processed_batch = self.preprocess_data(data_batch)
        
        # Make predictions
        start_time = time.time()
        
        # Get predictions and probabilities
        predictions = self.model.predict(processed_batch)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(processed_batch)
        else:
            probabilities = None
        
        # Calculate total detection time
        total_detection_time = time.time() - start_time
        
        # Prepare results
        results = []
        for i, prediction in enumerate(predictions):
            result = {
                'prediction': int(prediction),
                'is_anomaly': bool(prediction),  # Assuming 0 is normal, 1 is anomaly
                'confidence': float(max(probabilities[i])) if probabilities is not None else None,
                'detection_time': total_detection_time / len(predictions)
            }
            results.append(result)
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize the system
    system = AnomalyDetectionSystem(
        model_path='./models/random_forest.pkl',
        scaler_path='./models/scaler.pkl',
        feature_names_path='./data/feature_names.csv',
        selected_features_path='./data/selected_features.txt'
    )
    
    # Example data point (replace with actual network flow data)
    example_data = {
        'Flow Duration': 100,
        'Total Fwd Packets': 5,
        'Total Backward Packets': 3,
        # Add other features as needed
    }
    
    # Detect anomaly
    result = system.detect_anomaly(example_data)
    
    # Print result
    print("\nAnomaly Detection Result:")
    print(f"Prediction: {'Anomaly' if result['is_anomaly'] else 'Normal'}")
    if result['confidence'] is not None:
        print(f"Confidence: {result['confidence']:.4f}")
    print(f"Detection Time: {result['detection_time']*1000:.2f} ms")
```

## 9. Running the Complete Pipeline

 script to run the complete pipeline:

```python
# run_pipeline.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Create necessary directories
os.makedirs('./data', exist_ok=True)
os.makedirs('./models', exist_ok=True)
os.makedirs('./plots', exist_ok=True)

# Step 1: Install required packages
print("Step 1: Installing required packages...")
from implementation_setup import install_required_packages
install_required_packages()

# Step 2: Download and load the dataset
print("\nStep 2: Loading the dataset...")
from data_loader import download_cicids2017, load_cicids2017
download_cicids2017()
# Uncomment after downloading the dataset
# df = load_cicids2017()
# df.to_csv('./data/combined_dataset.csv', index=False)

# For demonstration, we'll assume the dataset is already downloaded
print("For this demonstration, we'll assume the dataset is already downloaded.")
print("Please download the CICIDS2017 dataset and place it in the './data' directory.")
print("Then uncomment the relevant lines in this script.")

# Step 3: Perform exploratory data analysis
print("\nStep 3: Performing exploratory data analysis...")
# Uncomment after loading the dataset
# from exploratory_analysis import explore_dataset
# explore_dataset(df)

# Step 4: Preprocess the data
print("\nStep 4: Preprocessing the data...")
# Uncomment after loading the dataset
# from data_preprocessing import preprocess_data
# X_train, X_test, y_train, y_test, classes = preprocess_data(df)
# 
# # Save preprocessed data
# np.save('./data/X_train.npy', X_train)
# np.save('./data/X_test.npy', X_test)
# np.save('./data/y_train.npy', y_train)
# np.save('./data/y_test.npy', y_test)
# np.save('./data/class_names.npy', classes)
# 
# # Save feature names
# pd.DataFrame({'Feature': df.drop('Label', axis=1).columns}).to_csv('./data/feature_names.csv', index=False)
# 
# # Save scaler
# scaler = StandardScaler()
# scaler.fit(df.drop('Label', axis=1))
# joblib.dump(scaler, './models/scaler.pkl')

# Step 5: Perform feature selection
print("\nStep 5: Performing feature selection...")
# Uncomment after preprocessing the data
# from feature_selection import select_features
# feature_names = pd.read_csv('./data/feature_names.csv')['Feature'].tolist()
# X_train_selected, X_test_selected, selected_features = select_features(
#     X_train, X_test, y_train, feature_names, method='random_forest', k=20
# )
# 
# # Save selected features
# np.save('./data/X_train_selected.npy', X_train_selected)
# np.save('./data/X_test_selected.npy', X_test_selected)
# with open('./data/selected_features.txt', 'w') as f:
#     for feature in selected_features:
#         f.write(f"{feature}\n")

# Step 6: Train and evaluate machine learning models
print("\nStep 6: Training and evaluating machine learning models...")
# Uncomment after feature selection
# from model_training import train_and_evaluate_models
# results_ml = train_and_evaluate_models(X_train_selected, X_test_selected, y_train, y_test, classes)

# Step 7: Train and evaluate deep learning model
print("\nStep 7: Training and evaluating deep learning model...")
# Uncomment after feature selection
# from deep_learning_model import train_and_evaluate_ann
# results_dl = train_and_evaluate_ann(X_train_selected, X_test_selected, y_train, y_test, classes)

# Step 8: Initialize the anomaly detection system
print("\nStep 8: Initializing the anomaly detection system...")
# Uncomment after training models
# from anomaly_detection_system import AnomalyDetectionSystem
# system = AnomalyDetectionSystem(
#     model_path='./models/random_forest.pkl',
#     scaler_path='./models/scaler.pkl',
#     feature_names_path='./data/feature_names.csv',
#     selected_features_path='./data/selected_features.txt'
# )

print("\nPipeline execution completed!")
print("Note: Some steps were skipped in this demonstration.")
print("Please download the dataset and uncomment the relevant lines to run the complete pipeline.")
```
