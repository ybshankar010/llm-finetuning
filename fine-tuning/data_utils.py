import pickle
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import requests
import tarfile

def load_reviews(folder):
    reviews = []
    labels = []
    for label in ['pos', 'neg']:
        path = os.path.join("aclImdb", folder, label)
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), encoding="utf-8") as f:
                reviews.append(f.read())
                labels.append(1 if label == "pos" else 0)
    return reviews, labels

def get_data(folder="train", count_per_class=2500):
    texts, labels = load_reviews(folder)
    df = pd.DataFrame({"text": texts, "label": labels})
    n_per_class = count_per_class
    sampled = pd.concat([
        grp.sample(n=n_per_class, random_state=42)
        for _, grp in df.groupby("label")
    ]).sample(frac=1, random_state=42)
    return sampled["text"].tolist(), sampled["label"].tolist()

def prepare_and_save_data(save_path="./dataset.pkl"):
    """
    Prepare data and save to pickle file
    """
    print("🚀 Preparing data...")
    
    # Download and extract if needed
    DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    DATA_DIR = "aclImdb"
    
    if not os.path.exists(DATA_DIR):
        print("📥 Downloading dataset...")
        response = requests.get(DATA_URL, stream=True)
        with open("aclImdb_v1.tar.gz", "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024)):
                if chunk:
                    f.write(chunk)
        with tarfile.open("aclImdb_v1.tar.gz", "r:gz") as tar:
            tar.extractall()

    print("📊 Loading and splitting data...")
    train_texts, train_labels = get_data("train")
    test_texts, test_labels = get_data("test")
    X_train, X_val, y_train, y_val = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)
    
    # Create data dictionary
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': test_texts,
        'y_test': test_labels
    }
    
    # Save to pickle
    print(f"💾 Saving data to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ Data saved! Sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(test_texts)}")
    return X_train, y_train, X_val, y_val, test_texts, test_labels

def load_data(load_path="./dataset.pkl"):
    """
    Load data from pickle file
    """
    print(f"📂 Loading data from {load_path}...")
    
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"✅ Data loaded! Sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def prepare_data_smart(save_path="./dataset.pkl"):
    """
    Smart function: load if exists, otherwise prepare and save
    """
    if os.path.exists(save_path):
        print("📂 Found existing data file, loading...")
        return load_data(save_path)
    else:
        print("🚀 No data file found, preparing fresh data...")
        return prepare_and_save_data(save_path)
