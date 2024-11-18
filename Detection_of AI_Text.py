# Importing necessary libraries for further implementation and integration
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import keras_nlp
from keras.models import Model
from keras.layers import Activation
from keras.optimizers import AdamW
from keras.losses import BinaryCrossentropy
from keras.metrics import AUC
from keras.utils import to_categorical

# Updating the preprocessing, training, testing, and evaluation logic based on the notebook insights

# Define tokenizer setup using DeBERTaV3
def preprocess_texts(tokenizer, texts, max_seq_length):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=max_seq_length, return_tensors="tf")

# Unified preprocessing
def prepare_data(train_path, external_data_paths, tokenizer, max_seq_length):
    # Load main training data
    train_data = pd.read_csv(train_path)
    train_data['label'] = train_data['generated'].copy()

    # Load and process external data
    ext_data_frames = [pd.read_csv(path) for path in external_data_paths]
    for df in ext_data_frames:
        df.rename(columns={"model": "source"}, inplace=True)
        df["label"] = 1 if "generated" in df.columns else 0

    external_data = pd.concat(ext_data_frames, ignore_index=True)

    # Combine datasets
    combined_data = pd.concat([train_data, external_data], ignore_index=True)
    combined_data['text'] = combined_data['text'].fillna('')  # Handle missing text values

    # Tokenize text
    tokenized_texts = preprocess_texts(tokenizer, combined_data['text'].tolist(), max_seq_length)
    labels = combined_data['label'].values

    return tokenized_texts, labels

# Build the DebertaV3Classifier
def build_model(sequence_length, num_classes):
    classifier = keras_nlp.models.DebertaV3Classifier.from_preset(
        "deberta_v3_base_en", num_classes=num_classes
    )
    inputs = classifier.input
    logits = classifier(inputs)
    outputs = Activation("sigmoid")(logits)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=AdamW(5e-6),
        loss=BinaryCrossentropy(label_smoothing=0.02),
        metrics=[AUC(name="auc")]
    )
    return model

# Train the model
def train_model(model, tokenized_texts, labels, batch_size, epochs, val_split=0.2):
    X_train, X_val, y_train, y_val = train_test_split(
        tokenized_texts["input_ids"], labels, test_size=val_split, random_state=42
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs
    )
    return model, history

# Evaluate the model
def evaluate_model(model, test_texts, test_labels, tokenizer, sequence_length):
    test_tokenized = preprocess_texts(tokenizer, test_texts, sequence_length)
    predictions = model.predict(test_tokenized["input_ids"])
    predicted_labels = (predictions > 0.5).astype(int)
    print(classification_report(test_labels, predicted_labels))
    auc_score = roc_auc_score(test_labels, predictions)
    print(f"AUC Score: {auc_score:.4f}")

# Main workflow to integrate all steps
def main():
    # Paths and configurations
    train_path = "/path/to/train_essays.csv"
    external_data_paths = ["/path/to/external_data1.csv", "/path/to/external_data2.csv"]
    max_seq_length = 200
    batch_size = 8
    epochs = 3

    # Initialize tokenizer
    tokenizer = keras_nlp.tokenizers.DebertaV3Tokenizer.from_preset("deberta_v3_base_en")

    # Preprocessing
    tokenized_data, labels = prepare_data(train_path, external_data_paths, tokenizer, max_seq_length)

    # Build model
    model = build_model(sequence_length=max_seq_length, num_classes=1)

    # Train model
    trained_model, training_history = train_model(model, tokenized_data, labels, batch_size, epochs)

    # Testing and evaluation
    test_path = "/path/to/test_essays.csv"
    test_data = pd.read_csv(test_path)
    evaluate_model(trained_model, test_data["text"].tolist(), test_data["label"].tolist(), tokenizer, max_seq_length)

if __name__ == "__main__":
    main()
