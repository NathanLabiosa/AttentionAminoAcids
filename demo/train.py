import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    Bidirectional,
    LSTM,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
    Input,
    Concatenate,
    GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import argparse

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config


class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)  # Pass additional kwargs to superclass
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim)
        self.pos_emb = Embedding(input_dim=self.maxlen, output_dim=self.embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super(TokenAndPositionEmbedding, self).get_config()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim
        })
        return config

def prepare_sequence(seq, tokenizer, max_length):
    """Converts a protein sequence to a padded sequence of token IDs."""
    # Tokenize the sequence
    tokenized = tokenizer.texts_to_sequences([seq])
    # Pad sequence to have a uniform length
    padded = pad_sequences(tokenized, maxlen=max_length)
    return padded


def predict_class(seq, numerical_data, tokenizer, model, max_length, label_binarizer, scaler_path):
    """Predicts the class for a given protein sequence and its numerical data."""
    # Load the scaler
    scaler = joblib.load(scaler_path)  # Ensure the scaler was saved at the same location
    # Prepare the sequence data
    prepared_seq = prepare_sequence(seq, tokenizer, max_length)
    # Scale the numerical data
    prepared_num = numerical_data.reshape(1, -1)
    prepared_num = scaler.transform(prepared_num)  # Apply scaling
    # Predict using the model
    prediction = model.predict([prepared_seq, prepared_num])
    predicted_class = label_binarizer.inverse_transform(prediction)[0]
    return predicted_class

# Check for available GPUs
gpus = tf.config.experimental.list_physical_devices("GPU")
print("Num GPUs Available: ", len(gpus))

# Set memory growth for all GPUs
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print("Failed to set memory growth: ", e)

# Parameters
max_length = 350
num_classes = 30  # Adjust based on your actual number of classes
num_numerical_features = 12

# Model parameters
vocab_size = 26 + 1  # 26 letters in the protein alphabet + padding token
maxlen = 350
embed_dim = 100  # Embedding size for each token
num_heads = 5  # Number of attention heads
ff_dim = 256  # Hidden layer size in feed forward network inside transformer

# Load data
parser = argparse.ArgumentParser(description='Train protein classification model')
parser.add_argument('--data', type=str, help='Path to the CSV file containing the data')
parser.add_argument('--save_path', type=str, help='Path to the CSV file containing the data')
args = parser.parse_args()

data = pd.read_csv(args.data)

# Initialize and fit tokenizer
unique_chars = (
    "ACDEFGHIKLMNPQRSTVWY"  # Add any additional characters if they appear in your data
)
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(list(unique_chars))

# Convert sequences
sequences = tokenizer.texts_to_sequences(data["sequence"].apply(str))
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')

lb = LabelBinarizer()
lb.fit(data["classification"].unique())  # Fit to unique class labels
labels = lb.transform(data["classification"])


#residueCount,macromoleculeType,classification,structureMolecularWeight,crystallizationTempK,densityMatthews,densityPercentSol,phValue,Isoelectric_Point,Aromaticity,Instability_Index,Helix,Turn,Sheet

numerical_columns = ['residueCount', 'structureMolecularWeight', 'crystallizationTempK', 'densityMatthews', 'densityPercentSol', 'phValue', 'Isoelectric_Point', 'Aromaticity', 'Instability_Index', 'Helix', 'Turn', 'Sheet']  # Replace these with your actual column names
numerical_data = data[numerical_columns]

scaler = StandardScaler()
numerical_data_scaled = scaler.fit_transform(numerical_data)
# Save the scaler to disk
joblib.dump(scaler, args.save_path)

# Apply oversampling
ros = RandomOverSampler(random_state=42)

# Apply oversampling to both sequence and numerical features
X_resampled, y_resampled = ros.fit_resample(np.hstack((padded_sequences, numerical_data_scaled)), labels)

# Split numerical data back from sequences after resampling
X_resampled_sequences = X_resampled[:, :max_length]  # assuming sequence length columns come first
X_resampled_numerical = X_resampled[:, max_length:]  # the rest are numerical features

# Split into training and testing sets
# Split sequences and labels
X_train_sequences, X_test_sequences, y_train, y_test = train_test_split(
    X_resampled_sequences, y_resampled, test_size=0.2, random_state=42)

# Split numerical features
X_train_numerical, X_test_numerical = train_test_split(
    X_resampled_numerical, test_size=0.2, random_state=42)


# Sequence input and embedding
sequence_input = Input(shape=(maxlen,), name='sequence_input')
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
seq_embedding = embedding_layer(sequence_input)

# Transformer block
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = seq_embedding
for _ in range(4):
    x = transformer_block(x, training=True)

x = GlobalAveragePooling1D()(x)

# Numerical input
numerical_input = Input(shape=(num_numerical_features,), name='numerical_input')
num_x = Dense(256, activation='relu')(numerical_input)
num_x = Dropout(0.2)(num_x)
num_x = Dense(64, activation='relu')(num_x)

# Concatenate sequence and numerical data
combined = Concatenate()([x, num_x])

# Dense layers after concatenation
x = Dense(4096, activation="relu")(combined)
x = Dropout(0.1)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(256, activation="relu")(x)
outputs = Dense(num_classes, activation="softmax")(x)

# Build the model
model = Model(inputs=[sequence_input, numerical_input], outputs=outputs)

# Show model summary
print(model.summary())

# Optimizer and model compilation
learning_rate = 1e-4  # Set your desired learning rate
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Save the model after every epoch and only save the best model
checkpoint = ModelCheckpoint(
    args.save_path + '_model_epoch_{epoch:02d}.keras',   # Path where the model will be saved
    save_best_only=True,            # Only save a model if `val_loss` has improved
    monitor='val_loss',             # Monitor validation loss to decide whether to save
    verbose=1,                      # Logging level; 1 will print some log messages
    save_freq='epoch',              # 'epoch' means save after each epoch
    save_weights_only=False         # False means save the whole model
)

# Train the model
history = model.fit(
    [X_train_sequences, X_train_numerical],  # Provide both inputs as a list
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=([X_test_sequences, X_test_numerical], y_test),  # Validation data similarly needs dual inputs
    callbacks=[checkpoint]  # Add your checkpoint callback here
)

# Evaluate the model
test_loss, test_acc = model.evaluate(
    [X_test_sequences, X_test_numerical],  # Provide both inputs as a list
    y_test
)
print(f"Test Accuracy: {test_acc}")

model.save(args.save_path)  # Saves the model in the specified path

probabilities = model.predict([X_test_sequences, X_test_numerical])
# Get the top 5 predictions for each sample
top_5_predictions = np.argsort(probabilities, axis=1)[:, -5:]

# Assuming `y_test` is your true labels in one-hot encoded form
true_classes = np.argmax(y_test, axis=1)

# Check if the true class is in the top 5 predictions
top5_correct = [true_classes[i] in top_5_predictions[i] for i in range(len(true_classes))]

# Calculate the accuracy of having the true class in the top 5 predictions
top5_accuracy = np.mean(top5_correct)
print(f"Top-5 Accuracy: {top5_accuracy:.2f}")