import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    Layer,
    LayerNormalization,
    MultiHeadAttention
)
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from imblearn.over_sampling import RandomOverSampler

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
data = pd.read_csv("/kaggle/input/cleaned-protein-classification/cleaned_protein_classfication.csv")

# Initialize and fit tokenizer
unique_chars = "ACDEFGHIKLMNPQRSTVWY"
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(list(unique_chars))

# Convert sequences
sequences = tokenizer.texts_to_sequences(data["sequence"].apply(str))
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')

# Label binarization
lb = LabelBinarizer()
lb.fit(data["classification"].unique())
labels = lb.transform(data["classification"])

# Load and apply scaler
scaler = joblib.load('/kaggle/input/scalar/other/default/1/scaler (1).joblib')
numerical_columns = ['residueCount', 'structureMolecularWeight', 'crystallizationTempK', 'densityMatthews', 'densityPercentSol', 'phValue', 'Isoelectric_Point', 'Aromaticity', 'Instability_Index', 'Helix', 'Turn', 'Sheet']
numerical_data = data[numerical_columns]
numerical_data_scaled = scaler.transform(numerical_data)

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

# Load the model (make sure the directory and format are correct)
model = load_model(
    '/kaggle/input/mixed_8/keras/default/1/model_epoch_03.keras',
    custom_objects={
        'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
        'TransformerBlock': TransformerBlock
    }
)

test_index = 0  # Index of the test example to display
test_sequence = X_test_sequences[test_index].reshape(1, -1)
test_numerical = X_test_numerical[test_index].reshape(1, -1)
actual_label = lb.inverse_transform(y_test[test_index].reshape(1, -1))[0]

# Prediction
prediction = model.predict([test_sequence, test_numerical])
predicted_label = lb.inverse_transform(prediction)[0]

# Display the input features and model output
print("Test Sequence (Tokenized and Padded):", test_sequence)
print("Test Numerical Features (Scaled):", test_numerical)
print("Actual Label:", actual_label)
print("Predicted Label:", predicted_label)

# User inputs their amino acid sequence
my_amino_acid_sequence = input("Enter your amino acid sequence: ")
#AITGIFFGSDTGNTENIAKMIQKQLGKDVADVHDIAKSSKEDLEAYDILLLGIPTWYYGEAQCDWDDFFPTLEEIDFNGKLVALFGCGDQEDYAEYFCDALGTIRDIIEPRGATIVGHWPTAGYHFEASKGLADDDHFVGLAIDEDRQPELTAERVEKWVKQISEELHLDEILNA

# User inputs numerical features
input_numerical_features = input("Enter numerical features separated by commas (residueCount, structureMolecularWeight, crystallizationTempK, densityMatthews, densityPercentSol, phValue, Isoelectric_Point, Aromaticity, Instability_Index, Helix, Turn, Sheet): ")
#350.0,40901.7,295.0,2.2,44.1,7.0,4.213099479675290,0.10857142857142900,30.04571428571430,0.33142857142857100,0.28,0.37142857142857100

# Split the input string into a list and convert to float
numerical_features_list = [float(num) for num in input_numerical_features.split(',')]

# Tokenize and pad the sequence
user_sequence = tokenizer.texts_to_sequences([my_amino_acid_sequence])
user_padded_sequence = pad_sequences(user_sequence, maxlen=max_length, padding='pre')

# Scale numerical features
user_numerical_features = np.array([numerical_features_list])  # Convert list to numpy array for scaling
user_numerical_features_scaled = scaler.transform(user_numerical_features)

# Prediction
user_prediction = model.predict([user_padded_sequence, user_numerical_features_scaled])
user_predicted_label = lb.inverse_transform(user_prediction)[0]

print("Predicted Classification for the Provided Sequence and Numerical Features:", user_predicted_label)
#Electron Transport