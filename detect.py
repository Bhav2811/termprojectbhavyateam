import pandas as pd
import numpy as np
import gzip
import requests
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Uncomment the following line if you have ann_visualizer installed
# from ann_visualizer.visualize import ann_viz    

from keras.models import Sequential
from keras.utils import plot_model
from keras.layers import Dense
from keras.optimizers import Adam

# Load the dataset
df = pd.read_csv('/kaggle/input/kdd-cup-1999-data/kddcup.data_10_percent.gz', header=None)
cols = pd.read_csv('/kaggle/input/kdd-cup-1999-data/kddcup.names', header=None)

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Load and process attack types
with open('/kaggle/input/kdd-cup-1999-data/training_attack_types', 'r') as f:
    attack_types = f.read()
    print("\nAttack Types:")
    print(attack_types)

# Create a dictionary of attack types
types = {'normal': 'normal'}
for line in attack_types.split("\n"):
    if line:
        attack, description = line.split(" ", 1)
        types[attack] = description

print("\nAttack Types Dictionary:")
print(types)

# Process column names
if cols.iloc[0, 0] == 'back':
    cols = cols.drop(cols.index[0])
    cols.reset_index(drop=True, inplace=True)

cols = cols.dropna(axis=1)
print("\nColumns after dropping NaN:")
print(cols.head())

# Split the first two columns by ':'
cols[[0, 1]] = cols[0].str.split(':', expand=True)
print("\nColumns after splitting:")
print(cols.head())

# Assign column names to the dataframe
names = cols[0].tolist()
names.append('label')
df.columns = names
print("\nDataframe with column names:")
print(df.head())

# Map labels to attack types
df['Attack Type'] = df['label'].apply(lambda x: types.get(x[:-1], 'unknown'))
print("\nDataframe with Attack Type:")
print(df.head())

# Display value counts and percentages for Attack Types
AT_count = df['Attack Type'].value_counts()
AT_per = AT_count / len(df) * 100
print("\nAttack Type Counts:")
print(AT_count)
print("\nAttack Type Percentages:")
print(AT_per)

# Display value counts and percentages for labels
lab_count = df['label'].value_counts()
lab_per = lab_count / len(df) * 100
print("\nLabel Counts:")
print(lab_count)
print("\nLabel Percentages:")
print(lab_per)

# Dataset information
print("\nDataset Information:")
print(f"Shape: {df.shape}")
print(f"Number of features: {len(df.columns)}")
print(f"Number of unique services: {df.service.nunique()}")
print(f"Number of labels: {df['label'].nunique()}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Identify categorical features
categorical = df.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical Features:")
print(categorical)

# Remove 'label' and 'Attack Type' from categorical features
categorical = [col for col in categorical if col not in ['label', 'Attack Type']]
print("\nExtracted Categorical Features:")
print(categorical)

# Visualize protocol_type distribution
fig, ax = plt.subplots(figsize=(7, 7))
sns.countplot(x='protocol_type', data=df, ax=ax, palette='Blues_d')
sns.set_style("darkgrid")

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom')

plt.title('Protocol Type Distribution')
plt.show()

# Protocol type percentages
print("\nProtocol Type Percentages:")
print(df['protocol_type'].value_counts(normalize=True) * 100)

# Visualize service distribution
fig, ax = plt.subplots(figsize=(17, 7))
sns.countplot(x='service', data=df, ax=ax, palette='Spectral', order=df['service'].value_counts().index, linewidth=0)
sns.set_style("dark")
plt.xticks(rotation=90)
plt.title('Service Distribution')
plt.show()

# Service percentages
print("\nService Percentages:")
print(df['service'].value_counts(normalize=True) * 100)

# Visualize flag distribution
fig, ax = plt.subplots(figsize=(10, 8))
sns.countplot(x='flag', data=df, ax=ax, palette='Blues_r', order=df['flag'].value_counts().index, linewidth=0)
plt.title('Flag Distribution')
plt.show()

# Flag percentages
print("\nFlag Percentages:")
print(df['flag'].value_counts(normalize=True) * 100)

# Calculate percentage of specific attack types
attack_sum_percentage = (
    df['Attack Type'].value_counts().sum() - 
    df['Attack Type'].value_counts().iloc[:3].sum()
) / df['Attack Type'].value_counts().sum() * 100
print(f"\nPercentage of all attack types except top 3: {attack_sum_percentage:.2f}%")

# Visualize Attack Type distribution
fig, ax = plt.subplots(figsize=(15, 5))
sns.countplot(x='Attack Type', data=df, ax=ax, palette='Greens_r', order=df['Attack Type'].value_counts().index, linewidth=0)
plt.title('Attack Type Distribution')
plt.show()

# Top 3 attack types
top_3_attacks = df['Attack Type'].value_counts().index[:3].tolist()
print(f"\nTop 3 Attack Types: {top_3_attacks}")

# Heatmap for missing values
fig, axis = plt.subplots(figsize=(12, 8))
sns.heatmap(df.isnull(), cmap='cool', cbar=False)
plt.title("Missing Values in the Dataset")
plt.xlabel("Features")
plt.ylabel("Rows")
plt.show()
print("We can see that there are no missing values in the dataset.")

# Remove columns with only one unique value
df = df[[col for col in df.columns if df[col].nunique() > 1]]
print(f"\nShape after removing constant columns: {df.shape}")

# Correlation matrix
corr = df.corr()
print("\nCorrelation Matrix:")
print(corr)

# Visualize correlation matrix
fig, ax = plt.subplots(figsize=(17, 15))
sns.heatmap(corr, cmap='coolwarm', ax=ax, linewidths=0.1)
plt.title("Correlation Between Features")
plt.show()

# Identify highly correlated pairs
high_corr = corr.abs() > 0.8
high_corr_pairs = high_corr.unstack().sort_values(ascending=False).drop_duplicates()
high_corr_pairs = high_corr_pairs[high_corr_pairs].index.tolist()
print("\nHighly Correlated Pairs (Correlation > 0.8):")
print(high_corr_pairs)

# Drop highly correlated columns to reduce multicollinearity
columns_to_drop = [
    'num_root', 'srv_rerror_rate', 'dst_host_srv_rerror_rate', 'dst_host_rerror_rate',
    'srv_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_serror_rate', 'dst_host_same_srv_rate'
]
df.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')
print(f"\nShape after dropping highly correlated columns: {df.shape}")

# Check data types
print("\nData Types:")
print(df.dtypes)

# Label encoding for categorical features
Le = LabelEncoder()
for col in categorical:
    df[col] = Le.fit_transform(df[col])

# Drop 'service' as it's encoded and might not be needed
df.drop(['service'], axis=1, inplace=True, errors='ignore')

print("\nDataframe after encoding and dropping 'service':")
print(df.head())

# Save the processed dataframe
df.to_csv('processed_kdd.csv', index=False)
print("\nProcessed data saved to 'processed_kdd.csv'.")

# Prepare features and target
X = df.drop(['label', 'Attack Type'], axis=1)
y = df['Attack Type']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Convert targets to numpy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

# One-hot encode the target variables
encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

# Function to plot training history
def plot_history(history, title):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Experiment with different learning rates
learning_rates = [0.1, 0.01, 0.001, 0.0001]
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
ax = ax.flatten()

for i, lr in enumerate(learning_rates):
    opt = Adam(learning_rate=lr)
    model = Sequential([
        Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='relu'),
        Dense(12, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)
    
    ax[i].plot(history.history['accuracy'], label='Train Accuracy')
    ax[i].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[i].set_title(f'Learning Rate: {lr}')
    ax[i].set_xlabel('Epoch')
    ax[i].set_ylabel('Accuracy')
    ax[i].legend()

plt.tight_layout()
plt.show()

# Experiment with different activation functions
activation_funcs = ['sigmoid', 'tanh', 'relu', 'elu']
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
ax = ax.flatten()

for i, act_fn in enumerate(activation_funcs):
    opt = Adam(learning_rate=0.001)
    model = Sequential([
        Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='relu'),
        Dense(12, activation=act_fn),
        Dense(y_train.shape[1], activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)
    
    ax[i].plot(history.history['accuracy'], label='Train Accuracy')
    ax[i].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[i].set_title(f'Activation Function: {act_fn}')
    ax[i].set_xlabel('Epoch')
    ax[i].set_ylabel('Accuracy')
    ax[i].legend()

plt.tight_layout()
plt.show()

# Build the final model
opt = Adam(learning_rate=0.001)
model = Sequential([
    Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='relu'),
    Dense(12, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train, 
    epochs=10, 
    validation_split=0.2, 
    batch_size=64, 
    verbose=1
)

# Plot the model architecture
plot_model(model, to_file='model_diagram.png', show_shapes=True, show_layer_names=True)
print("\nModel architecture saved to 'model_diagram.png'.")

# Uncomment the following line if you have ann_visualizer installed
# ann_viz(model, title="Neural Network Model", view=True, filename="model.gv")

# Plot training history
plot_history(history, 'Model Accuracy')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Scatter plot of loss vs accuracy
plt.figure(figsize=(8, 6))
plt.scatter(loss, accuracy, color='b')
plt.title('Loss vs Accuracy')
plt.xlabel('Loss')
plt.ylabel('Accuracy')
plt.show()

# Save the trained model
model.save('NN_model.h5')
print("\nTrained model saved to 'NN_model.h5'.")
