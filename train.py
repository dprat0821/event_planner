import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from joblib import dump

from config import MODEL_PATH, MODEL_ENCODER, MODEL_SCALER, MODEL_NAME

# Load data
data = pd.read_csv('updated_data.csv')

# Preprocess data
enc = OneHotEncoder()
scaler = StandardScaler()

# Assuming categorical features are 'office_location', 'event_month', 'prefixed_hotel_id'
categorical_data = enc.fit_transform(data[['office_location', 'event_month', 'prefixed_hotel_id']])
numerical_data = scaler.fit_transform(data[['group_size', 'days_duration', 'total_budget']])

# Save the encoders
dump(enc, MODEL_PATH + MODEL_ENCODER)
dump(scaler, MODEL_PATH + MODEL_SCALER)


X = np.hstack([categorical_data.toarray(), numerical_data])
y = data['nps_label']


# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_val, y_val))


model.save(MODEL_PATH + MODEL_NAME, save_format='tf')
