import numpy as np
import pandas as pd
import h5py
from skimage.transform import resize
import os
import tensorflow as tf

class EventDataLoader:
    def __init__(self, h5_file_path, events_csv_path):
        self.h5_file_path = h5_file_path
        self.events_csv_path = events_csv_path

    def get_event_ids(self, image_type="vil"):
        # Load CSV and extract event IDs for the specified image type
        events_df = pd.read_csv(self.events_csv_path)
        event_ids = events_df[events_df["img_type"] == image_type]["id"].dropna().unique().tolist()
        return event_ids

    def load_event_data(self, event_id, modality=None):
        # Load event data from the HDF5 file
        with h5py.File(self.h5_file_path, 'r') as f:
            if event_id not in f:
                raise KeyError(f"Event ID {event_id} not found in the HDF5 file!")

            if modality:
                # Load specific modality data
                if modality not in f[event_id]:
                    raise KeyError(f"Data for modality '{modality}' not found in event ID '{event_id}'!")
                data = f[event_id][modality][:]
            else:
                # Load the whole event data if no modality is specified
                data = f[event_id][:]

        return data  # Shape: (H, W, T) or (H, W, T, C)

    @staticmethod
    def normalize(data):
        # Normalize data to [0, 1]
        min_val, max_val = data.min(), data.max()
        return (data - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(data)

    def preprocess_data(self, event_ids, input_frames=12, target_frames=12, image_size=(128, 128), modalities=None):
        """
        Preprocess data for single or multiple modalities.

        Parameters:
        - event_ids (list): List of event IDs to process.
        - input_frames (int): Number of input frames.
        - target_frames (int): Number of target frames.
        - image_size (tuple): Size to resize spatial dimensions (H, W).
        - modalities (list or None): List of modalities to load. If None, only one modality is processed.

        Returns:
        - X (np.ndarray): Preprocessed input data of shape (samples, H, W, input_frames, C).
        - y (np.ndarray): Preprocessed target data of shape (samples, H, W, target_frames, C).
        """
        X, y = [], []

        for event_id in event_ids:
            if modalities:
                # Multi-modal processing
                modality_arrays = []
                for mod in modalities:
                    data = self.load_event_data(event_id, modality=mod)  # Load data for the modality
                    resized_data = np.stack([resize(data[:, :, t], image_size, preserve_range=True) for t in range(data.shape[2])], axis=-1)
                    modality_arrays.append(resized_data)

                # Combine modalities along a new axis
                data = np.stack(modality_arrays, axis=-1)  # Shape: (H, W, T, M)
            else:
                # Single modality processing
                data = self.load_event_data(event_id)  # Shape: (H, W, T)
                data = np.stack([resize(data[:, :, t], image_size, preserve_range=True) for t in range(data.shape[2])], axis=-1)

            # Normalize data
            data = self.normalize(data)

            # Construct input and target sequences
            T = data.shape[2]  # Total time frames
            C = data.shape[3] if modalities else 1  # Channel count (1 for single modality)
            for i in range(max(0, T - (input_frames + target_frames))):
                X.append(data[:, :, i:i+input_frames, :] if modalities else data[:, :, i:i+input_frames])
                y.append(data[:, :, i+input_frames:i+input_frames+target_frames, :] if modalities else data[:, :, i+input_frames:i+input_frames+target_frames])

        # Convert to NumPy arrays
        X = np.array(X)  # Shape: (num_samples, H, W, input_frames, C)
        y = np.array(y)  # Shape: (num_samples, H, W, target_frames, C)

        if not modalities:
            # Add channel dimension for single modality
            X = np.expand_dims(X, axis=-1)
            y = np.expand_dims(y, axis=-1)

        return X, y

    def split_train_test(self, X, y, split_ratio=0.8):
        # Split data into train and test sets based on time order
        split_idx = int(split_ratio * len(X))
        return (X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:])
    
    def preprocess_data_no_y(self, event_ids, input_frames=12, image_size=(128, 128), modality="vil"):

        X_dict = {}
        with h5py.File(self.h5_file_path, 'r') as f:
            for event_id in event_ids:
                if event_id in f and modality in f[event_id]:
                    data = f[event_id][modality][:]  # Shape: (384, 384, T)

                    # Resize and normalize
                    resized_data = np.stack([resize(data[:, :, t], image_size) for t in range(data.shape[2])], axis=-1)
                    normalized_data = self.normalize(resized_data)

                    # Expand dimensions to match input shape requirements
                    X_dict[event_id] = np.expand_dims(normalized_data, axis=-1)

        return X_dict

    def predict_and_save(self, model_path, event_ids, output_dir="predictions", input_frames=12, image_size=(128, 128)):
        """Load a model, preprocess data, make predictions, and save the results."""
        # Load the model
        model = tf.keras.models.load_model(model_path)

        # Preprocess data (no y needed)
        X_dict = self.preprocess_data_no_y(event_ids, input_frames=input_frames, image_size=image_size)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Predict and save results
        for event_id, X_input in X_dict.items():
            X_input = np.expand_dims(X_input, axis=0)  # Add batch dimension: (1, 128, 128, T, 1)

            # Make prediction
            y_pred = model.predict(X_input)
            y_pred = np.squeeze(y_pred, axis=(0, -1))

            # Resize prediction to original size (384, 384)
            y_pred_resized = np.stack([resize(y_pred[:, :, t], (384, 384)) for t in range(y_pred.shape[2])], axis=-1)

            # Save prediction as .npy file
            output_filename = f"prediction-{event_id}.npy"
            output_path = os.path.join(output_dir, output_filename)
            np.save(output_path, y_pred_resized.astype(np.float32))

        print("Predictions completed and saved!")
