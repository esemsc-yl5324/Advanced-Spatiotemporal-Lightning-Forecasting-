import matplotlib.pyplot as plt
import numpy as np

class PlotTimeSteppingPrediction:
    def __init__(self):
        pass

    def plot_loss(self, history):    
        # **plot loss**
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss', marker='o')
        plt.plot(history.history['val_loss'], label='Validation Loss', marker='s')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_input_target(X, y, sample_index=0):
        input_sample = X[sample_index]  # (128, 128, 12, 1)
        target_sample = y[sample_index]  # (128, 128, 12, 1)

        H, W, T, C = input_sample.shape

        fig, axes = plt.subplots(2, T, figsize=(20, 6))
        for t in range(T):
            axes[0, t].imshow(input_sample[:, :, t, 0], cmap='turbo')
            axes[0, t].axis('off')
            axes[0, t].set_title(f"Input Frame {t+1}")

            axes[1, t].imshow(target_sample[:, :, t, 0], cmap='turbo')
            axes[1, t].axis('off')
            axes[1, t].set_title(f"Target Frame {t+1}")

        axes[0, 0].set_ylabel("Input", fontsize=14)
        axes[1, 0].set_ylabel("Target", fontsize=14)

        plt.suptitle("Input vs Target Frames", fontsize=16)
        plt.tight_layout()
        plt.show()


    def visualize_sample_prediction(self, model, X, y, sample_index=None, colormap='turbo', figsize=(20, 6)):
        # Select a random sample if no index is provided
        if sample_index is None:
            sample_index = np.random.randint(0, X.shape[0])

        # Extract the input and ground truth sample
        input_sample = X[sample_index:sample_index + 1]  # Shape: (1, H, W, T, C)
        ground_truth = y[sample_index]  # Shape: (H, W, T, C)

        # Generate prediction
        predicted_output = model.predict(input_sample)  # Shape: (1, H, W, T, C)
        predicted_output = np.squeeze(predicted_output, axis=0)  # Remove batch dimension

        H, W, T = ground_truth.shape[:3]

        # Create subplots
        fig, axes = plt.subplots(3, T, figsize=figsize)

        for t in range(T):
            # Plot input frame
            axes[0, t].imshow(input_sample[0, :, :, t, -1], cmap=colormap)
            axes[0, t].axis('off')
            axes[0, t].set_title(f"Input Frame {t + 1}")

            # Plot ground truth frame
            axes[1, t].imshow(ground_truth[:, :, t, 0], cmap=colormap)
            axes[1, t].axis('off')
            axes[1, t].set_title(f"Ground Truth Frame {t + 1}")

            # Plot predicted frame
            axes[2, t].imshow(predicted_output[:, :, t, 0], cmap=colormap)
            axes[2, t].axis('off')
            axes[2, t].set_title(f"Predicted Frame {t + 1}")

        # Add row labels
        axes[0, 0].set_ylabel("Input", fontsize=14)
        axes[1, 0].set_ylabel("Ground Truth", fontsize=14)
        axes[2, 0].set_ylabel("Prediction", fontsize=14)

        plt.suptitle("Input vs Ground Truth vs Prediction", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_surprise_prediction(self, pred_vil, storm_id, colormap='turbo', figsize=(15, 10)):
        
        T = pred_vil.shape[2]  # Number of frames

        # Determine the grid layout (rows and columns)
        rows = (T + 3) // 4  # Arrange up to 4 frames per row
        fig, axes = plt.subplots(rows, 4, figsize=figsize)

        # Flatten axes for easier iteration
        axes = axes.flat if rows > 1 else [axes]

        for i, ax in enumerate(axes):
            if i < T:
                ax.imshow(pred_vil[:, :, i], cmap=colormap)
                ax.set_title(f"Frame {i + 1}")
            ax.axis("off")

        plt.suptitle(f"Predicted VIL Frames for Storm {storm_id}", fontsize=16)
        plt.tight_layout()
        plt.show()
