import os
import random
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import IPython
from skimage.transform import resize
from scipy.optimize import linear_sum_assignment


class LightningDataset(torch.utils.data.Dataset):
    def __init__(self, h5_file, event_ids, upscale_shape=(384, 384)):
        """
        Dataset that extracts input frames, normalizes them, and handles lazy loading.
        """
        self.h5_file_path = h5_file.filename  # Store file path for dynamic access
        self.upscale_shape = upscale_shape
        self.event_ids = event_ids
        self.num_original_frames = 36  # 36 frames per event

        # Normalization ranges (min-max scaling)
        self.norm_ranges = {
            "ir069": (-7209, -1855),
            "ir107": (-6760, 2714),
            "vil": (0, 254),
            "vis": (234, 12529),
        }

        # **Precompute lightning counts per frame**
        self.labels = {}
        with h5py.File(self.h5_file_path, "r") as h5_file:  # Open dynamically
            for event_id in self.event_ids:
                event = h5_file[event_id]  # Read from HDF5 file
                strikes_per_frame = np.zeros(self.num_original_frames)

                # Sum up lightning strikes in each 5-min (300s) interval
                for strike_time in event["lght"][:, 0]:
                    frame_idx = int(strike_time // 300)
                    if 0 <= frame_idx < self.num_original_frames:
                        strikes_per_frame[frame_idx] += 1

                self.labels[event_id] = torch.tensor(
                    strikes_per_frame, dtype=torch.float32
                )

    def normalize(self, image, min_val, max_val):
        """Min-Max normalize the image to range [0, 1]."""
        return (image - min_val) / (max_val - min_val)

    def __len__(self):
        return self.num_original_frames * len(self.event_ids)

    def __getitem__(self, idx):
        """
        Get a single frame and its lightning strike count.
        """
        # **Reopen HDF5 file dynamically**
        with h5py.File(self.h5_file_path, "r") as h5_file:
            event_idx = idx // self.num_original_frames  # Which event?
            frame_idx = idx % self.num_original_frames  # Which 5-minute frame?
            event_id = self.event_ids[event_idx]
            event = h5_file[event_id]  # Read event data

            # **Extract & Normalize Channels**
            vis = self.normalize(
                event["vis"][:, :, frame_idx], *self.norm_ranges["vis"]
            )
            ir069 = self.normalize(
                event["ir069"][:, :, frame_idx], *self.norm_ranges["ir069"]
            )
            ir107 = self.normalize(
                event["ir107"][:, :, frame_idx], *self.norm_ranges["ir107"]
            )
            vil = self.normalize(
                event["vil"][:, :, frame_idx], *self.norm_ranges["vil"]
            )

            # **Upscale IR images**
            ir069 = torch.tensor(ir069, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            ir107 = torch.tensor(ir107, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            ir069 = F.interpolate(
                ir069, size=(384, 384), mode="bilinear", align_corners=False
            ).squeeze()
            ir107 = F.interpolate(
                ir107, size=(384, 384), mode="bilinear", align_corners=False
            ).squeeze()

            # **Convert to Tensor & Stack**
            vis = torch.tensor(vis, dtype=torch.float32).unsqueeze(0)
            ir069 = ir069.unsqueeze(0)
            ir107 = ir107.unsqueeze(0)
            vil = torch.tensor(vil, dtype=torch.float32).unsqueeze(0)
            inputs = torch.cat([vil, ir069, ir107, vis], dim=0)

            # **Get Target Lightning Strike Count**
            target = self.labels[event_id][frame_idx]

            return inputs, target


class LightningCNN(nn.Module):
    def __init__(self):
        super(LightningCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (16, 192, 192)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32, 96, 96)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (64, 48, 48)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (128, 24, 24)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flatten to (128 * 24 * 24)
            nn.Linear(128 * 24 * 24, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Final output layer (single lightning count)
            nn.Softplus(),  # ReLU replaced with Softplus to allow smooth non-negative outputs
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x.squeeze()  # Ensure a scalar output


def evaluate_model_on_event(model, dataset, event_id, device):
    """
    Evaluate model on a single event and plot actual vs. predicted lightning strikes.

    Args:
        model: Trained LightningCNN model.
        dataset: LightningDataset instance.
        event_id: ID of the event to evaluate.
        device: "cuda" or "cpu".
    """
    model.eval()  # Set model to evaluation mode

    # Find all indices corresponding to the selected event
    event_indices = [
        i
        for i in range(len(dataset))
        if dataset.event_ids[i // dataset.num_original_frames] == event_id
    ]

    # Prepare lists to store values
    actual_lightnings = []
    predicted_lightnings = []

    with torch.no_grad():  # No gradient calculation needed for evaluation
        for i in event_indices:
            inputs, actual_count = dataset[i]  # Get input features and actual count
            inputs = inputs.to(device).unsqueeze(0)  # Add batch dimension

            # Get model prediction
            predicted_count = model(inputs).cpu().item()

            # Store values
            actual_lightnings.append(actual_count.item())
            predicted_lightnings.append(predicted_count)

    # Accumulate lightning strikes over time
    actual_lightnings = np.cumsum(actual_lightnings)
    predicted_lightnings = np.cumsum(predicted_lightnings)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, 37),
        actual_lightnings,
        label="Actual Lightning Strikes",
        marker="o",
        linestyle="-",
    )
    plt.plot(
        range(1, 37),
        predicted_lightnings,
        label="Predicted Lightning Strikes",
        marker="x",
        linestyle="--",
    )

    plt.xlabel("Frame (5-min interval)")
    plt.ylabel("Cumulative Lightning Strikes")
    plt.title(f"Event {event_id}: Actual vs. Predicted Lightning Strikes")
    plt.legend()
    plt.grid(True)
    plt.show()


def predict_flash_locations(vis_image, vil_image, predicted_flashes):
    """
    Predicts lightning strike locations based on a combination of VIS and VIL images.

    Args:
        vis_image (numpy.ndarray): VIS satellite image of shape (H, W).
        vil_image (numpy.ndarray): VIL satellite image of shape (H, W).
        predicted_flashes (int): Number of flashes predicted by the model for the frame.

    Returns:
        List of tuples [(x1, y1), (x2, y2), ...] of predicted flash locations.
    """

    # Flatten both images
    vis_flat = vis_image.flatten().astype(np.float32)
    vil_flat = vil_image.flatten().astype(np.float32)

    # Shift each to positive values
    vis_flat -= vis_flat.min()
    vil_flat -= vil_flat.min()

    # Combine 50:50
    combined_flat = 0.5 * vis_flat + 0.5 * vil_flat

    # Avoid division by zero if combined is all zeros
    total_sum = combined_flat.sum()
    if total_sum == 0:
        # In the unlikely event there's no signal at all, return empty
        # or you could return random locations, etc.
        return []

    # Normalize to create a probability distribution
    combined_flat /= total_sum

    # Get indices of the highest probability flash locations
    # np.argsort() returns ascending order, so we take the last `predicted_flashes`.
    flash_indices = np.argsort(combined_flat)[-predicted_flashes:]

    # Convert flattened indices back to (x, y) coordinates
    width = vis_image.shape[1]
    flash_coords = [(idx % width, idx // width) for idx in flash_indices]

    return flash_coords


def plot_flash_predictions(event_id, frame_indices, test_dataset):
    """
    Plots multiple frames for a given event, showing VIS image with actual & predicted flash locations.

    Args:
        event_id (str): Event ID to plot.
        frame_indices (list): List of frame indices to plot.
    """

    # **Reopen HDF5 file dynamically to fetch event data**
    with h5py.File(test_dataset.h5_file_path, "r") as h5_file:
        event_data = h5_file[event_id]

        # **Set up figure for multiple frames**
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # 2 rows, 5 columns

        for ax, frame_idx in zip(axes.flat, frame_indices):
            # **Extract VIS image for this frame**
            vis_image = event_data["vis"][:, :, frame_idx]

            # **Extract actual lightning locations from the dataset**
            actual_flashes = event_data["lght"]  # Shape (num_flashes, 5)

            # **Filter flashes that occurred in this frame (5-min window)**
            time_start = frame_idx * 300  # Frame start time in seconds
            time_end = (frame_idx + 1) * 300  # Frame end time
            actual_flash_mask = (actual_flashes[:, 0] >= time_start) & (
                actual_flashes[:, 0] < time_end
            )
            actual_flash_coords = actual_flashes[actual_flash_mask][
                :, [3, 4]
            ]  # Extract x (col 4) & y (col 5)

            # **Extract predicted flash locations from probability distribution**
            predicted_locations = predicted_locations_per_event[event_id][frame_idx]

            # **Plot VIS image as background**
            ax.imshow(vis_image, cmap="gray", vmin=0, vmax=10000)

            # **Overlay actual lightning locations**
            if len(actual_flash_coords) > 0:
                ax.scatter(
                    actual_flash_coords[:, 0],
                    actual_flash_coords[:, 1],
                    color="red",
                    marker="x",
                    s=50,
                    label="Actual Flashes",
                )

            # **Overlay predicted lightning locations**
            if len(predicted_locations) > 0:
                predicted_x, predicted_y = zip(*predicted_locations)
                ax.scatter(
                    predicted_x,
                    predicted_y,
                    color="blue",
                    marker="+",
                    s=50,
                    label="Predicted Flashes",
                )

            # **Plot settings**
            ax.set_title(f"Frame {frame_idx}")
            ax.set_xticks([])
            ax.set_yticks([])

        # **Legend & Layout**
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="x",
                color="r",
                linestyle="None",
                markersize=8,
                label="Actual Flashes",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="+",
                color="b",
                linestyle="None",
                markersize=8,
                label="Predicted Flashes",
            ),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=2)
        plt.suptitle(
            f"Event {event_id}: Actual vs. Predicted Lightning Locations", fontsize=16
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def distribute_flashes_randomly_over_time(event_index):
    """
    Distributes predicted flashes over 300 seconds per frame for an event specified by index.
    Ensures the output is sorted by time (`t`).

    Args:
        event_index (int): The index of the event in `test_dataset.event_ids`.

    Returns:
        torch.Tensor: A tensor of shape (10800, 3) with (t, x, y) for predicted flashes, sorted by `t`.
    """

    # **Retrieve the event ID using the index**
    if event_index >= len(test_dataset.event_ids):
        raise IndexError(
            f"Event index {event_index} is out of range. Max: {len(test_dataset.event_ids)-1}"
        )

    event_id = test_dataset.event_ids[event_index]
    print(f"Processing Event: {event_id}")

    # **Ensure event exists in predictions**
    if (
        event_id not in predicted_counts_per_event
        or event_id not in predicted_locations_per_event
    ):
        raise KeyError(f"Event ID {event_id} not found in predicted data.")

    num_frames = 36  # Each event has 36 frames
    seconds_per_frame = 300  # 5 minutes per frame
    total_seconds = num_frames * seconds_per_frame  # 10800 seconds

    flash_data = []  # Store (t, x, y) tuples

    for frame_idx in range(num_frames):
        start_time = frame_idx * seconds_per_frame  # Start second for the frame
        end_time = start_time + seconds_per_frame  # End second

        # **Retrieve predicted flashes and locations safely**
        predicted_flashes = (
            predicted_counts_per_event[event_id][frame_idx]
            if frame_idx < len(predicted_counts_per_event[event_id])
            else 0
        )
        predicted_locations = (
            predicted_locations_per_event[event_id][frame_idx]
            if frame_idx < len(predicted_locations_per_event[event_id])
            else []
        )

        if predicted_flashes == 0 or not predicted_locations:
            continue  # Skip empty frames

        # **Randomly assign times for flashes within the 300s window**
        flash_times = np.random.randint(start_time, end_time, size=predicted_flashes)
        flash_coords = np.array(
            predicted_locations[:predicted_flashes]
        )  # Get top `predicted_flashes` locations

        # **Ensure enough locations exist**
        if flash_coords.shape[0] < predicted_flashes:
            extra_needed = predicted_flashes - flash_coords.shape[0]
            extra_coords = np.random.choice(
                flash_coords, size=extra_needed, replace=True
            )
            flash_coords = np.vstack((flash_coords, extra_coords))

        # **Stack (t, x, y)**
        flash_data.extend(np.column_stack((flash_times, flash_coords)))

    # **Convert to Tensor**
    flash_tensor = torch.tensor(flash_data, dtype=torch.float32)

    # **Sort the tensor by `t` (column 0)**
    flash_tensor = flash_tensor[flash_tensor[:, 0].argsort()]

    return flash_tensor


def distribute_flashes_evenly_over_time(event_index):
    """
    Distributes predicted flashes evenly across 300 seconds per frame.
    Flash locations are randomly assigned from available predictions.

    Args:
        event_index (int): The index of the event in `test_dataset.event_ids`.

    Returns:
        torch.Tensor: A tensor of shape (10800, 3) with (t, x, y) for predicted flashes, sorted by `t`.
    """

    # **Retrieve the event ID using the index**
    if event_index >= len(test_dataset.event_ids):
        raise IndexError(
            f"Event index {event_index} is out of range. Max: {len(test_dataset.event_ids)-1}"
        )

    event_id = test_dataset.event_ids[event_index]
    print(f"Processing Event: {event_id}")

    # **Ensure event exists in predictions**
    if (
        event_id not in predicted_counts_per_event
        or event_id not in predicted_locations_per_event
    ):
        raise KeyError(f"Event ID {event_id} not found in predicted data.")

    num_frames = 36  # Each event has 36 frames
    seconds_per_frame = 300  # 5 minutes per frame
    total_seconds = num_frames * seconds_per_frame  # 10800 seconds

    flash_data = []  # Store (t, x, y) tuples

    for frame_idx in range(num_frames):
        start_time = frame_idx * seconds_per_frame  # Start second for the frame
        end_time = start_time + seconds_per_frame  # End second

        # **Retrieve predicted flashes and locations safely**
        predicted_flashes = (
            predicted_counts_per_event[event_id][frame_idx]
            if frame_idx < len(predicted_counts_per_event[event_id])
            else 0
        )
        predicted_locations = (
            predicted_locations_per_event[event_id][frame_idx]
            if frame_idx < len(predicted_locations_per_event[event_id])
            else []
        )

        if predicted_flashes == 0 or not predicted_locations:
            continue  # Skip empty frames

        # **Evenly distribute flash times**
        if predicted_flashes > 1:
            flash_times = np.linspace(
                start_time, end_time - 1, num=predicted_flashes, dtype=int
            )
        else:
            flash_times = np.array(
                [start_time + seconds_per_frame // 2]
            )  # Place in middle if only 1 flash

        # **Randomly assign locations from predicted list**
        flash_coords = np.array(predicted_locations)

        # **Ensure we have enough locations (if fewer locations than flashes, duplicate some)**
        if flash_coords.shape[0] < predicted_flashes:
            extra_needed = predicted_flashes - flash_coords.shape[0]
            extra_coords = np.random.choice(
                flash_coords, size=extra_needed, replace=True
            )
            flash_coords = np.vstack((flash_coords, extra_coords))

        # **Shuffle locations to prevent structured patterns**
        np.random.shuffle(flash_coords)

        # **Stack (t, x, y)**
        flash_data.extend(
            np.column_stack((flash_times, flash_coords[:predicted_flashes]))
        )

    # **Convert to Tensor**
    flash_tensor = torch.tensor(flash_data, dtype=torch.float32)

    # **Sort the tensor by `t` (column 0)**
    flash_tensor = flash_tensor[flash_tensor[:, 0].argsort()]

    return flash_tensor
