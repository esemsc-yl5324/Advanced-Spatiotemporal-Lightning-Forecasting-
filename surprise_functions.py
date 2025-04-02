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


def plot_flash_predictions_seconds(event_index):
    """
    Plots every 100th second for an event, showing VIS image with actual & predicted flash locations.

    Args:
        event_index (int): Index of the event in `test_dataset.event_ids`.
    """

    # **Retrieve the event ID using index**
    if event_index >= len(test_dataset.event_ids):
        raise IndexError(
            f"Event index {event_index} is out of range. Max: {len(test_dataset.event_ids)-1}"
        )

    event_id = test_dataset.event_ids[event_index]
    print(f"Plotting Event: {event_id}")

    # **Ensure event exists in predictions**
    if (
        event_id not in predicted_counts_per_event
        or event_id not in predicted_locations_per_event
    ):
        raise KeyError(f"Event ID {event_id} not found in predicted data.")

    # **Generate flash tensor for this event**
    flash_tensor = distribute_flashes_over_time(
        event_index
    )  # Shape: (10800, 3) -> (t, x, y)

    # **Filter every 100th second**
    flash_tensor = flash_tensor[flash_tensor[:, 0] % 100 == 0]

    # **Reopen HDF5 file dynamically to fetch VIS images**
    with h5py.File(test_dataset.h5_file_path, "r") as h5_file:
        event_data = h5_file[event_id]

        # **Set up figure for multiple seconds**
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # 2 rows, 5 columns

        for ax, (t, x, y) in zip(
            axes.flat, flash_tensor[:10]
        ):  # Select first 10 filtered entries
            t = int(t)  # Ensure time is integer
            frame_idx = t // 300  # Find which frame this second belongs to

            # **Extract VIS image for this frame**
            vis_image = event_data["vis"][:, :, frame_idx]

            # **Extract actual lightning locations for this second**
            actual_flashes = event_data["lght"]
            actual_flash_mask = actual_flashes[:, 0] == t
            actual_flash_coords = actual_flashes[actual_flash_mask][:, [3, 4]]

            # **Extract predicted flash locations**
            predicted_flash_mask = flash_tensor[:, 0] == t
            predicted_flash_coords = flash_tensor[predicted_flash_mask][:, [1, 2]]

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
            if len(predicted_flash_coords) > 0:
                ax.scatter(
                    predicted_flash_coords[:, 0],
                    predicted_flash_coords[:, 1],
                    color="blue",
                    marker="+",
                    s=50,
                    label="Predicted Flashes",
                )

            # **Plot settings**
            ax.set_title(f"Second {t}")
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
            f"Event {event_id}: Actual vs. Predicted Lightning Locations (Every 100th Second)",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def resize_to_384(image):
    """
    Rescales a 3D array so that the first two dimensions become 384x384
    while keeping the third dimension unchanged. This is done via
    `scipy.ndimage.zoom`.

    The function calculates zoom factors based on the image's current
    height/width and applies bilinear interpolation (order=1).

    Args:
        image (numpy.ndarray): A 3D array of shape (H, W, T) where
            H and W are spatial dimensions, and T is typically a time
            or channel dimension.

    Returns:
        numpy.ndarray: A 3D array resized to (384, 384, T).
    """
    zoom_factors = [384 / image.shape[0], 384 / image.shape[1], 1]
    return zoom(image, zoom_factors, order=1)


def normalize_event(
    event,
    global_vis_max=None,
    global_ir069_max=None,
    global_ir069_min=None,
    global_ir107_max=None,
    global_ir107_min=None,
):
    """
    Normalizes event data using min-max or max-based scaling.
    If global min/max are not provided, they are taken from the event itself.

    Args:
        event (dict): Contains 'vis', 'ir069', 'ir107', 'vil' arrays.
        global_vis_max (float): Max value for 'vis'. If None, use event's own max.
        global_ir069_max (float): Max value for 'ir069'. If None, use event's own max.
        global_ir069_min (float): Min value for 'ir069'. If None, use event's own min.
        global_ir107_max (float): Max value for 'ir107'. If None, use event's own max.
        global_ir107_min (float): Min value for 'ir107'. If None, use event's own min.

    Returns:
        dict: The same event dict with normalized arrays.

    """
    # Determine fallback values if needed
    if global_vis_max is None:
        global_vis_max = event["vis"].max()

    if global_ir069_max is None or global_ir069_min is None:
        global_ir069_max = event["ir069"].max()
        global_ir069_min = event["ir069"].min()

    if global_ir107_max is None or global_ir107_min is None:
        global_ir107_max = event["ir107"].max()
        global_ir107_min = event["ir107"].min()

    # Normalize
    event["vis"] = event["vis"] / global_vis_max * 0.85
    event["ir069"] = (event["ir069"] - global_ir069_min * 0.85) / (
        global_ir069_max * 0.85 - global_ir069_min * 0.85
    )
    event["ir107"] = (event["ir107"] - global_ir107_min * 0.85) / (
        global_ir107_max * 0.85 - global_ir107_min * 0.85
    )

    return event


def create_frame_based_dataset(event):
    """
    Extracts frames from the arrays (vis, ir069, ir107) at a stride of 5,
    returning X.

    Specifically, we pick every 5th time index from [0, total_frames).
    For each selected time step t, three channels (vis, ir069, ir107)
    are stacked into one frame for X.

    Args:
        event (dict):
            - 'vis', 'ir069', 'ir107': 3D arrays of shape (H, W, T)

    Returns:
        (numpy.ndarray, numpy.ndarray):
            X: shape (T', H, W, 3),
               where T' = total_frames // 5 (approximately)
    """
    X = []

    for t in range(36):
        x = np.stack(
            [event["vis"][:, :, t], event["ir069"][:, :, t], event["ir107"][:, :, t]],
            axis=-1,
        )

        X.append(x)

    return np.array(X)


def load_event_subset(event_ids, h5_path="/content/data/surprise_task2.h5"):
    """
    Loads selected events from the HDF5 file, applies resizing, normalization,
    and converts them into frame-based datasets.

    Args:
        event_ids (list): List of event IDs to load.
        h5_path (str): HDF5 file path with event data.

    Returns:
        (numpy.ndarray, numpy.ndarray):
            Concatenated X arrays across all chosen events.
    """
    X = []
    with h5py.File(h5_path, "r") as f:
        for event_id in event_ids:
            # Extract arrays
            event = {key: f[event_id][key][:] for key in ["vis", "ir069", "ir107"]}

            # Resize certain channels to 384×384
            event["ir069"] = resize_to_384(event["ir069"])
            event["ir107"] = resize_to_384(event["ir107"])

            # # Normalize all channels
            event = normalize_event(
                event,
                global_vis_max=global_vis_max,
                global_ir069_max=global_ir069_max,
                global_ir069_min=global_ir069_min,
                global_ir107_max=global_ir107_max,
                global_ir107_min=global_ir107_min,
            )

            # Convert to frame-based (X, Y)
            x_event = create_frame_based_dataset(event)
            X.append(x_event)

    # Concatenate across all events
    return np.concatenate(X, axis=0)


def make_gif(outfile, files, fps=10, loop=0):
    """
    Saves a sequence of image files as a GIF.
    files: list of filepaths
    fps: frames per second
    loop: how many times to loop the gif (0=forever)
    """
    imgs = [PIL.Image.open(f) for f in files]
    imgs[0].save(
        fp=outfile,
        format="gif",
        append_images=imgs[1:],
        save_all=True,
        duration=int(1000 / fps),
        loop=loop,
    )
    im = IPython.display.Image(filename=outfile)
    im.reload()
    return im


def plot_predicted_vil(eid, vil_3d, output_gif=False, save_gif=False):
    """
    Visualize predicted vil for a single event with 36 frames (shape=384x384x36).
    If output_gif=True, generate a GIF by saving each frame as .png
    and then combining.
    """
    # vil_3d shape: (384,384,36) float32/float64
    H, W, T = vil_3d.shape
    assert T == 36, "Expected 36 frames for each event"

    def plot_frame(ti):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(vil_3d[:, :, ti], vmin=0, vmax=255, cmap="turbo")  # or adapt colormap
        ax.set_title(f"Predicted VIL\nEvent: {eid}, Frame: {ti}")
        ax.axis("off")
        if output_gif:
            file = f"_temp_{eid}_{ti}.png"
            fig.savefig(
                file, bbox_inches="tight", dpi=100, pad_inches=0.02, facecolor="white"
            )
            plt.close()
        else:
            plt.show()

    if output_gif:
        # Generate all frames
        for ti in range(T):
            plot_frame(ti)

        # Combine frames into GIF
        gif_file = f"{eid}_prediction.gif"
        frames_list = [f"_temp_{eid}_{ti}.png" for ti in range(T)]
        im = make_gif(gif_file, frames_list, fps=5)
        # Cleanup temp frames
        for f in frames_list:
            if os.path.exists(f):
                os.remove(f)

        # Display the GIF in notebook
        IPython.display.display(im)

        # Optionally keep the gif_file or remove if save_gif=False
        if not save_gif and os.path.exists(gif_file):
            os.remove(gif_file)

    else:
        # Just show a few frames, e.g. 0, 17, 34
        plot_frame(0)
        plot_frame(17)
        plot_frame(34)
