import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_lightning_occurrences(event_id, start_time, end_time, window_size, xlim_start, xlim_end):
    """
    Plot lightning occurrences for a specific event and compute the average within a given window size.

    Parameters:
    - event_id: Event ID (string)
    - start_time: Start time for calculation (integer)
    - end_time: End time for calculation (integer)
    - window_size: Window size for averaging (integer)
    - xlim_start: Start time for plt.xlim() (integer)
    - xlim_end: End time for plt.xlim() (integer)
    """
    # Read HDF5 data file
    file_path = "data/train.h5"
    
    with h5py.File(file_path, 'r') as f:
        event_lght_data = pd.DataFrame(
            data=f[event_id]["lght"][:],
            columns=["t", "lat (deg)", "lon (deg)", "vil pixel x", "vil pixel y"]
        )
    
    # Filter data within the specified time range
    event_lght_data = event_lght_data[(event_lght_data["t"] >= start_time) & (event_lght_data["t"] <= end_time)]
    
    # Count lightning occurrences at each time step t
    lightning_counts = event_lght_data.groupby("t").size().reset_index(name="count")
    
    # Compute bin edges to ensure full coverage up to max t
    bin_edges = np.arange(start_time, end_time + window_size, window_size)
    
    # Assign time t values to corresponding time bins
    lightning_counts["time_bin"] = pd.cut(lightning_counts["t"], bins=bin_edges, include_lowest=True, labels=False)
    
    # Compute the mean lightning count within each window size
    mean_counts = lightning_counts.groupby("time_bin")["count"].mean().reset_index()
    
    # Generate stepwise constant lines for visualization
    bin_x = [bin_edges[i] for i in range(len(mean_counts)) for _ in range(2)]
    bin_y = [mean_counts["count"].iloc[i] for i in range(len(mean_counts)) for _ in range(2)]
    
    # Plotting
    plt.figure(figsize=(14, 5))
    
    # Plot original lightning occurrences as a line graph
    plt.plot(lightning_counts["t"], lightning_counts["count"], marker="o", markersize=2, linestyle="-", color="b", alpha=0.5, label="Original Lightning Count")
    
    # Plot stepwise average
    plt.step(bin_x, bin_y, where='post', color="r", linewidth=2, label=f"{window_size}-step Average")
    
    plt.xlabel("Time (t)")
    plt.ylabel("Lightning Count")
    plt.title(f"Lightning Occurrences Over Time with {window_size}-step Mean for Event {event_id}")
    plt.grid(True)
    plt.legend()
    
    # Set x-axis range
    plt.xlim(xlim_start, xlim_end)
    
    # Show plot
    plt.show()

