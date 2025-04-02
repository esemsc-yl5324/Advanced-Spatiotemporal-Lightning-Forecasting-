def get_event_ids(num):
    """
    Randomly selects `num` unique event IDs from 'events.csv' and splits them
    into train/test sets (80%/20%). If `num` exceeds the total, it is capped
    at the maximum available.

    Args:
        num (int): Number of event IDs to sample.

    Returns:
        (list, list): Two lists of event IDs, the first (80%) for the main set
        and the second (20%) for testing.
    """
    df = pd.read_csv("data/events.csv", parse_dates=["start_utc"])
    unique_ids = df["id"].unique()

    # If 'num' exceeds the total number of available events, limit it
    if num > len(unique_ids):
        num = len(unique_ids)

    # Randomly select 'num' events (no repetition)
    sampled_ids = random.sample(list(unique_ids), num)

    # Split 20% of them into a test set, and the remaining 80% into event_ids
    test_ids = sampled_ids[:int(num*0.2)]
    event_ids = sampled_ids[int(num*0.2):]
    return event_ids, test_ids


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

def normalize_event(event, global_vis_max = None,
                    global_ir069_max = None, global_ir069_min = None,
                    global_ir107_max = None, global_ir107_min = None):

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
        global_vis_max = event['vis'].max()

    if global_ir069_max is None or global_ir069_min is None:
        global_ir069_max = event['ir069'].max()
        global_ir069_min = event['ir069'].min()

    if global_ir107_max is None or global_ir107_min is None:
        global_ir107_max = event['ir107'].max()
        global_ir107_min = event['ir107'].min()

    # Normalize
    event['vis'] = event['vis'] / global_vis_max * 0.85
    event['ir069'] = (event['ir069'] - global_ir069_min * 0.85) / (global_ir069_max * 0.85 - global_ir069_min * 0.85)
    event['ir107'] = (event['ir107'] - global_ir107_min * 0.85) / (global_ir107_max * 0.85 - global_ir107_min * 0.85)
    event['vil'] = event['vil'] / 255.0

    return event

def create_frame_based_dataset(event):
    """
    Extracts frames from the 3D arrays (vis, ir069, ir107, vil) at a stride of 5,
    returning frame-wise (X, Y) pairs.

    Specifically, we pick every 5th time index from [0, total_frames).
    For each selected time step t, three channels (vis, ir069, ir107)
    are stacked into one frame for X, while vil forms the corresponding Y.

    Args:
        event (dict):
            - 'vis', 'ir069', 'ir107': 3D arrays of shape (H, W, T)
            - 'vil': 3D array of shape (H, W, T)

    Returns:
        (numpy.ndarray, numpy.ndarray):
            X: shape (T', H, W, 3),
               where T' = total_frames // 5 (approximately)
            Y: shape (T', H, W),
               matching X in the time dimension
    """
    X, Y = [], []
    total_frames = event['vil'].shape[2]

    # Take every 5th frame
    for t in range(0, total_frames, 5):
        x = np.stack([
            event['vis'][:, :, t],
            event['ir069'][:, :, t],
            event['ir107'][:, :, t]
        ], axis=-1)
        y = event['vil'][:, :, t]
        X.append(x)
        Y.append(y)

    return np.array(X), np.array(Y)

def load_event_subset(event_ids, h5_path="data/train.h5"):
    """
    Loads selected events from the HDF5 file, applies resizing, normalization,
    and converts them into frame-based (X, Y) datasets.

    Args:
        event_ids (list): List of event IDs to load.
        h5_path (str): HDF5 file path with event data.

    Returns:
        (numpy.ndarray, numpy.ndarray):
            Concatenated X and Y arrays across all chosen events.
    """
    X, Y = [], []
    with h5py.File(h5_path, 'r') as f:
        for event_id in event_ids:
            # Extract arrays
            event = {key: f[event_id][key][:] for key in ['vis', 'ir069', 'ir107', 'vil']}

            # Resize certain channels to 384×384
            event['ir069'] = resize_to_384(event['ir069'])
            event['ir107'] = resize_to_384(event['ir107'])

            # # Normalize all channels
            event = normalize_event(event, global_vis_max = global_vis_max,
                    global_ir069_max = global_ir069_max, global_ir069_min = global_ir069_min,
                    global_ir107_max = global_ir107_max, global_ir107_min = global_ir107_min)

            # Convert to frame-based (X, Y)
            x_event, y_event = create_frame_based_dataset(event)
            X.append(x_event)
            Y.append(y_event)

    # Concatenate across all events
    return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)

def visualize_sample(X, Y, idx):

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(X[idx, :, :, 0], cmap='gray')  # VIS
    plt.title("VIS")
    plt.subplot(1, 4, 2)
    plt.imshow(X[idx, :, :, 1], cmap='viridis')  # IR069
    plt.title("IR069")
    plt.subplot(1, 4, 3)
    plt.imshow(X[idx, :, :, 2], cmap='inferno')  # IR107
    plt.title("IR107")
    plt.subplot(1, 4, 4)
    plt.imshow(Y[idx], cmap='turbo')  # VIL
    plt.title("VIL")
    plt.show()

def create_dataloaders(
    X_train, Y_train,
    X_val, Y_val,
    train_batch_size=16,
    val_batch_size=16,
    shuffle=True
):
    """
    Converts (X_train, Y_train, X_val, Y_val) into PyTorch tensors,
    rearranges dimensions, and packages them into DataLoaders for
    training and validation.

    Args:
        X_train, Y_train, X_val, Y_val (numpy.ndarray):
            - X_* shape: (N, H, W, C), where N is the number of samples
            - Y_* shape: (N, H, W)
        train_batch_size (int): Batch size for the training DataLoader.
        val_batch_size (int): Batch size for the validation DataLoader.
        shuffle (bool): Whether to shuffle the training set.

    Returns:
        (DataLoader, DataLoader): The training and validation DataLoaders.

    """

    # 1) Convert to PyTorch tensors and adjust dimensions
    #    X: (batch, channel, height, width) => permute(0, 3, 1, 2)
    #    Y: (batch, 1, height, width) => unsqueeze(1)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32).permute(0, 3, 1, 2)
    Y_val_t   = torch.tensor(Y_val,   dtype=torch.float32).unsqueeze(1)

    # 2) Create TensorDataset objects
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    val_dataset   = TensorDataset(X_val_t,   Y_val_t)

    # 3) Wrap them with DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=shuffle
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False
    )

    return train_loader, val_loader

# Utility functions for "denormalization"
def denormalize_vis(vis, global_vis_max):
    """
    Converts VIS data back to its original scale by multiplying
    with the stored global maximum.
    """
    return vis * global_vis_max

def denormalize_vil(vil, global_vil_max):
    """
    Converts VIL data back to its original 0–255 range by multiplying
    with the specified global maximum.
    """
    return vil * global_vil_max

# Helper to pick valid sample indices (random selection)
def get_test_sample_idx(data, max_samples=3):
    """
    Randomly selects up to 'max_samples' indices from 'data'
    without repetition.
    """
    return np.random.choice(len(data), max_samples, replace=False)
