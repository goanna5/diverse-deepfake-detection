"""Load video metadata and extract evenly spaced frames from each video.
"""

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Number of frames extracted from each video
FRAMES_PER_VID = 5


def get_df(path_prefix, subset) -> pd.DataFrame:
    """Create a DataFrame from the metadata CSV and preprocess it.

    Combines the path and filename columns into a single video filepath,
    drops unnecessary columns, removes mismatched video/audio entries and
    converts the category codes into ``real``/``fake`` labels.

    Args:
        path_prefix: Prefix prepended to each video's relative path.
        subset: If True, read ``meta_data_subset.csv``; otherwise read
            ``meta_data.csv``.

    Returns:
        A DataFrame with a unique ``id`` column first, a ``vid_filepath``
        column, and a ``label`` column containing ``real`` or ``fake``.
    """
    if subset:
        df = pd.read_csv("meta_data_subset.csv")
    else:
        df = pd.read_csv("meta_data.csv")

    # Combine filename and path columns
    df["vid_filepath"] = path_prefix + df["path"].str.cat(
        df["filename"], sep="/"
    )

    # Drop unnecessary columns
    df = df.drop(
        columns=["target1", "target2", "method", "type", "path", "filename"]
    )

    # Remove any rows with fake video-real audio, and real video-fake audio
    df = df[df.category != "B"]
    df = df[df.category != "C"]

    # Replace category A and D with real and fake
    df["category"] = df["category"].replace('A', 'real')
    df["category"] = df["category"].replace('D', 'fake')

    # Rename column
    df.rename(columns={"category": "label"}, inplace=True)

    # Add unique id column at start
    df.insert(0, "id", range(len(df)))

    return df


def extract_img_frames(video_path: str) -> list[np.ndarray]:
    """Return evenly spaced BGR frames from a video file.

    Reference: Claude was used to help write this function, along with
    https://www.geeksforgeeks.org/python/python-program-extract-frames-using-opencv/

    Args:
        video_path: Full (not relative) path of the single video to process.

    Returns:
        A list of up to ``FRAMES_PER_VID`` frames, each a NumPy array of
        shape ``(height, width, 3)`` with dtype ``uint8``. Returns an empty
        list if the video cannot be opened or has no frames. Fewer frames
        are returned if any frame fails to be read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    # Spread indices evenly across the video duration
    indices = [
        int(total_frames * i / FRAMES_PER_VID) for i in range(FRAMES_PER_VID)
    ]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frames.append(frame)

    cap.release()
    return frames


def extract_all_img_frames(df: pd.DataFrame, path_prefix: str,
                           save_imgs: bool = False) -> pd.DataFrame:
    """Extract ``FRAMES_PER_VID`` frames from every video in the dataset.

    Reference: Claude was used to help iterate through the dataset;
    ``tqdm`` provides the progress bar.

    Args:
        df: Dataset metadata. Each row must have ``id``, ``vid_filepath``,
            ``label`` (real or fake), ``race``, ``gender`` and ``source``.
        path_prefix: Prefix for the directory in which frames are saved
            (under ``frames/``).
        save_imgs: If True, save each extracted frame as a JPEG file.

    Returns:
        A DataFrame with one row per extracted frame, containing the video
        metadata plus ``frame_filepath`` and ``frame_idx`` columns. Videos
        that do not yield exactly ``FRAMES_PER_VID`` frames are skipped.
    """
    frame_records = []

    # Loop through each data row in the dataset df
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting frames"):
        vid_filepath = row["vid_filepath"]
        id = row["id"]
        frames = extract_img_frames(vid_filepath)

        # Skip if extraction failed (didn't extract the expected frames)
        if len(frames) != FRAMES_PER_VID:
            continue

        # Loop through each frame in the video
        for i, frame_matrix in enumerate(frames):
            frame_filepath = f"{path_prefix}frames/{id}_{i}.jpg"

            if save_imgs:
                cv2.imwrite(frame_filepath, frame_matrix)

            frame_records.append({
                "id": id,
                "label": row["label"],
                "race": row["race"],
                "gender": row["gender"],
                "vid_filepath": vid_filepath,
                "frame_filepath": frame_filepath,
                "frame_idx": i,
                "source": row["source"],
            })

    frames_df = pd.DataFrame(frame_records)
    return frames_df


def get_frames(subset: bool, save_imgs: bool = False):
    """Load the dataset metadata and extract frames from every video.

    Args:
        subset: If True, use the subset metadata CSV; otherwise the full one.
        save_imgs: If True, save each extracted frame to disk as a JPEG.

    Returns:
        A DataFrame with one row per extracted frame (see
        ``extract_all_img_frames``).
    """
    path_prefix = "../Dataset/"

    df = get_df(path_prefix, subset)
    return extract_all_img_frames(df, path_prefix, save_imgs)