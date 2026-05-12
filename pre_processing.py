import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

FRAMES_PER_VID = 5
PATH_PREFIX = "/Users/annastevens/Documents/dataset-subset/"

def setup() -> pd.DataFrame:
    """Create dataframe from CSV and preprocess data (remove unneccessary fields, entries, etc.)
    """

    df = pd.read_csv("meta_data.csv")

    # combine filename and path columns
    df["vid_filepath"] = PATH_PREFIX + df["path"].str.cat(df["filename"], sep="/")

    # drop unnecessary columns
    df = df.drop(columns=["target1", "target2", "method", "type", "source", "path", "filename"])

    # remove any rows with fake audio (not looking at this)
    df = df[df.category != "B"]
    df = df[df.category != "D"]

    # replace category A and C with real and fake
    df["category"] = df["category"].replace('A', 'real')
    df["category"] = df["category"].replace('C', 'fake')

    # rename column
    df.rename(columns={"category": "label"}, inplace=True)
    
    # add unique id column at start
    df.insert(0, "id", range(len(df)))

    return df


def extract_img_frames(video_path: str) -> list[np.ndarray]:
    """Return evenly-spaced BGR frames from a video file.
    REF: claude was used to help make this function and https://www.geeksforgeeks.org/python/python-program-extract-frames-using-opencv/

    Params (do proper docstrings later but for now so that I know what it returns and takes in):
    video_path: full file path (not relative) of singular video to process (as a str)

    Returns:
    the 5 frames as a numpy array each in one big list (inc. shape and dtype info)
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
    indices = [int(total_frames * i / FRAMES_PER_VID) for i in range(FRAMES_PER_VID)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frames.append(frame)

    cap.release()
    return frames


def extract_all_img_frames(df: pd.DataFrame, save_imgs: bool = False) -> pd.DataFrame:
    """ Extract 5 frames from each video for all vids in dataset
    REF: claude was used to help enumerate through dataset - tqdm for nice loading

    Params:
    df: the data from the dataset (each entry in dataset will have name, filepath, label (real or fake), and ethnicity)
    save_imgs: bool, if true will save image files to specified location

    Returns:
    new dataframe with the img frame data + vid data
    """

    frame_records = []

    # loop through each data row in dataset df
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting frames"):
        vid_filepath = row["vid_filepath"]
        id = row["id"]
        frames = extract_img_frames(vid_filepath)

        # loop through each frame in vid
        for i, frame_matrix in enumerate(frames):
            frame_filepath = f"{PATH_PREFIX}frames/{id}_{i}.jpg"
            
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
                "frame_matrix": frame_matrix,
            })
            
    frames_df = pd.DataFrame(frame_records)
    return frames_df


if __name__ == "__main__":
    df = setup()
    frames = extract_all_img_frames(df, False)
    print(frames)
