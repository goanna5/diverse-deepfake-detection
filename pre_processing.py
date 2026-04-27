import numpy as np
import cv2


def extract_img_frames(video_path: str) -> list[np.ndarray]:
    """Return evenly-spaced BGR frames from a video file.
    REF: claude was used to help make this function and https://www.geeksforgeeks.org/python/python-program-extract-frames-using-opencv/

    Params (do proper docstrings later but for now so that I know what it returns and takes in):
    video_path: full file path (not relative) of singular video to process (as a str)

    Returns:
    the 5 frames as a numpy array each in one big list (inc. shape and dtype info)
    """
    n = 5 # num frames to extract

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    # Spread indices evenly across the video duration
    indices = [int(total_frames * i / n) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frames.append(frame)
            #cv2.imwrite(f"/Users/annastevens/Documents/frame{idx}.jpg", frame)

    cap.release()
    return frames

#print(len(extract_img_frames("/Users/annastevens/Downloads/sample_vid.mov")))