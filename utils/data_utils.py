from collections import defaultdict
import os
import cv2

splits_for_firstTry = {
    "train": [1, ],
    "val": [0, ],
    "test":  [4, ], 
}

splits = {
    "train": [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
    "val": [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
    "test":  [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47], 
}

class Box:
    def __init__(self, x: int, y: int,w: int, h: int, frame_id: str, action: str):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.frame_id = frame_id
        self.action = action


class ClipAnnotation:
    """Each clip has 9 annotated frames. 
    python class to store annotations for a video clip
    
    Parametrs:
    frame_annotations: dict[str, list[Box]] → A dictionary where:
    The keys are strings (likely frame identifiers, such as frame numbers).
    The values are lists of Box objects, which likely represent bounding boxes for detected objects in each frame.
    """

    def __init__(self,clip_id:str, frame_annotations: dict[str, list[Box]]):
        self.clip_id = clip_id
        self.frame_annotations = frame_annotations


class VideoAnnotation:
    """
    This VideoAnnotation class is designed to store annotations for a video,
    where each video consists of multiple clips,
    and each clip has associated activities and frame-level annotations.

    Prameters:
    video_id: str → A unique identifier for the video.
    clip_activities: dict[str, str] → A dictionary mapping clip IDs (str) to activities (str) performed in that clip (e.g., "walking", "running").
    clip_annotations: dict[str, ClipAnnotation] → A dictionary mapping clip IDs (str) to their respective ClipAnnotation objects (which contain frame-level bounding box annotations).
    """
    def __init__(self, video_id: str, clip_activities: dict[str, str], clip_annotations: dict[str, ClipAnnotation]):
        self.video_id = video_id
        self.clip_activities = clip_activities
        self.clip_annotations = clip_annotations




def _load_clip_annotation(clip_id: str, tracking_annotations_path:str)-> ClipAnnotation:
    """
    This function _load_clip_annotation is designed to load clip-level annotations from a tracking file and return a ClipAnnotation object. 

    Parametrs:
    clip_id: str → The unique identifier for the clip being processed.
    tracking_annotations_path: str → Path to a text file containing tracking annotations.

    Returns: A ClipAnnotation object.
    """
    # Proceccing the file
    player_boxes = defaultdict(list)
    with open(tracking_annotations_path) as f:
        for line in f:
            data = line.split()
            palyer_id, x, y, w, h = map(int, data[:5])
            frame_id = data[5]
            action = data[-1]
            player_boxes[palyer_id].append(Box(x, y, w, h, frame_id, action))
    
    frame_annotations = defaultdict(list)
    for boxes in player_boxes.values():
        for box in boxes[6:14]:
            frame_annotations[box.frame_id].append(box)

    return ClipAnnotation(clip_id, frame_annotations)


def _load_clip_activities(annotation_path: str):
    """
    This function _load_clip_activities is designed to load clip-level activity annotations from a file and return a dictionary mapping clip IDs to activity labels.
    
    """
    clip_activities = {}
    with open(annotation_path) as f:
        for line in f:
            data = line.split()
            clip_id = data[0].split(".")[0]
            clip_activities[clip_id] = data[1]

    return clip_activities


def load_annotations(videos_dir: str, annotations_dir: str) -> dict[str, VideoAnnotation]:
    """
    This function load_annotations is responsible for loading annotations for multiple videos.
    It processes video directories, extracts clip-level activities and frame annotations, 
    and returns a dictionary mapping video IDs to VideoAnnotation objects.

    Parametrs:
    videos_dir: str → Path to the directory containing video folders.
    annotations_dir: str → Path to the directory containing annotation files.

    Returns: A dictionary {video_id: VideoAnnotation}.
    """
    # dict to store annotations for each video.
    annotations = {}
    # Iterating over videos. Retrives all video IDs(folder names) in videos_dir and sorts them.
    for video_id in sorted(os.listdir(videos_dir)):
        # Loading Clip Activities
        video_dir = os.path.join(videos_dir, video_id)
        # 🔴 Skip if not a directory (prevents 'Info.txt' from causing errors)
        if not os.path.isdir(video_dir):
            # print(f"Skipping {video_dir}, not a directory")  # Debugging
            continue
        clip_activities = _load_clip_activities(os.path.join(video_dir, "annotations.txt"))

        # Empty dictionary for clip-level annotations.
        clip_annotations = {}
        # Iterating over clips (Iterates over all clip folders in the current video_dir.)
        for clip_id in os.listdir(video_dir):
            # Processing each Clip
            clip_dir = os.path.join(annotations_dir, video_id, clip_id)
            if not os.path.isdir(clip_dir):
                continue

            # Constructs the annoation file path(clip_id.txt inside clip_dir)
            annotation_path = os.path.join(clip_dir, f"{clip_id}.txt")
            # Calls _load_clip_annotation to extract frame-level bounding boxes.
            clip_annotation = _load_clip_annotation(clip_id, annotation_path)
            # Stors the ClipAnnotation object in clip_annotations
            clip_annotations[clip_id] = clip_annotation

        # Creating a VideioAnnotation object
        annotations[video_id] = VideoAnnotation(video_id, clip_activities, clip_annotations)
    # Returns the final dictionary {video_id: VideoAnnotation}.
    return annotations

"""
Example file structure:

videos/
│── video_001/
│   ├── annotations.txt  # Clip activities
│   ├── clip_001/
│   ├── clip_002/
│
annotations/
│── video_001/
│   ├── clip_001/
│   │   ├── clip_001.txt  # Frame annotations
│   ├── clip_002/
│       ├── clip_002.txt
"""

def visualize_clip(videos_dir: str, video_id: int, clip_id: int, annotations: dict[str, VideoAnnotation]):
    """
    This function visualize_clip is designed to display annotated frames for a given video clip.

    Prameters:
    videos_dir: str → Path to the directory containing videos.
    video_id: int → ID of the video to visualize.
    clip_id: int → ID of the clip within the video.
    annotations: dict[str, VideoAnnotation] → Dictionary of annotations.

    """
    # Extracting the Annotations
    annotation = annotations[str(video_id)]
    # Finds the Clipannotation for the given clip_id. iterates through frames and retrives bounding boxes.
    for frame_id, boxes in annotation.clip_annotations[str(clip_id)].frame_annotations.items():
        # Loading the frame
        image_path = os.path.join(videos_dir, str(video_id), str(clip_id), f"{frame_id}.jpg")
        image = cv2.imread(image_path)

        # Drawing the Bounding Boxes
        for box in boxes:
            cv2.rectangle(image, (box.x, box.y), (box.w, box.h), (0, 255, 0), 2)
            cv2.putText(
                image,
                box.action,
                (box.x, box.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        cv2.imshow(f"Video {video_id} Clip {clip_id}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
