"""
Face recognition and emotion detection module using DeepFace.

This module provides functions for face detection, recognition, emotion analysis,
and management of a face database for known persons. Uses DeepFace library with
various backend detectors and recognition models.
"""

import asyncio
from typing import List
from av import VideoFrame
import cv2
from deepface import DeepFace
from pandas import DataFrame
from pathlib import Path
import sys
import os
from contextlib import contextmanager
from pipeline.constant import VIDEO_DB_PATH, VIDEO_THRESHOLD


@contextmanager
def suppress_stdout():
    """
    Context manager to suppress stdout output.

    Redirects standard output to /dev/null to hide verbose output from
    libraries like DeepFace during processing.

    :return: Context manager that suppresses stdout within its scope

    Example:
        >>> with suppress_stdout():
        ...     # Code here won't print to stdout
        ...     result = some_verbose_function()
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# THRESHOLD = VIDEO_THRESHOLD
DB_PATH = VIDEO_DB_PATH


def capture_video():
    """
    Capture video frames from the default webcam.

    Opens a video capture window and yields frames continuously until
    the user presses 'q' to quit. Automatically releases resources when done.

    :return: Generator yielding numpy arrays representing video frames in BGR format

    Note:
        This function displays a live webcam feed window. Press 'q' to exit.
    """
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # show the live frame
        cv2.imshow("Webcam Feed", frame)
        # press 'q' to quit the video
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        yield frame
    cap.release()
    cv2.destroyAllWindows()


def detect_emotion(frame):
    """
    Detect and print the dominant emotion in a frame.

    Analyzes facial expressions using DeepFace and prints the detected
    dominant emotion. Uses MediaPipe as the detection backend.

    :param frame: Image frame as numpy array (BGR format from OpenCV)

    Note:
        Prints the dominant emotion to stdout. If no face is detected,
        prints an error message.
    """
    try:
        analysis = DeepFace.analyze(
            frame,
            actions=["emotion"],
            detector_backend="mediapipe",
            enforce_detection=False,
        )
        print(f"Detected emotion {analysis[0]['dominant_emotion']}")  # type: ignore
    except Exception as e:
        print("No face detected: ", e)


def find_person(frame):
    """
    Search for matching faces in the known faces database.

    Compares the face(s) in the given frame against all known faces in
    the database directory. Returns similarity metrics and matches.

    :param frame: Image frame as numpy array containing face(s) to identify
    :return: List of DataFrames, one per detected face. Each DataFrame contains:
             - 'identity': Path to matching face image in database
             - Distance/similarity metrics
             - Bounding box coordinates
             Empty DataFrame if no match found below threshold

    Note:
        Suppresses DeepFace's verbose output during processing.
    """
    person = None
    with suppress_stdout():
        person = DeepFace.find(
            img_path=frame,
            db_path=str(DB_PATH),
            enforce_detection=False,
            batched=False,
            # detector_backend="yolov8",
            # model_name="Facenet512",
        )
    return person


def extract_faces(frame):
    """
    Extract all face regions from a frame.

    Detects and extracts facial regions from the input frame, returning
    cropped face images normalized to [0, 1] range.

    :param frame: Image frame as numpy array (BGR format from OpenCV)
    :return: List of cropped face images as numpy arrays with values in [0, 1].
             Returns empty list if no faces detected.

    Note:
        Face detection is not enforced, so the function won't raise errors
        on images without faces.
    """
    return [
        face["face"] for face in DeepFace.extract_faces(frame, enforce_detection=False)
    ]


def save_new_person(frame):
    """
    Save a new person's face to the database from a file path.

    Creates a new directory with an auto-generated name (person_N) and
    saves the face image as 'snapshot.jpg'.

    :param frame: Path to the image file containing the face to save

    Note:
        The person is assigned an ID based on the current number of
        persons in the database (person_1, person_2, etc.).
    """
    new_name = f"person_{len(list(DB_PATH.iterdir())) + 1}"
    new_dir = DB_PATH / new_name
    new_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(new_dir / "snapshot.jpg"), cv2.imread(frame))


def save_new_person_pyav(frame: VideoFrame):
    """
    Save a new person's face to the database from a PyAV VideoFrame.

    Creates a new directory with an auto-generated name (person_N) and
    saves the frame as 'snapshot.jpg'.

    :param frame: PyAV VideoFrame object containing the face to save
    :return: Auto-generated name for the new person (e.g., "person_1")

    Note:
        This is the preferred method when working with video streams
        using PyAV, as it avoids unnecessary format conversions.
    """
    new_name = f"person_{len(list(DB_PATH.iterdir())) + 1}"
    new_dir = DB_PATH / new_name
    new_dir.mkdir(parents=True, exist_ok=True)

    frame.to_image().save(new_dir / "snapshot.jpg")
    return new_name


if __name__ == "__main__":
    # person = asyncio.run(find_person("./test1.jpg"))
    # print(person)
    image = Path("./test_data/test.jpg")
    # faces = asyncio.run(
    #     # extract_faces("/home/shreyanshp/Projects/Dymensia/pipeline/test1.jpg")
    #     extract_faces(str(image))
    # )
    faces = extract_faces(str(image))
    for cropped_face in faces:
        cv2.imshow("face", cropped_face)
        # print(f"cropped face np array: {face['face']}")
        while cv2.waitKey(1) & 0xFF != ord("q"):
            continue
        # persons: List[DataFrame] = asyncio.run(find_person(cropped_face))  # type: ignore
        persons: List[DataFrame] = find_person(cropped_face)  # type:ignore
        print(type(persons[0]))
        for person in persons:
            print(f"Detected persons: {person['identity']}")
            name = ""
            if person.empty:
                name = save_new_person_pyav(
                    VideoFrame.from_ndarray(
                        (cropped_face * 255).astype("uint8"), format="rgb24"
                    )
                )
            # print(Path(person["identity"].iloc[0]).parent.name)
    # print(asyncio.run(find_person(faces)))
    # print(person[0].columns)
    # if person[0].empty:
    #     save_new_person("./test1.jpg")
    # for frame in capture_video():
    #     detect_emotion(frame)
