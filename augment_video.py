from vidaug import augmentors as va
import cv2
import numpy as np
import time

def write_video(output, images):
    """
    Outputs a video file from a list of images
    """
    # Define the video shape
    width = images[0].shape[0]
    height = images[0].shape[1]
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))
    # Write out frame to video
    for image in images:
        out.write(image)
    # Release everything if job is finished
    out.release()


def select_frames(frames, frames_per_video):
    """
    Select a certain number of frames determined by the number (frames_per_video)
    :param frames: list of frames
    :param frames_per_video: number of frames to select
    :return: selection of frames
    """
    step = len(frames)//frames_per_video
    if step == 0:
        step = 1
    first_frames_selection = frames[::step]
    final_frames_selection = first_frames_selection[:frames_per_video]

    return final_frames_selection


def get_rgb_videoclip(rgb_videoclip, frames_per_video, frame_height, frame_width):
    """
    From an RGB channeled video clip returns a random frame and a number of frames indicated by frames_per_video
    :param rgb_videoclip: the source video clip in RGB
    :param frames_per_video: number of frames per video to select
    :param frame_height: frame height
    :param frame_width: frame width
    :return: selected number of frames
    """
    cap = cv2.VideoCapture(rgb_videoclip)

    frames = list()
    if not cap.isOpened():
        cap.open(rgb_videoclip)
    ret = True
    while (True and ret):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

    # The following operations are intended to select a precise number of frames
    # and to resize them according to the decided setup of frame_height/width

    selected_frames = select_frames(frames, frames_per_video)

    # Application of data augmentation
    seq = augmentor(selected_frames[0].shape)
    selected_frames = seq(selected_frames)

    # Resizing frames to fit the decided setup
    resized_selected_frames = list()
    for selected_frame in selected_frames:
        resized_selected_frame = cv2.resize(selected_frame, (frame_width, frame_height))
        resized_selected_frames.append(resized_selected_frame)

    # return frame, video_clip
    return np.asarray(resized_selected_frames)

def augmentor(frame_shape):
    """
    Prepares the video data augmentator by applying some augmentations to it
    """
    height = frame_shape[0]
    width = frame_shape[1]
    sometimes = lambda aug: va.Sometimes(0.5, aug)  # Used to apply augmentor with 50% probability

    seq = va.Sequential([
        # randomly crop video with a size of (height-60 x width-60)
        # height and width are in this order because the instance is a ndarray
        sometimes(va.RandomCrop(size=(height - 60, width - 60))),
        sometimes(va.HorizontalFlip()),  # horizontally flip
        sometimes(va.Salt(ratio=100)),  # salt
        sometimes(va.Pepper(ratio=100))  # pepper
    ])
    return seq

if __name__ == "__main__":
    # Video arguments
    rgb_videoclip = "some_random_videoclip.mp4"
    frames_per_video = 20
    frame_height = frame_width = 224
    iterations_number = 1
    start = time.time()

    # Apply video data augmentation
    iterations = range(0, iterations_number)
    for iteration in iterations:
        video_aug_filename = "some_random_video_augmented_"+str(iteration)+".mp4"
        video = get_rgb_videoclip(rgb_videoclip, frames_per_video, frame_height, frame_width)
        write_video(video_aug_filename, video)
    end = time.time()

    # Compute elapsed time
    print(end-start)