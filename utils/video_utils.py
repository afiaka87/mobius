import os

import cv2
import numpy as np
from moviepy.editor import AudioFileClip, ImageSequenceClip
from pydub import AudioSegment
from scipy.ndimage import gaussian_filter1d


def convert_audio_to_waveform_video(audio_file: str, video_file: str):

    # convert possible Path to str
    if isinstance(audio_file, os.PathLike):
        audio_file = str(audio_file)
        print(f"Converted audio_file to str: {audio_file}")
    if isinstance(video_file, os.PathLike):
        video_file = str(video_file)
        print(f"Converted output_file to str: {video_file}")

    if not audio_file.endswith(".mp3") and not audio_file.endswith(".wav"):
        raise ValueError("Audio file must be in MP3 or WAV format")

    if not video_file.endswith(".mp4"):
        raise ValueError("Output file must be in MP4 format")

    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file {audio_file} not found")

    # Convert MP3/WAV to raw audio data
    audio = AudioSegment.from_file(audio_file)

    # Convert to mono if it's not already
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Load samples to a numpy array
    samples = np.array(audio.get_array_of_samples())

    # Audio and video properties
    sample_rate = audio.frame_rate
    duration = len(samples) / sample_rate
    video_fps = 30
    frame_count = int(video_fps * duration)

    height, width = 120, 480

    line_color = (118, 168, 91)
    background_color = (34, 34, 34)

    def generate_frame(
        sample_start: int,
        sample_end: int,
        total_frames: int,
        current_frame: int,
        volume_threshold: float = 0.1,
    ) -> np.ndarray:
        # Return a solid background-color frame for the first and last frames
        if current_frame == 0 or current_frame == total_frames - 1:
            return np.full(
                (height, width, 3), background_color, dtype=np.uint8
            )  # early return, no waveform

        frame = np.full((height, width, 3), background_color, dtype=np.uint8)

        # Extract the relevant samples
        y = samples[sample_start:sample_end]

        # Signals will always be speech, so we massage the data to for visualization and aesthetics

        # First, we normalize the signal to fit within the frame height with padding
        padding_factor = 0.1
        y_min = y.min()
        y_max = y.max()
        y_range = y_max - y_min

        if y_range == 0:
            y = np.zeros_like(y)
        else:
            y = ((y - y_min) / y_range) * (height * (1 - 2 * padding_factor)) + (
                height * padding_factor
            )  # normalize

        # Apply Gaussian smoothing for a more natural appearance
        y_smoothed = gaussian_filter1d(y, sigma=5, mode="nearest")

        # Generate x values for the waveform based on the frame width.
        x = np.linspace(0, width, len(y_smoothed), endpoint=False).astype(np.int32)

        # Draw the waveform, omitting parts below the volume threshold
        for i in range(len(x) - 1):
            if y_smoothed[i] > volume_threshold * height:
                pt1 = (int(x[i]), int(y_smoothed[i]))
                pt2 = (int(x[i + 1]), int(y_smoothed[i + 1]))
                cv2.line(frame, pt1, pt2, line_color, 2)

        return frame

    # Generate frames for the video
    # We skip the first and last frames to avoid showing the abrupt start and end
    # of the audio clip
    # We also display samples of the waveform a few frames before the audio starts, for visual effect
    num_frames_before_audio = 0
    frames = []
    samples_per_frame = int(len(samples) / frame_count)

    for i in range(frame_count):
        start_idx = max(0, i * samples_per_frame - num_frames_before_audio)
        end_idx = (i + 1) * samples_per_frame
        frame = generate_frame(start_idx, end_idx, frame_count, i, volume_threshold=0.1)
        frames.append(frame)

    audio_clip = AudioFileClip(audio_file)
    video_clip = ImageSequenceClip(frames, fps=video_fps)
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(
        video_file,
        codec="libx264",
        audio_codec="aac",
        fps=video_fps,
        preset="ultrafast",
    )
    return video_file
