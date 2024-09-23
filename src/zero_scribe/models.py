from pydantic import BaseModel, Field
from pathlib import Path
import zipfile
import os
import requests
from typing import Any, List
import whisperx

from zero_scribe.consts import WHISPER_BATCH_SIZE, WHISPER_DEVICE
from zero_scribe.ml_models import load_whisper_model


class CraigAudioFile(BaseModel):
    """Craig Audio File

    The output of unzipping the download from craig
    This model represents a single audio file
    The discord username is extracted from the file name
    """

    path: Path = Field(
        ..., title="Path", description="The path to the Craig audio file"
    )
    user_name: str = Field(
        None,
        title="User Name",
        description="The name of the user who recorded the audio",
    )

    def model_post_init(self, __context: Any) -> None:
        # parse the username from the file name
        self.user_name = self.path.stem.split("-")[1].split("_")[0]


class CraigAudioData(BaseModel):
    """Craig Unzipped

    Holds data about the unzipped Craig audio files
    Includes a class method to download and unzip the file
    """

    files_path: Path = Field(
        ...,
        title="Files Path",
        description="The base path of the unzipped Craig audio files",
    )
    audio_files: List[CraigAudioFile] = Field(
        default_factory=list,
        title="Audio Files",
        description="The audio files that were unzipped from the Craig download",
    )
    info_file: Path = Field(
        None,
        title="Info File",
        description="The info file that was unzipped from the Craig download",
    )

    def model_post_init(self, __context: Any):
        valid_extensions = {".aac", ".wav", ".flac"}
        audio_files = [
            file
            for file in self.files_path.iterdir()
            if file.suffix in valid_extensions
        ]
        self.audio_files = [CraigAudioFile(path=file) for file in audio_files]
        self.info_file = self.files_path / "info.txt"

    @classmethod
    def from_url(cls, url: str, output_dir: Path):
        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define the path for the downloaded zip file
        zip_path = output_dir / "downloaded_file.zip"

        # makedir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download the file
        response = requests.get(url)
        with open(zip_path, "wb") as file:
            file.write(response.content)

        # Unzip the file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)

        # Delete the zip file
        os.remove(zip_path)
        return cls(base_path=output_dir)


class GameTranscription(BaseModel):
    """Game Transcription

    Holds references to multiple transcriptions
    Includes a class method to generate multiple transcriptions from a CraigUnzipped object

    Includes an output method for generating a single merged transcription
    """


class TranscriptionSegment(BaseModel):
    """Transcription Segment

    The output of whisperX is a list of segments
    Each segment has a start and end time, the text, and the words
    """

    start: float
    end: float
    text: str
    words: List[dict]
    user_name: str


class WhisperXTranscription(BaseModel):
    """WhisperX Transcription

    The output of whisperX coerced to a pydantic model
    Includes and additional field for the user name
    """

    segments: List[TranscriptionSegment]
    user_name: str

    @classmethod
    def from_craig_audio_file(cls, craig_file: CraigAudioFile):
        model = load_whisper_model()
        audio = whisperx.load_audio(craig_file.path)
        result = model.transcribe(audio, batch_size=WHISPER_BATCH_SIZE)

        language = result.get("language", "en")

        model_a, metadata = whisperx.load_align_model(
            language_code=language, device=WHISPER_DEVICE
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            WHISPER_DEVICE,
            return_char_alignments=False,
        )

        # apply username to all segments
        for segment in result["segments"]:
            segment["user_name"] = craig_file.user_name

        transcription = cls(user_name=craig_file.user_name, **result)
        return transcription


class MultiTranscript(BaseModel):
    """Multi-Transcription

    Holds references to multiple transcriptions
    Includes a class method to generate multiple transcriptions from a CraigUnzipped object

    Includes an output method for generating a single merged transcription
    """

    transcriptions: List[WhisperXTranscription] = []

    def save_merged_transcription(self, output_path: Path):
        """Create a single merged transcription from the multiple transcriptions"""

        # combine all segments and sort by start time
        all_segments: List[TranscriptionSegment] = []
        for transcription in self.transcriptions:
            all_segments.extend(transcription.segments)
            all_segments.sort(key=lambda x: x.start)

        # iterate through all segments and print out the speaker and text
        last_speaker = None
        full_transcript = ""
        for segment in all_segments:
            if segment.user_name != last_speaker:
                start_time: float = segment.start
                hours = int(start_time // 3600)
                minutes = int((start_time % 3600) // 60)
                seconds = int(start_time % 60)
                start_time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
                if last_speaker is not None:
                    full_transcript += "\n\n"
                full_transcript += f"{segment.user_name} ({start_time_str}):\n"
            # add spaces between subsequent segments from the same speaker
            if segment.user_name == last_speaker:
                full_transcript += f" {segment.text}"
            else:
                full_transcript += f"{segment.text}".strip()
            # full_transcript += f"{segment.text}"
            last_speaker = segment.user_name

        # save to output path
        with open(output_path, "w") as f:
            f.write(full_transcript)

    @classmethod
    def from_craig_data(cls, craig_dir: CraigAudioData):
        transcriptions = []
        for audio_file in craig_dir.audio_files:
            transcription = WhisperXTranscription.from_craig_audio_file(audio_file)
            transcriptions.append(transcription)
        return cls(transcriptions=transcriptions)
