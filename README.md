# Batch Audio Processing with Whisper and LiteLLM

This project is designed to process audio files using Whisper for speech-to-text transcription and LiteLLM for generating responses to the transcribed text. It supports both batch processing of audio files and direct text input for response generation.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Features](#features)
- [Troubleshooting](#troubleshooting)

## Introduction

This tool automates the process of transcribing audio files and generating intelligent responses using LiteLLM. It is built with Whisper for transcription and utilizes LiteLLM's API for response generation.

## Requirements

- **Python 3.8+**: Ensure you have Python installed on your system.
- **Whisper Model**: Uses the Whisper "turbo" model for transcription.
- **LiteLLM API Key**: Requires an API key for LiteLLM.
- **OpenRouter API Key**: Needed if using OpenRouter for LiteLLM access.
- **Rich Console**: For enhanced console output.
- **Litellm Library**: For interacting with LiteLLM.
- **ffmpeg**: For load audio.

### Installation

To install the required packages, run:

```bash
pip install -r requirements.txt
```

To install ffmpeg, run:
```bash
choco install ffmpeg (https://chocolatey.org/install)
```

## Usage

### Command-Line Arguments

The project uses `argparse` for command-line arguments. You can choose to process audio files in a default "inputs" folder or provide paths to specific text files.

#### Process Audio Files

```bash
python app.py -a
```

#### Process Text Files

```bash
python app.py -t file1.txt file2.txt
```

### Environment Variables

Set the following environment variables in a `.env` file:

- `API_KEY`: Your OpenRouter API key.
- `LLM_MODEL`: The LiteLLM model to use. (e.g openrouter/deepseek/deepseek-r1-zero:free, azure_ai/DeepSeek-R1)
- `FOLDER`: The folder where your .wav or audio files located (Optional)
- `API_VERSION`: The API version of the models (Optional)
- `LLM_URL`: The url base api of the models (Optional)

### Run Bat File
```bash
run.bat
```

## Features

- **Batch Audio Processing**: Transcribes multiple audio files and generates responses.
- **Text Input Support**: Allows processing text files directly for response generation.
- **Timing Information**: Includes transcription and response generation times in output files.
- **Output Organization**: Saves transcriptions and responses in separate folders.

## Troubleshooting

- **Missing Dependencies**: Ensure all required packages are installed.
- **API Key Issues**: Verify that your API keys are correct and properly set in the `.env` file.
- **File Not Found Errors**: Check that all file paths are correct and accessible.
