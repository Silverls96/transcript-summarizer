import os
import time
import whisper
import argparse
from rich.console import Console
from dotenv import load_dotenv
from litellm import completion
from prompt import PROMPT

# Load environment variables
load_dotenv()

# Initialize console
console = Console()

# Load Whisper model
stt = whisper.load_model("turbo")

# Get API key and model
api_key = os.environ.get("OPENROUTER_API_KEY")
api_model = os.environ.get("LLM_MODEL")

def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using LiteLLM.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The generated response.
    """
    response = completion(
        model=api_model,  # Using Deepseek via OpenRouter
        messages=[{"role": "user", "content": PROMPT.format(text=text)}],
        api_key=api_key
    )
    
    # Extract the response content
    return response.choices[0].message.content.strip()


# Define function to transcribe audio
def transcribe(audio_file) -> str:
    """
    Transcribes the given audio file using Whisper.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(audio_file, fp16=False, language="id")  # Set fp16=True if using a GPU
    return result["text"].strip()


def transcribe_mel(audio_tensor):
    options = whisper.DecodingOptions(language="id")
    result = whisper.decode(stt, audio_tensor, options)
    text = result.text.strip()
    return text

def process_audio_file(file_path):
    """
    Process a single audio file: transcribe it and generate an LLM response.
    
    Args:
        file_path (str): Path to the audio file.
    """
    console.print(f"[cyan]Processing file: {file_path}")
    
    # # Load audio file
    # # load audio and pad/trim it to fit 30 seconds
    # with console.status(f"Loading audio file {os.path.basename(file_path)}...", spinner="earth"):
    #     audio = whisper.load_audio(file_path)
    #     audio = whisper.pad_or_trim(audio)

    #     # make log-Mel spectrogram and move to the same device as the model
    #     mel = whisper.log_mel_spectrogram(audio, stt.dims.n_mels).to(stt.device)
    # # Transcribe
    # with console.status("Transcribing...", spinner="earth"):
    #     transcription = transcribe_mel(mel)
        
    # Transcribe with timer
    start_transcription = time.time()
    transcription = transcribe(file_path)
    end_transcription = time.time()
    transcription_time = end_transcription - start_transcription

    console.print(f"[yellow]Transcription: {transcription}")
    
    # Generate response with timer
    start_response = time.time()
    response = get_llm_response(transcription)
    end_response = time.time()
    response_time = end_response - start_response
   
    console.print(f"[cyan]Assistant response: {response}")
    
    # Create output directories if they don't exist
    transcribe_dir = "transcribe"
    response_dir = "response"
    
    os.makedirs(transcribe_dir, exist_ok=True)
    os.makedirs(response_dir, exist_ok=True)
    
    # Save transcription with timing info
    transcription_path = os.path.join(transcribe_dir, f"{base_name}_transcription.txt")
    with open(transcription_path, 'w') as f:
        f.write(f"Transcription Time: {transcription_time:.2f} seconds\n")
        f.write(transcription)
    
    console.print(f"[green]Transcription saved to {transcription_path}")
    
    # Save response with timing info
    response_path = os.path.join(response_dir, f"{base_name}_response.txt")
    with open(response_path, 'w') as f:
        f.write(f"Response Generation Time: {response_time:.2f} seconds\n")
        f.write(response)
    
    console.print(f"[green]Response saved to {response_path}")


def process_folder(folder_path, output_dir=None):
    """
    Process all .wav files in a folder.
    
    Args:
        folder_path (str): Path to the folder containing audio files
        output_dir (str, optional): Directory to save output files
    """
    if not os.path.exists(folder_path):
        console.print(f"[red]Folder {folder_path} does not exist.")
        return
        
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Get all .wav files in the folder
    wav_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                if f.lower()]
    
    if not wav_files:
        console.print(f"[yellow]No .wav files found in {folder_path}")
        return
        
    console.print(f"[blue]Found {len(wav_files)} .wav files in {folder_path}")
    
    # Process each file
    for file_path in wav_files:
        if not os.path.exists(file_path):
            console.print(
                f"[red]ERROR: File {file_path} does not exist!  "
                "Skipping this file."
            )
            continue  # Skip to the next file

        try:
            process_audio_file(file_path, output_dir)
            # results.append((file_path, result))
        except Exception as e:
            console.print(f"[red]Error processing {file_path}: {str(e)}")

def process_text_for_response(text, output_dir=None):
    """
    Process a text input and generate an LLM response.
    
    Args:
        text (str): The input text.
        output_dir (str, optional): Directory to save response file.
    """
    console.print(f"[cyan]Processing text: {text}")
    
    # Generate response with timer
    start_response = time.time()
    response = get_llm_response(text)
    end_response = time.time()
    response_time = end_response - start_response
    
    console.print(f"[cyan]Assistant response: {response}")
    
    # Save response with timing information
    if output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save response to a file
        response_path = os.path.join(output_dir, "response.txt")
        with open(response_path, 'w') as f:
            f.write(f"Response Generation Time: {response_time:.2f} seconds\n")
            f.write(response)
        
        console.print(f"[green]Response saved to {response_path}")


# Define function to process text files for response generation
def process_text_files(file_paths, output_dir=None):
    """
    Process specific text files and generate LLM responses for them.
    
    Args:
        file_paths (list): List of text file paths.
        output_dir (str, optional): Directory to save response files.
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    results = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            console.print(f"[red]File {file_path} does not exist.")
            continue
            
        if not file_path.lower().endswith('.txt'):
            console.print(f"[yellow]File {file_path} is not a text file. Skipping.")
            continue
            
        try:
            console.print(f"[cyan]Processing text file: {file_path}")
            
            # Read the text file
            with open(file_path, 'r') as f:
                text = f.read().strip()
                
            console.print(f"[yellow]Text content: {text}")
            
            # Generate response
            process_text_for_response(text, output_dir)
            
            results.append((file_path, text))
            
        except Exception as e:
            console.print(f"[red]Error processing {file_path}: {str(e)}")
    
    # Save a summary file
    if output_dir and results:
        summary_path = os.path.join(output_dir, "text_processing_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Text processing summary\n")
            f.write(f"Processed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files processed: {len(results)}\n\n")
            
            for file_path, text in results:
                f.write(f"File: {os.path.basename(file_path)}\n")
                f.write(f"Text: {text}\n\n")
                
        console.print(f"[green]Summary saved to {summary_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch Audio Processing with Whisper and LiteLLM')
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument('-a', '--audio', action='store_true', help='Process audio files in the default "inputs" folder')
    group.add_argument('-t', '--text', nargs='+', help='Process specific text files for response generation')
    
    args = parser.parse_args()
    
    if args.audio:
        console.print("[cyan]Processing audio files in the default 'inputs' folder.")
        process_folder(os.environ.get("FOLDER", "inputs"))
        
    elif args.text:
        console.print("[cyan]Processing text files for response generation.")
        output_dir = "response"
        process_text_files(args.text, output_dir)
        
    console.print("[blue]Processing completed.")