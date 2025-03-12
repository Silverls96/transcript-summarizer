import time
import os
import threading
import numpy as np
import whisper
from rich.console import Console
from litellm import completion
from prompt import PROMPT

console = Console()
stt = whisper.load_model("turbo")
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
        messages=[{"role": "user", "content": PROMPT}],
        api_key=api_key
    )
    
    # Extract the response content
    return response.choices[0].message.content.strip()


def transcribe(audio_file) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.

    Args:
        audio file: The audio data to be transcribed.

    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(audio_file, fp16=False, language="id")  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text

def transcribe(audio_tensor):
    options = whisper.DecodingOptions(language="id")
    result = whisper.decode(stt, audio_tensor, options)
    text = result.text.strip()
    return text

def process_audio_file(file_path, output_dir=None):
    """
    Process a single audio file: transcribe it and generate an LLM response.
    
    Args:
        file_path (str): Path to the audio file
        output_dir (str, optional): Directory to save output files
    
    Returns:
        tuple: (transcription, llm_response)
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
    #     transcription = transcribe(mel)
    
        
    # Transcribe
    with console.status("Transcribing...", spinner="earth"):
        transcription = transcribe(file_path)
    console.print(f"[yellow]Transcription: {transcription}")
    
    # Generate response
    with console.status("Generating response...", spinner="earth"):
        response = get_llm_response(transcription)
    console.print(f"[cyan]Assistant response: {response}")
    
    # Save results if output directory is provided
    if output_dir:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Save transcription to a separate file
        transcription_path = os.path.join(output_dir, f"{base_name}_transcription.txt")
        with open(transcription_path, 'w') as f:
            f.write(transcription)
        console.print(f"[green]Transcription saved to {transcription_path}")
        
        # Save LLM response to a separate file
        response_path = os.path.join(output_dir, f"{base_name}_response.txt")
        with open(response_path, 'w') as f:
            f.write(response)
        console.print(f"[green]Response saved to {response_path}")
    
    return transcription, response


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
    results = []
    for file_path in wav_files:
        try:
            result = process_audio_file(file_path, output_dir)
            results.append((file_path, result))
        except Exception as e:
            console.print(f"[red]Error processing {file_path}: {str(e)}")
    
    # Save a summary file
    if output_dir and results:
        summary_path = os.path.join(output_dir, "processing_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Processing summary for folder: {folder_path}\n")
            f.write(f"Processed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files processed: {len(results)}\n\n")
            
            for file_path, (transcription, response) in results:
                f.write(f"File: {os.path.basename(file_path)}\n")
                f.write(f"Transcription: {transcription}\n")
                f.write(f"Response: {response}\n\n")
                
        console.print(f"[green]Summary saved to {summary_path}")
    
    return results


def process_multiple_folders(folders, output_base_dir=None):
    """
    Process multiple folders containing audio files.
    
    Args:
        folders (list): List of folder paths
        output_base_dir (str, optional): Base directory for output files
    """
    all_results = {}
    
    for folder in folders:
        folder_name = os.path.basename(os.path.normpath(folder))
        output_dir = None
        if output_base_dir:
            output_dir = os.path.join(output_base_dir, folder_name)
            
        console.print(f"[blue]Processing folder: {folder}")
        results = process_folder(folder, output_dir)
        all_results[folder] = results
        
    return all_results

def process_text_files(file_paths, output_dir=None):
    """
    Process specific text files and generate LLM responses for them.
    
    Args:
        file_paths (list): List of text file paths
        output_dir (str, optional): Directory to save response files
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
            with console.status("Generating response...", spinner="earth"):
                response = get_llm_response(text)
            console.print(f"[cyan]Assistant response: {response}")
            
            # Save the response
            if output_dir:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                response_path = os.path.join(output_dir, f"{base_name}_response.txt")
                with open(response_path, 'w') as f:
                    f.write(response)
                console.print(f"[green]Response saved to {response_path}")
                
            results.append((file_path, text, response))
            
        except Exception as e:
            console.print(f"[red]Error processing {file_path}: {str(e)}")
    
    # Save a summary file
    if output_dir and results:
        summary_path = os.path.join(output_dir, "text_processing_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Text processing summary\n")
            f.write(f"Processed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files processed: {len(results)}\n\n")
            
            for file_path, text, response in results:
                f.write(f"File: {os.path.basename(file_path)}\n")
                f.write(f"Text: {text}\n")
                f.write(f"Response: {response}\n\n")
                
        console.print(f"[green]Summary saved to {summary_path}")
    
    return results

if __name__ == "__main__":
    console.print("[cyan]Batch Audio Processing with Whisper and LiteLLM started!")
    
    try:
        # Option 1: Process a single folder
        # folder_path = input("Enter the path to the folder containing .wav files: ")
        # output_dir = "output"
        # process_folder(folder_path, output_dir)
        
        # Option 2: Process multiple folders
        while True:
            choice = console.input("[cyan]Choose an option:\n1. Process a single folder of audio files\n2. Process multiple folders of audio files\n3. Process specific text files\n4. Exit\nYour choice: ")
            
            if choice == "1":
                folder_path = console.input("Enter the path to the folder containing .wav files: ")
                output_dir = console.input("Enter the output directory (press Enter to use 'output'): ") or "output"
                process_folder(folder_path, output_dir)
                
            elif choice == "2":
                folders_input = console.input("Enter folder paths (separated by commas): ")
                folders = [f.strip() for f in folders_input.split(",")]
                output_dir = console.input("Enter the base output directory (press Enter to use 'output'): ") or "output"
                process_multiple_folders(folders, output_dir)
            elif choice == "3":
                console.print("[cyan]Select text files to process:")
                select_method = console.input("How would you like to select files?\n1. Enter file paths\n2. Browse interactively\nYour choice: ")
                
                if select_method == "1":
                    files_input = console.input("Enter text file paths (separated by commas): ")
                    file_paths = [f.strip() for f in files_input.split(",")]
                # else:
                #     start_dir = console.input("Enter starting directory (press Enter for current directory): ") or "."
                #     file_paths = select_files_interactively('.txt', start_dir)
                    
                if file_paths:
                    output_dir = console.input("Enter the output directory (press Enter to use 'output'): ") or "output"
                    process_text_files(file_paths, output_dir)
                else:
                    console.print("[yellow]No files selected.")
            elif choice == "4":
                break
                
            else:
                console.print("[red]Invalid choice. Please try again.")
                
    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Processing completed.")