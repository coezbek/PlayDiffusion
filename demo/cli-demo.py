import argparse
import os
from pathlib import Path

def run_asr(audio_path, local_model="tiny"):
    """
    Runs Automatic Speech Recognition on the input audio file.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    print(f"Loading '{local_model}' Whisper model for ASR...")
    try:
      import whisper_timestamped as whisper
    except ImportError:
      print("Please install dependencies for demo such as 'whisper_timestamped' package: uv sync --extra demo")
      # Exit
      exit(1)

    audio_data = whisper.load_audio(audio_path)
    model = whisper.load_model(local_model)

    print("Transcribing audio...")
    transcript = whisper.transcribe(model, audio_data, language="en")
    transcript_text = transcript.get("text", "")

    word_times = []
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            word_times.append({
                "word": word["text"],
                "start": word["start"],
                "end": word["end"]
            })

    print(f"ASR complete. Recognized text: '{transcript_text}'")
    return transcript_text, word_times

def run_inpainter(input_text, output_text, word_times, audio_path):
    """
    Runs the inpainting process to modify the audio.
    """
    # Using default values for advanced options for a minimal CLI

    # Instantiate the main class
    print("Loading PlayDiffusion model for inpainting...")
    from playdiffusion import PlayDiffusion, InpaintInput
    inpainter = PlayDiffusion()

    print("Running inpainting...")
    inpainting_result = inpainter.inpaint(InpaintInput(
        input_text=input_text,
        output_text=output_text,
        input_word_times=word_times,
        audio=audio_path,
        num_steps=30,
        init_temp=1.0,
        init_diversity=1.0,
        guidance=0.5,
        rescale=0.7,
        topk=25,
        audio_token_syllable_ratio=None # Use automatic calculation
    ))
    return inpainting_result

def main():
    """
    Main function to run the CLI demo.
    """
    parser = argparse.ArgumentParser(description="CLI demo for audio inpainting using PlayDiffusion.")
    
    # Sample file contains text: I don't know what happened. I follow the recipe perfectly but the cake just deflated. I'm so embarrassed. I hope no one saw that. I'd be so mortified if they did.
    parser.add_argument("--input_audio", type=str, default="assets/ears_p007_emo_embarassment_sentences.wav", help="Path to the input audio file to be modified.")

    parser.add_argument("--output_text", type=str, default="I don't know what happened. I followed the recipe perfectly but the chocolate mousse just deflated. I'm so embarrassed. . I hope no one saw that. I'd be so mortified if they did.", help="The desired output text to be generated in the audio.")
    parser.add_argument("--output_path", type=Path, default="output.wav", help="Path to save the resulting in-painted audio file.")

    args = parser.parse_args()

    # 1. Run ASR to get the initial text and word timings
    input_text, word_times = run_asr(args.input_audio)

    # 2. Run the inpainter with the ASR output and the desired new text
    sample_rate, audio_data = run_inpainter(input_text, args.output_text, word_times, args.input_audio)

    # 3. Save the output audio
    from scipy.io.wavfile import write as write_wav
    print(f"Saving output audio to {args.output_path}")
    write_wav(args.output_path, sample_rate, audio_data)
    print("Done!")

if __name__ == '__main__':
    main()