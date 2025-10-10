import argparse
from pathlib import Path
import torch
import torchaudio

import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logging.getLogger('playdiffusion').setLevel(logging.WARNING)
logging.getLogger('playdiffusion.inference').setLevel(logging.INFO)

from playdiffusion.utils.config import PlayDiffusionConfigurable

def main():
    """
    Main function to run the vocoder roundtrip test.
    """
    parser = argparse.ArgumentParser(
        description="Minimal working example for PlayDiffusion vocoder roundtrip (encode->decode)."
    )
    parser.add_argument(
        "--wav_path", type=Path, default="assets/ears_p007_emo_embarassment_sentences.wav", help="Path to the input WAV file."
    )
    parser.add_argument(
        "--output_path", type=str, default="output.wav", help="Path to save the output WAV file."
    )
    parser.add_argument(
        "--config_path", type=Path, default="checkpoints/config.yaml", help="Path to the model config.yaml file (checkpoints/config.yaml if None)."
    )
    args = parser.parse_args()

    # 1. Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load PlayDiffusion models using the config file
    print("Loading PlayDiffusion models...")
    playdiffusion = PlayDiffusionConfigurable(config_path=args.config_path, device=device)
    modelmanager = playdiffusion.mm
    print(" -> Models loaded successfully.")

    # 3. Load and preprocess the input audio
    print(f"Loading audio from: {args.wav_path}")
    try:
        wav_original, sr_original = torchaudio.load_with_torchcodec(args.wav_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return
    print(f" -> Original audio shape: {wav_original.shape}, Sample Rate: {sr_original} Hz")

    # Ensure audio is mono by averaging channels if necessary
    if wav_original.shape[0] > 1:
        wav_original = torch.mean(wav_original, dim=0, keepdim=True)

    # 4. Encode audio waveform into audio tokens (units)
    # The speech tokenizer expects a 16kHz sample rate.
    speech_tokenizer_sr = 16000
    if sr_original != speech_tokenizer_sr:
        print(f"Resampling audio from {sr_original} Hz to {speech_tokenizer_sr} Hz for speech tokenizer...")
        resampler_st = torchaudio.transforms.Resample(sr_original, speech_tokenizer_sr)
        wav_16k = resampler_st(wav_original)
        print(f" -> Resampled audio shape: {wav_16k.shape}")
    else:
        wav_16k = wav_original

    print("Encoding audio to tokens...")
    with torch.inference_mode():
        # The tokenizer expects a 1D tensor for a single file, and returns (1, T)
        audio_tokens, _ = modelmanager.speech_tokenizer.waveform_to_units(wav_16k.squeeze(0).to(device))
        print(f" -> Encoded to audio tokens with shape: {audio_tokens.shape}")

        # 5. Get the voice embedding from the original audio
        # The voice encoder expects a specific sample rate (e.g., 24kHz).
        voice_encoder_sr = modelmanager.voice_encoder.mel_sample_rate
        if sr_original != voice_encoder_sr:
            print(f"Resampling audio from {sr_original} Hz to {voice_encoder_sr} Hz for voice encoder...")
            resampler_ve = torchaudio.transforms.Resample(sr_original, voice_encoder_sr)
            wav_ve = resampler_ve(wav_original)
            print(f" -> Resampled audio shape: {wav_ve.shape}")
        else:
            wav_ve = wav_original

        print("Extracting voice embedding...")
        # get_voice_embedding expects a batch, so the shape should be (B, T)
        # Its output is (B, 1, EmbDim), so we squeeze to (B, EmbDim) for the vocoder
        vocoder_emb = modelmanager.voice_encoder.get_voice_embedding(wav_ve.to(device)).squeeze(1)
        print(f" -> Extracted voice embedding with shape: {vocoder_emb.shape}")

        # 6. Decode audio tokens back into a waveform using the vocoder
        print("Decoding tokens back to waveform...")
    
        # The vocoder expects tokens of shape (B, T) and embedding of shape (B, EmbDim)
        reconstructed_audio = modelmanager.vocoder(audio_tokens.to(device), vocoder_emb.to(device)).squeeze(0) # [1, T]
        print(f" -> Reconstructed audio with shape: {reconstructed_audio.shape}")

    # 7. Save the reconstructed audio to disk
    output_sr = modelmanager.vocoder.output_frequency
    
    # Ensure the output directory exists
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    torchaudio.save_with_torchcodec(
        args.output_path,
        reconstructed_audio.cpu(),
        output_sr
    )
    
    print(f"Saved reconstructed audio to:\n  {args.output_path} (Sample Rate: {output_sr} Hz)")

if __name__ == "__main__":
    main()