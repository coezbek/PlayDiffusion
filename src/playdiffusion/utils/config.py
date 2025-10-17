import yaml
from typing import Dict
import torch
import logging
logger = logging.getLogger(__name__)

from playdiffusion import PlayDiffusion
from pathlib import Path

class PlayDiffusionConfigurable(PlayDiffusion):
    """
    A subclass of the main PlayDiffusion class that overrides the model loading
    to use a YAML config file instead of hardcoded paths.
    
    The YAML config `checkpoints/config.yaml` should be placed alongside the checkpoint files 
    downloaded e.g. with `hf download repo --local-dir checkpoints`

    The config file must contain at least the following fields:

    ```yaml
    models:
      inpainter: "last_250k_fixed.pkl"
      text_tokenizer: "tokenizer-multi_bpe16384_merged_extended_58M.json"
      speech_tokenizer: "xlsr2_1b_v2_custom.pt"
      kmeans_layer: "kmeans_10k.npy"
      voice_encoder: "voice_encoder_1992000.pt"
      vocoder: "v090_g_01105000"
    ```
    """
    def __init__(self, config_path: str = "checkpoints/config.yaml", device: str = "cuda"):

        logger.info(f"Loading finetuning config from: {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        for key in ['inpainter', 'text_tokenizer', 'speech_tokenizer', 'kmeans_layer', 'voice_encoder', 'vocoder']:
            if key not in self.config['models']:
                raise ValueError(f"Missing required model path for '{key}' in config.")
            
            self.config['models'][key] = str(Path(config_path).parent.joinpath(self.config['models'][key]).resolve())

        super().__init__(device=device)

    def load_preset(self) -> Dict:
        """ Builds the 'preset' dictionary from paths defined in a config file. """

        model_paths = self.config['models']

        preset = dict(
            vocoder=dict(
                checkpoint=model_paths['vocoder'],
                kmeans_layer_checkpoint=model_paths['kmeans_layer'],
                dtype=torch.float32, # Really unsure why they used half()
            ),
            tokenizer=dict(
                vocab_file=model_paths['text_tokenizer'],
            ),
            speech_tokenizer=dict(
                checkpoint=model_paths['speech_tokenizer'],
                kmeans_layer_checkpoint=model_paths['kmeans_layer'],
                sample_rate=16000,
            ),
            voice_encoder=dict(
                checkpoint=model_paths['voice_encoder'],
            ),
            inpainter=dict(
                checkpoint=model_paths['inpainter'],
                
            ),
        )
        return preset