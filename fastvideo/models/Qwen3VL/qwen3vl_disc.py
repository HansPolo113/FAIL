from typing import List, Union

import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import SpectralNorm
from PIL import Image


class SpectralLinear(nn.Linear):
    """Linear layer with spectral normalization"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class SimpleDiscriminatorHead(nn.Module):
    """Simple single linear layer discriminator head"""
    def __init__(
        self,
        hidden_size: int,
        using_spec_norm: bool = True,
    ):
        super().__init__()
        self.head = (SpectralLinear if using_spec_norm else nn.Linear)(hidden_size, 1)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to produce outputs close to 0"""
        # Initialize final layer with small weights and zero bias
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class Qwen3VLDiscriminator(nn.Module):
    """
    VLM-based discriminator using Qwen3-VL as backbone.

    Takes both images and text prompts as input, freezes the visual encoder,
    keeps the language model trainable, and adds a discriminator head.

    Gradient Preservation:
        This discriminator preserves gradients through the image processing pipeline by:
        1. Using Qwen2VLImageProcessorFast which operates on torch tensors
        2. Avoiding PIL image conversions that would break the computational graph
        3. Keeping all image operations on GPU throughout the forward pass
        This enables discriminator gradients to flow back to the policy model in
        adversarial training methods like FAIL.

    Args:
        model_path: Path to Qwen3-VL model checkpoint
        using_spec_norm: Whether to use spectral normalization
        freeze_vision_encoder: Whether to freeze vision encoder (default: True)
        image_size: Target image size for preprocessing
    """

    def __init__(
        self,
        model_path: str,
        using_spec_norm: bool = True,
        freeze_vision_encoder: bool = True,
        image_size: int = 512,
    ):
        super().__init__()
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        # Load pre-trained Qwen3-VL model
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # Load processor
        self.processor = Qwen3VLProcessor.from_pretrained(model_path)

        self.image_size = image_size

        # Get hidden size from language model config before removing LM head
        hidden_size = self.model.language_model.config.hidden_size

        # Remove LM head to save GPU memory
        if hasattr(self.model, 'lm_head'):
            del self.model.lm_head
            self.model.lm_head = None

        # Convert all parameters to bfloat16 to ensure uniform dtype for FSDP
        # Some parameters like LayerNorm might still be in float32 after loading
        self.model = self.model.to(torch.bfloat16)

        # Freeze visual encoder
        if freeze_vision_encoder:
            for param in self.model.visual.parameters():
                param.requires_grad = False
            self.model.visual.eval()

        # Keep language model trainable
        for param in self.model.language_model.parameters():
            param.requires_grad = True

        # Create discriminator head and convert to bfloat16
        self.disc_head = SimpleDiscriminatorHead(
            hidden_size=hidden_size,
            using_spec_norm=using_spec_norm,
        ).to(torch.bfloat16)

        self.freeze_vision_encoder = freeze_vision_encoder

    def _extract_last_token_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract embeddings of the last valid (non-padded) token for each sample.

        Args:
            hidden_states: [batch_size, seq_length, hidden_size]
            attention_mask: [batch_size, seq_length]

        Returns:
            last_token_embeddings: [batch_size, hidden_size]
        """
        batch_size = hidden_states.shape[0]

        # Find the last valid token position for each sample
        # attention_mask is 1 for valid tokens, 0 for padding
        seq_lengths = attention_mask.sum(dim=1)  # [batch_size]
        last_token_indices = seq_lengths - 1  # [batch_size]

        # Clamp to valid range to avoid index errors
        last_token_indices = torch.clamp(last_token_indices, min=0, max=hidden_states.shape[1] - 1)

        # Extract last token embeddings
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        last_token_embeddings = hidden_states[batch_indices, last_token_indices]

        return last_token_embeddings

    def _prepare_images(
        self,
        images: Union[torch.Tensor, List[Image.Image]],
    ) -> Union[torch.Tensor, List[Image.Image]]:
        """
        Prepare images for processor while preserving gradients.

        Args:
            images: Either tensor [batch, 3, H, W] in [-1, 1] range or list of PIL images

        Returns:
            Tensor [batch, 3, H, W] in [0, 1] range or list of PIL images
        """
        if isinstance(images, torch.Tensor):
            # Convert from [-1, 1] to [0, 1] range for the processor
            # Keep as tensor on GPU to preserve gradients
            images = (images + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            return images
        else:
            return images

    def _prepare_text(
        self,
        prompts: List[str],
    ) -> List[str]:
        """
        Prepare text prompts with vision tokens for Qwen3-VL.

        Args:
            prompts: List of text prompts

        Returns:
            List of formatted prompts with vision tokens
        """
        # Format prompts with image placeholders for Qwen3-VL
        # The processor will replace these with appropriate vision tokens
        formatted_prompts = []
        for prompt in prompts:
            # Create message format expected by Qwen3-VL
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},  # Image placeholder
                        {"type": "text", "text": f"{prompt}"},
                    ]
                }
            ]
            # Apply chat template
            formatted_text = self.processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=False,
            )
            formatted_prompts.append(formatted_text)

        return formatted_prompts

    def forward(
        self,
        images: Union[torch.Tensor, List[Image.Image]],
        prompts: List[str],
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through VLM discriminator.

        Args:
            images: Either tensor [batch, 3, H, W] in [-1, 1] range or list of PIL images
            prompts: List of text prompts (length = batch_size)
            return_features: Whether to return intermediate features

        Returns:
            logits: [batch_size, 1] discriminator logits
            (optional) features: last token embeddings if return_features=True
        """
        # Prepare images (preserve gradients if tensor)
        prepared_images = self._prepare_images(images)

        # Prepare text with vision tokens
        formatted_prompts = self._prepare_text(prompts)

        # Get device
        device = next(self.parameters()).device

        # Process through Qwen3-VL processor
        # The processor supports tensor inputs and will preserve gradients
        # IMPORTANT: Tensor images are already in [0, 1] range, so disable rescaling
        batch_inputs = self.processor(
            text=formatted_prompts,
            images=prepared_images,
            padding=True,
            return_tensors="pt",
            do_rescale=False,  # Images already in [0, 1], don't apply 1/255 rescaling
        )

        # Move to same device as model (keep tensors on GPU)
        batch_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch_inputs.items()}

        # Forward through Qwen3-VL model (without LM head)
        # We use the base model to get hidden states
        outputs = self.model.model(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            pixel_values=batch_inputs.get("pixel_values"),
            image_grid_thw=batch_inputs.get("image_grid_thw"),
        )

        # Extract hidden states
        hidden_states = outputs[0]  # [batch_size, seq_length, hidden_size]

        # Extract last token embeddings
        last_token_embeddings = self._extract_last_token_embeddings(
            hidden_states,
            batch_inputs["attention_mask"],
        )

        # Apply discriminator head
        logits = self.disc_head(last_token_embeddings)  # [batch_size, 1]

        if return_features:
            return logits, last_token_embeddings
        return logits

    def train(self, mode: bool = True):
        """Override train to keep vision encoder frozen"""
        super().train(mode)
        if self.freeze_vision_encoder:
            self.model.visual.eval()
        return self

    def save_pretrained(self, path):
        """Save discriminator head and trainable language model parameters"""
        save_dict = {
            'disc_head': self.disc_head.state_dict(),
            'language_model': self.model.language_model.state_dict(),
            'config': {
                'hidden_size': self.model.language_model.config.hidden_size,
            }
        }
        torch.save(save_dict, path)
        print(f"Discriminator checkpoint saved to {path}")

    def load_pretrained(self, path):
        """Load discriminator head and language model parameters"""
        checkpoint = torch.load(path, map_location='cpu')
        self.disc_head.load_state_dict(checkpoint['disc_head'])
        self.model.language_model.load_state_dict(checkpoint['language_model'])
        print(f"Discriminator checkpoint loaded from {path}")
