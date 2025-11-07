import re

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.models.midashenglm import MiDashengLMModel
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class MiDashengLMMultimodalProcessor(BaseMultimodalProcessor):
    """Multimodal processor for MiDashengLM audio-language model."""

    models = [MiDashengLMModel]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # MiDashengLM uses the same token format as Qwen2Audio
        self.AUDIO_TOKEN = "<|audio_bos|><|AUDIO|><|audio_eos|>"
        self.AUDIO_TOKEN_REGEX = re.compile(
            r"<\|audio_bos\|>(?:<\|AUDIO\|>)+<\|audio_eos\|>"
        )

        # Collect special token ids
        tokenizer = self._processor.tokenizer
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_bos|>")
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")

        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=self.AUDIO_TOKEN,
            audio_token_regex=self.AUDIO_TOKEN_REGEX,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

        # Register audio-related attributes
        self.ATTR_NAME_TO_MODALITY.update({
            "input_values": Modality.AUDIO,
            "audio_length": Modality.AUDIO,
        })

        # Add input_values to FEATURE_NAMES so it's recognized as a feature tensor
        if "input_values" not in self.FEATURE_NAMES:
            self.FEATURE_NAMES.append("input_values")

    def process_mm_data(self, input_text, images=None, videos=None, audios=None, **kwargs):
        """Override to use correct audio parameter name for MiDashengLM processor."""
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos
        if audios:
            # MiDashengLM processor uses 'audio' (singular) like Qwen2Audio
            kwargs["audio"] = audios
            kwargs["audio_kwargs"] = {}
            kwargs["audio_kwargs"].setdefault("truncation", False)

        processor = self._processor
        result = processor.__call__(
            text=[input_text],
            padding=True,
            return_tensors="pt",
            **kwargs,
        )

        # Move feature tensors to CPU if needed (backward compatible check)
        if not getattr(self.server_args, 'keep_mm_feature_on_device', False):
            for feature_name in ["input_values"]:
                if feature_name in result:
                    result[feature_name] = result[feature_name].cpu()

        return result

    async def process_mm_data_async(
        self,
        audio_data,
        input_text,
        **kwargs,
    ):
        """Process audio data for MiDashengLM model.

        Args:
            audio_data: Audio input data
            input_text: Text prompt
            **kwargs: Additional arguments

        Returns:
            Dictionary containing processed multimodal data
        """
        print(f"\n{'='*80}")
        print(f"[PROCESSOR DEBUG] process_mm_data_async called")
        print(f"[PROCESSOR DEBUG] audio_data is not None: {audio_data is not None}")
        print(f"[PROCESSOR DEBUG] input_text: {input_text}")
        print(f"{'='*80}\n")

        # Automatically prepend audio token if not present
        if audio_data and not self.AUDIO_TOKEN_REGEX.search(input_text):
            input_text = f"{self.AUDIO_TOKEN}{input_text}"
            print(f"[PROCESSOR DEBUG] Auto-prepended audio token, new input_text: {input_text}")

        base_output = self.load_mm_data(
            prompt=input_text,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        if base_output is None:
            print(f"[PROCESSOR DEBUG] base_output is None, returning None")
            return None

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )
        print(f"[PROCESSOR DEBUG] mm_items count: {len(mm_items)}")
        print(f"[PROCESSOR DEBUG] ret keys: {list(ret.keys())}")
        for i, item in enumerate(mm_items):
            print(f"[PROCESSOR DEBUG] mm_item[{i}] modality: {item.modality}")
            print(f"[PROCESSOR DEBUG] mm_item[{i}] has feature attr: {hasattr(item, 'feature')}")
            if hasattr(item, 'feature') and item.feature is not None:
                print(f"[PROCESSOR DEBUG] mm_item[{i}] feature shape: {item.feature.shape}")

        # MiDashengLM processor returns input_values (audio waveforms)
        # We need to extract audio_length from the input_values shape
        if "input_values" in ret and len(mm_items) > 0:
            # input_values shape is [batch_size, audio_length]
            input_values = ret["input_values"]
            # For MiDashengLM, audio_length is the actual waveform length
            audio_length = input_values.shape[-1] if input_values.ndim >= 2 else input_values.shape[0]
            mm_items[0].audio_length = audio_length
            print(f"[PROCESSOR DEBUG] Set audio_length={audio_length}, input_values shape: {input_values.shape}")

        result = {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "audio_start_id": self.audio_start_id,
            "audio_token_id": self.audio_token_id,
            "audio_end_id": self.audio_end_id,
        }
        print(f"[PROCESSOR DEBUG] Returning {len(result['mm_items'])} mm_items\n")
        return result
