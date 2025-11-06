import re

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.models.midashenglm import MiDashengLMForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class MiDashengLMMultimodalProcessor(BaseMultimodalProcessor):
    """Multimodal processor for MiDashengLM audio-language model."""

    models = [MiDashengLMForConditionalGeneration]

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

        self.ATTR_NAME_TO_MODALITY.update({"audio_length": Modality.AUDIO})

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
        base_output = self.load_mm_data(
            prompt=input_text,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        if base_output is None:
            return None

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        # MiDashengLM uses audio_length to track audio duration
        assert (
            "audio_length" in ret
        ), "audio_length not found in processor output"

        # Store audio length information
        audio_lengths = ret["audio_length"]
        if len(mm_items) > 0:
            mm_items[0].audio_length = audio_lengths[0] if len(audio_lengths) > 0 else None

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "audio_start_id": self.audio_start_id,
            "audio_token_id": self.audio_token_id,
            "audio_end_id": self.audio_end_id,
        }
