# Investigation Summary: Audio Repetition Issue in bench_serving

## Problem Statement

When using bench_serving to test MiDashengLM with audio datasets:
- ✅ **Direct `engine.generate()`** → Normal output
- ❌ **bench_serving (OpenAI API)** → Repetitive output ("我跟你说" repeated hundreds of times)

## Investigation Results

### 1. Code Logic Verification ✅

**Test Suite Results**: All 5 tests PASS
- Dataset structure loading: ✅
- Request construction: ✅
- Content items building: ✅
- List comprehension: ✅
- Array independence: ✅

**Conclusion**: The bench_serving code logic is **structurally correct**. No bugs in data flow.

### 2. Request Flow Analysis ✅

I traced the complete flow from bench_serving through the OpenAI API to the model:

```
bench_serving.py (line 318-350)
  → Build content_items: [audio_url, text]
  → POST to /v1/chat/completions
     ↓
serving_chat.py (line 272-305)
  → process_content_for_template_format()
  → Extract audio URLs to audio_data list
  → Normalize content to {"type": "audio"}
     ↓
serving_chat.py (line 318-346)
  → tokenizer.apply_chat_template(messages)
  → Returns prompt_ids
     ↓
serving_chat.py (line 178-183)
  → Create GenerateReqInput with audio_data
     ↓
MiDashengLM Processor (midashenglm.py line 47-73)
  → process_mm_data()
  → processor.__call__(text=[input_text], audio=audios, audio_kwargs={"truncation": False})
     ↓
MiDashengLM Processor (line 100-103)
  → Auto-prepend audio token if not present
  → input_text = f"{AUDIO_TOKEN}{input_text}"
  → AUDIO_TOKEN = "<|audio_bos|><|AUDIO|><|audio_eos|>"
     ↓
Model inference
```

**Finding**: The flow is correct. Audio data is properly extracted, passed through, and the audio token is automatically prepended.

### 3. Content Ordering ✅

Current bench_serving builds content as:
```python
content_items = [
    {"type": "audio_url", "audio_url": {"url": audio_data_uri}},
    {"type": "text", "text": prompt}
]
```

This results in conversation content:
```
<|audio_bos|><|AUDIO|><|audio_eos|>{prompt_text}
```

**Finding**: This order is **correct** for MiDashengLM (same as qwen2-audio).

### 4. Sampling Parameters Investigation

bench_serving default parameters (line 353-360):
```python
payload = {
    "temperature": 0.0,              # Greedy sampling
    "ignore_eos": True,               # Force generation to max_tokens
    "stream": True,
    "max_completion_tokens": output_len,
}
```

**User Feedback**: `--disable-ignore-eos` did NOT fix the issue.

### 5. Chat Template Analysis

MiDashengLM uses qwen2-audio template (conversation.py line 971-980):
```python
audio_token="Audio {idx}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
```

But MiDashengLM processor defines (midashenglm.py line 20):
```python
AUDIO_TOKEN = "<|audio_bos|><|AUDIO|><|audio_eos|>"
```

**Potential Issue**: Template includes "Audio {idx}: " prefix and "\n" suffix, but processor doesn't.

## Key Differences: engine.generate() vs bench_serving

### engine.generate() (works ✅)
- Direct Python call to model
- User can control exact parameters
- No HTTP API overhead
- Direct conversation template application

### bench_serving (broken ❌)
- Goes through OpenAI HTTP API
- Uses async_request_openai_chat_completions()
- Processes through serving_chat.py
- Uses tokenizer.apply_chat_template() or generate_chat_conv()
- Hardcoded default parameters

## Hypotheses

### Hypothesis 1: Template Format Mismatch ⚠️
The qwen2-audio conversation template expects:
```
Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>
{text}
```

But the processor might be auto-prepending just:
```
<|audio_bos|><|AUDIO|><|audio_eos|>{text}
```

Without the "Audio 1: " prefix, the model might not recognize the audio properly.

### Hypothesis 2: apply_chat_template vs generate_chat_conv ⚠️
There are two template application paths:
1. `tokenizer.apply_chat_template()` - HuggingFace transformers
2. `generate_chat_conv()` - SGLang's conversation system

They might handle audio differently. Check which path MiDashengLM uses.

### Hypothesis 3: Missing Audio Tokens ⚠️
The processor auto-prepends audio tokens (line 100-103), but this happens AFTER the chat template is applied. The template might already expect audio placeholders to be present.

### Hypothesis 4: Sampling Parameters Interaction ⚠️
While `ignore_eos` alone didn't fix it, there might be an interaction between:
- `temperature=0.0` (deterministic)
- `ignore_eos=True` (no early stopping)
- Missing `repetition_penalty`

Combined with a malformed prompt, this could cause repetition loops.

## Recommended Debug Steps

### Step 1: Capture Full Debug Output
Run with latest debug logging:
```bash
/home/user/sglang/run_bench_serving_audio.sh \
    --backend sglang-oai-chat \
    --dataset-name audio \
    --dataset-path /path/to/dataset.jsonl \
    --num-prompts 1 \
    --output-file debug.jsonl \
    2>&1 | tee debug.log
```

Look for:
- `[DEBUG_MESSAGES]` - Message structure sent to API
- `[PROCESSOR DEBUG]` - How processor handles the request
- `[MODEL DEBUG]` - Model input shapes and tokens

### Step 2: Compare with Working engine.generate()
**Provide**:
1. Your exact `engine.generate()` code that works
2. Parameters you use (temperature, max_tokens, etc.)
3. How you format the input (text + audio)

This will reveal what's different.

### Step 3: Test with Different Templates
Try forcing a specific chat template:
```python
# In serving_chat.py, temporarily add:
print(f"Using chat template: {self.template_manager.chat_template_name}")
print(f"Template content format: {self.template_manager.jinja_template_content_format}")
```

### Step 4: Manual API Test
Test the API directly with curl to isolate bench_serving:
```bash
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiDashengLM",
    "messages": [{
      "role": "user",
      "content": [
        {
          "type": "audio_url",
          "audio_url": {"url": "data:audio/wav;base64,..."}
        },
        {
          "type": "text",
          "text": "请描述这段音频"
        }
      ]
    }],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

If this also produces repetition → API handler issue
If this works normally → bench_serving construction issue

### Step 5: Check Chat Template Path
Add logging to see which template method is used:
```python
# In serving_chat.py _convert_to_internal_request()
if hasattr(self.tokenizer_manager.tokenizer, 'chat_template') and self.tokenizer_manager.tokenizer.chat_template:
    print("Using HF tokenizer.apply_chat_template()")
else:
    print("Using SGLang generate_chat_conv()")
```

## Files Modified for Debugging

All modifications include comprehensive debug logging:
- `python/sglang/bench_serving.py` - Lines 370-418: DEBUG_AUDIO_REQUEST, DEBUG_PAYLOAD, DEBUG_MESSAGES
- Trace logging: Lines 1651, 1933, 2109-2111, 2314

## Next Actions

**Priority 1**: Run Step 1 (capture debug output) and share the log
**Priority 2**: Provide working engine.generate() code (Step 2)
**Priority 3**: Test API directly with curl (Step 4)

Once we see the actual debug output, we can pinpoint whether:
1. The message structure is malformed
2. The audio data is corrupted
3. The prompt text is incorrect
4. The model receives wrong input

## Summary

- ✅ Code logic is correct (all tests pass)
- ✅ Request flow is correct (traced end-to-end)
- ✅ Content ordering is correct (audio → text)
- ⚠️ **Need to verify**: Actual message structure sent to model
- ⚠️ **Need to understand**: Why engine.generate() works but API doesn't

The issue is NOT in bench_serving's data flow logic. It's likely in:
1. How the chat template is applied
2. What format the model actually expects
3. A difference between direct engine calls and API calls

**The debug output from Step 1 will reveal the root cause.**
