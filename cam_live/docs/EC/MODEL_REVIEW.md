# Visual-grounding models: licence, revision, code review, how each is run

Reviewed 2026-11-26/27 by workstream EC. Weights are in `HF_HOME=/workspace/.hf_home`, downloaded once with
`HF_HUB_OFFLINE=0`; at run time everything is offline (`HF_HUB_OFFLINE=1`, exact snapshot directory, no network).
**No model is run with `trust_remote_code`. No custom model code from any of the three repositories is executed.**
Only safetensors weights are loaded (no pickle file was downloaded). All models are frozen (`eval()`,
`requires_grad_(False)`) and loaded lazily; the runtime path is one child process per model with an empty
environment (`grounding/worker.py`), and in-process loading is refused in a process marked as holding credentials.

Numbers below: `docs/evidence/EC/grounding_bench.json` (80 targets on 60 generated canvas pages, RTX A6000 shared
with other jobs, load average 8–10 during the run), command `python -m cam_live.harness.grounding.bench
--per-family 10`.

| | GoClick-Base | MAI-UI-2B | Vocaela-2-500M-1024R2 | reference baseline |
|---|---|---|---|---|
| repository | `HongxinLi/GoClick-Base` | `Tongyi-MAI/MAI-UI-2B` | `vocaela/Vocaela-2-500M-1024R2` | `grounding/reference.py` |
| pinned revision | `05633a495f9d2ce823372d0c7fd9b0b4bb67c6fa` | `503050934809558c8dfd2ddedaf9621fa74ac2de` | `b0280f2ce8e2f126a1c3c446c70e63ee7bbeaed2` | builtin-1 |
| licence (card metadata, licence file) | MIT | Apache-2.0 | **CC BY-NC-SA 4.0** | part of cam_live |
| custom code in the repository | yes (3 files, 189 kB) | none | none | – |
| class used | `transformers` Florence-2 (native) | `Qwen3VLForConditionalGeneration` | `SmolVLMForConditionalGeneration` | numpy |
| parameters | 270.8 M | 2 127.5 M | 498.0 M | 0 |
| GPU memory after load / peak | 528 / 695 MB | 4 059 / 4 299 MB | 971 / 1 088 MB | 0 |
| needs the shared GPU lock (> 4 GB) | no | **yes** | no | no |
| grounding accuracy | 0.925 | 0.938 | 0.988 | 0.500 |
| end-to-end operation success | 0.925 | 0.938 | 0.988 | 0.500 |
| incorrect actions / no effect / no proposal | 2 / 4 / 0 | 0 / 5 / 0 | 0 / 1 / 0 | 10 / 20 / 10 |
| model latency mean / median / p90 | 0.253 / 0.115 / 0.117 s | 1.720 / 1.639 / 1.759 s | 1.034 / 0.954 / 0.967 s | 1.259 / 0.917 / 2.995 s (CPU) |
| compute per successful operation | 0.273 s | 1.834 s | 1.047 s | 2.519 s (CPU) |
| recovered with the fallback grounder | 5 of 6 (MAI-UI-2B) | 4 of 5 (GoClick) | 0 of 1 (GoClick) | 34 of 40 (GoClick) |

Per family (grounding accuracy = end-to-end success in every cell of this run):

| family | GoClick-Base | MAI-UI-2B | Vocaela-2 | baseline |
|---|---|---|---|---|
| canvas buttons | 1.0 | 1.0 | 1.0 | 1.0 |
| icon-only controls | 0.9 | 1.0 | 1.0 | 0.0 |
| dense form | 0.9 | 0.5 | 1.0 | 0.0 |
| small targets (12–16 px) | 0.6 | 1.0 | 0.9 | 0.0 |
| look-alike distractors | 1.0 | 1.0 | 1.0 | 0.0 (10 wrong clicks) |
| three-step workflow (30 targets) | 1.0 | 1.0 | 1.0 | 1.0 |

What the numbers do not show: these are 80 synthetic targets drawn by one renderer in one visual style. They
say which model fails on which kind of control here; they are not an estimate for arbitrary websites. The
"more capable" model is not uniformly better (MAI-UI-2B clicked the label instead of the input field in 5 of 10
dense-form tasks), which is why the order of grounders is a prior to be calibrated from experience and not a rule.

## GoClick-Base (lightweight candidate, used by default)

* **Licence.** MIT (`license: mit` in the card). The card's title says "GoClick-Large" and lists
  `base_model: microsoft/Florence-2-large`, but the configuration and weights are the base model (6+6 BART layers,
  d=768, DaViT 128/256/512/1024).
* **Files downloaded.** `README.md`, `config.json`, `generation_config.json`, `preprocessor_config.json`,
  `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `model.safetensors` and, for review only,
  `configuration_florence2.py`, `modeling_florence2.py`, `processing_florence2.py`. **Not downloaded:**
  `optimizer.pt`, `pytorch_model.bin`, `scheduler.pt`, `rng_state_*.pth`, `training_args.bin` (pickle formats;
  never needed).
* **Custom code: what was read.** Pattern scan of the three files for `os.system`, `subprocess`, `eval(`,
  `exec(`, `pickle`, `requests`, `urllib`, `socket`, `open(`, `__import__`, `importlib`, `shutil`, `os.environ`,
  `torch.load`, `http`: the only hits are URLs in comments/docstrings and a helper `is_url` that tests
  `str.startswith("http")`. Imports: `torch`, `einops`, `timm` (`DropPath`, `trunc_normal_`), `transformers`,
  optional `flash_attn`. Read in full: the import block and class list of `modeling_florence2.py`,
  `ChannelAttention`, `_encode_image`, `_merge_input_ids_with_image_features`; in `processing_florence2.py` the
  constructor block that appends 1 024 tokens to the tokenizer and `_construct_prompts`. The remaining ~120 kB
  (BART copy, DaViT blocks, post-processing for other Florence tasks) was **not** read line by line.
* **Decision.** The custom code is not executed, so unread lines cannot run. It also could not have been used as
  published: the card requires `transformers==4.45`, `timm` and `einops`; this environment has `transformers` 5.24
  and neither package. Instead the checkpoint is loaded into the Florence-2 implementation inside `transformers`
  (`grounding/goclick.py`):
  * tensors are renamed by `convert_state_dict` (7 prefix rules + 5 block rules; `image_projection` is transposed
    into a `Linear` weight). Loading fails unless every tensor lands on a parameter of identical shape and nothing
    but the two tied embedding copies is left uninitialised;
  * `tie_word_embeddings=False`: in the published checkpoint the output head differs from the input embedding
    (max |difference| 0.37). With the default tying every answer collapsed to nearly the same point — found by
    measuring, see the note in `goclick.py`;
  * the tokenizer gets the same 1 024 extra tokens in the same order (`<loc_0>` must be id 50269, vocabulary
    51 289, checked at load); images are resized to 768×768 bicubic with ImageNet normalisation; the encoder input
    is `[577 image tokens, prompt tokens]`; decoding is a 3-beam search over at most 12 tokens written in
    `goclick.py` (the checkpoint's `no_repeat_ngram_size=3` is not applied: irrelevant for a 5-token answer).
* **Equivalence evidence.** There is no bit-level comparison with the original code (it cannot run here). The
  evidence is behavioural: identical outputs on CPU float32 and GPU float16, 74 of 80 benchmark targets correct,
  and the gpu-marked tests.
* **Reported confidence** is the product of the probabilities of the two coordinate tokens; it was 1.00 on
  almost every benchmark answer including wrong ones, so it is not usable as a calibrated signal.

## MAI-UI-2B (optional advanced model)

* **Licence.** Apache-2.0 (card metadata). Weights and configuration only; there is no code in the repository.
* **How it is run.** `Qwen3VLForConditionalGeneration` from `transformers`, bfloat16, GPU only. The grounding
  system prompt, the message order (instruction, then image) and the answer convention
  (`<answer>{"coordinate": [x, y]}</answer>` on a 0–999 scale) were taken from the publisher's public repository
  (`github.com/Tongyi-MAI/MAI-UI`: `MAI-UI/src/prompt.py`, `MAI-UI/src/mai_grounding_agent.py`, fetched read-only
  on 2026-11-26; the Hugging Face card does not document them). Greedy decoding, at most 256 new tokens.
* **What was tried.** (1) `AutoProcessor` — fails: the packaged Qwen3-VL processor constructs a video processor
  that needs `torchvision`, which is not installed (installing it would bring a second `torch` into the venv).
  (2) `Qwen3VLProcessor(video_processor=None)` — fails inside `ProcessorMixin`. (3) Used: the PIL image processor
  (`Qwen2VLImageProcessorPil`, verified equal to the reference patch layout to 0.0 difference) plus the tokenizer
  with the repository's chat template, placeholder expanded as the processor does. First result: predicted x
  followed true x with slope 0.50 (y: 1.07). Cause: this `transformers` version computes 2-D rotary positions for
  image tokens only when `mm_token_type_ids` is passed. After adding it: slope 0.997 / 1.012, mean error 5 px,
  14 of 14 single-button probes.
* **Resources.** 4.3 GB peak GPU memory: above the 4 GB line of the shared GPU, so `heavy=True`; the visual
  executor takes the shared lock around each call and closes the model before releasing it, the benchmark held
  the lock for 197 s. Its text output other than the coordinate (it writes a short "reasoning" string) is ignored.

## Vocaela-2-500M-1024R2 (optional, evaluation only)

* **Licence.** CC BY-NC-SA 4.0 (`LICENSE.md` in the repository: non-commercial, share-alike, because of its
  training data). **Consequence:** it is integrated for benchmarking only. `VocaelaGrounder` refuses to load
  unless `accept_noncommercial=True`, the registry marks it `noncommercial`, and it is not in
  `grounding.DEFAULT_ORDER` or `VisualConfig.grounder_order`. Whether the operator's use is non-commercial is the
  operator's decision; it must not be enabled for a live objective without that decision.
* **How it is run.** `AutoModelForImageTextToText` (SmolVLM class in `transformers`), float16. System message and
  JSON action format from the model card. Only the first `coordinate` is read; the `action` it names is ignored.
  The packaged `SmolVLMProcessor` needs `torchvision` and `num2words`; inputs are built with
  `SmolVLMImageProcessorPil`, the tokenizer's chat template and `transformers`' own `get_image_prompt_string`.
* **Result.** Best accuracy of the four on this benchmark (79 of 80) at about 1 GB; it cannot be the default
  because of its licence.

## Reference grounder (baseline for comparison only)

Rendered-text template matching with the fonts installed on this host plus a colour-blob rule; no model. It finds
controls that carry the literal label (canvas buttons and the workflow: 40 of 40) and nothing else; on look-alike
rows it clicked a wrong control 10 times out of 10. It exists to show what "no grounding model" costs.
