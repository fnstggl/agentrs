"""Workstream A: semantic grounding and executable actions (spec Parts 2 and 9).

encoder.QwenSemantic        pretrained semantic backbone (token states, embeddings, LoRA decoder path, cache)
tokenizer.ObservationTokenizer   observations / tools / goals / actions -> CAM tokens
bridge.LatentBridge         P = W(R, G, B): soft prefix embeddings for the decoder
grammar                     schema-walking constrained argument generation (typed heads + Qwen strings)
actions.ActionGenerator     pi(tool) x pi(evidence) x prod pi(arg parts); sample / exact log_prob
sandbox                     local sandboxed code execution (Crucible's Landlock+seccomp runner)
objectives                  semantic learning losses (spec 2.3)
"""
