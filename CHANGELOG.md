## Unreleased

### Fixed

- RAG embedding now uses llamadart's native batch call instead of one isolate round-trip per
  chunk, sizes chunks with the model's tokenizer instead of a fixed 2000-character cut (which is
  roughly 500–700 tokens and therefore overran the 512-token encoder context on long chunks),
  and reads the embedding width from GGUF metadata instead of spending an inference pass on a
  probe string.
- Disposal no longer races llama.cpp teardown. `LlmModelIsolated`, `LlamaRagCoordinator` and
  `LlamaEmbeddingProvider` used to send a `dispose` message and immediately kill the worker
  isolate (or sleep 200 ms and hope), so `engine.dispose()` rarely finished. They now wait for
  the worker's acknowledgement — with a 5 s timeout — before killing it.
- Cancelling a stream after it finished no longer calls `cancelGeneration()`. That call is
  engine-wide, so with `maxParallelSequences > 1` it could abort an unrelated generation.
- llamadart's typed exceptions survive the worker isolate boundary instead of being flattened
  into `Exception('$e')`. Callers can again tell `LlamaBackendInitializationException` ("GPU
  backend failed, retry on CPU") from `LlamaModelException` ("model file is corrupt") or
  `LlamaUnsupportedException`. The `LlamaException` hierarchy is now re-exported.

### Added

- **Tool calling and structured output.** Declare tools per request with
  `GenerationOverrides.tools` (`LlmTool`), plus `toolChoice`, `parallelToolCalls` and
  `responseFormat`. Completed calls arrive on `StreamingChunk.toolCalls`, reassembled from the
  fragments llamadart streams. None of llamadart 0.8.x's tool work — the area it invested most
  of its releases in — was reachable before. `LlmTool` carries no handler: generation runs in a
  worker isolate, so tool execution stays on the caller's side.
- **KV-cache persistence.** `LocalModel.saveState` / `loadState` / `supportsStatePersistence`
  restore a conversation without re-ingesting its prompt.
- **Tokenization.** `LocalModel.tokenize`, `detokenize`, `countTokens`, `contextSize`,
  `metadata` — the honest way to check what fits in the context window.
- **Diagnostics.** `LocalModel.diagnostics()` reports what the backend actually resolved:
  backend name, layers that really reached the GPU, GGUF file type, vision/audio support, VRAM.
  Previously the only sign that GPU offload had silently fallen back to CPU was the token rate.
- **Runtime LoRA.** `setLora` / `removeLora` / `clearLoras`; adapters were load-time only.
- **System prompts.** All three `sendPrompt*` methods take `systemPrompt`, which is sent as a
  real `LlamaChatRole.system` message instead of being prepended to the user turn where the chat
  template treats it as user text. `RagPipeline` now uses it for its instructions
  (`RagPipeline.defaultSystemPrompt`), and `RagEngine` accepts `systemPrompt:`.
- **Reasoning output.** `StreamingChunk.thinking` surfaces the reasoning channel, which llamadart
  keeps separate from the answer and which this plugin used to drop on the floor — models paid
  full decode latency for invisible output. `LlmConfig.enableThinking` turns it off, and
  `LlmConfig.thinkingBudget` (`ThinkingBudget`) caps tokens per reasoning block.
- **Per-request sampling.** `GenerationOverrides` on any `sendPrompt*` call adjusts temperature,
  token budget, seed, stop sequences, grammar and thinking for one request. Sampling used to be
  frozen when the model loaded, so a low-temperature RAG answer and a higher-temperature chat
  turn needed two loaded models.
- **KV cache quantization.** `LlmConfig.cacheTypeK` / `cacheTypeV` (`KvCacheType.q8_0` halves,
  `q4_0` quarters KV memory) with `flashAttention`, plus `kvUnified`, `useMmap`, `useMlock`,
  `ropeFrequencyBase`, `ropeFrequencyScale`, `presencePenalty` and `speculativeDecoding`.
  `LlmConfig.validate()` rejects a quantized KV cache with flash attention disabled — the
  combination llama.cpp refuses — before the worker isolate starts.
- `StreamingChunk.finishReason` (and the `isTruncated` shorthand): the final chunk now says
  whether generation ended cleanly (`'stop'`) or ran out of token budget (`'length'`). These
  were previously indistinguishable.
- The final chunk's `PerformanceMetrics` now come from llama.cpp's own counters via
  `LlamaEngine.getPerformanceContext()`, adding `promptTokens`, `promptEvalMs`, `evalMs` and an
  `isExact` flag. Live metrics remain chunk-count estimates (`isExact == false`) — a chunk is
  not a token once stream batching kicks in.

### Changed

- **BREAKING (API):** the `images` parameter on `sendPrompt*` is now `attachments`, typed
  `List<LlamaContentPart>`. `LlamaImageContent` still works; audio-capable models (Gemma 4,
  Qwen3-ASR) can now be reached with `LlamaAudioContent`, which the old signature made
  unreachable.
- **BREAKING (defaults):** unset sizing knobs now resolve to llamadart's own auto-sizing instead
  of fixed pre-0.8.16 numbers — `nGpuLayers` to full offload (was 64, which capped offload on
  larger models), `nBatch` to `min(nCtx, 2048)` (was 4096, twice the compute buffer llama.cpp
  asks for and painful on mobile), `nThreads` to llama.cpp's heuristic (was 6). Pass explicit
  values to keep the old behaviour.
- **BREAKING (RAG):** `RagPipeline.defaultPromptTemplate` no longer carries the instruction
  sentence — that moved to `defaultSystemPrompt` and is sent as a system message. A custom
  `promptTemplate` keeps working; it just no longer needs to carry instructions.
- **BREAKING (API):** `dispose()` returns `Future<void>` on `LlmInterface`, `LocalModel`,
  `LlmModelIsolated`, `LlmModelStandard` and `RagEngine`, and must be awaited. `LocalModel.loadModel`
  now awaits the previous model's teardown before loading the next one, instead of letting the
  new model allocate while the old one's native handles were still being freed.
- **BREAKING (iOS integration):** upgraded `llamadart` from `^0.6.10` to `^0.8.22` and raised the
  Flutter constraint to `>=3.38.0`. On iOS the llama.cpp runtime is now linked as an XCFramework
  through Swift Package Manager, so **apps must add `llamadart_llama_cpp_flutter: ^0.0.17` to their
  own `pubspec.yaml`** and must not disable Swift Package Manager. Without it llamadart falls back
  to a Flutter native asset whose generated framework declares `MinimumOSVersion 13.0` against a
  16.4 binary, which App Store validation rejects with ITMS-90208
  ([flutter/flutter#145104](https://github.com/flutter/flutter/issues/145104)).
  No Dart API changes were required.
- The iOS plugin now supports **Swift Package Manager** (`ios/mt_llmkit/Package.swift`) alongside
  CocoaPods; Swift sources moved from `ios/Classes/` to `ios/mt_llmkit/Sources/mt_llmkit/`.
- The privacy manifest (`PrivacyInfo.xcprivacy`) is now actually bundled, via SwiftPM `resources:`
  and the podspec's `resource_bundles`.

### Removed

- Dead, empty `ios/Frameworks/Llama.xcframework` directory tree.

## 0.0.1-beta.1

Initial beta release of **mt_llmkit**.

### Features

- **Local GGUF inference** — run quantized LLMs entirely on-device via [llamadart](https://pub.dev/packages/llamadart), with no internet connection required
- **Two execution backends** — `ModelBackend.isolate` (default, Dart Isolate, no UI jank) and `ModelBackend.inProcess` (lighter startup, supports `clean()`)
- **Three generation methods** on `LocalModel` / `LlmInterface`:
  - `sendPrompt` — raw token stream
  - `sendPromptComplete` — full response as a single `String`
  - `sendPromptStream` — token stream with live `PerformanceMetrics` (recommended)
- **Vision / multimodal** — supports LLaVA, Gemma 3, Qwen VL, SmolVLM and any `libmtmd`-compatible model via `LlmConfig.mmprojPath` and `LlamaImageContent`
- **Performance metrics** — `PerformanceMetrics` with `tokensGenerated`, `durationMs`, `tokensPerSecond`, `msPerToken` updated on every `StreamingChunk`
- **Cloud AI chat providers** — unified `AIChatProvider` interface with implementations for OpenAI, Google Gemini, Anthropic Claude, and Mistral AI; automatic retry with exponential back-off
- **Local RAG pipeline** — fully on-device `RagEngine` with document chunking, embedding (via a separate CPU isolate), cosine-similarity vector search, and optional index persistence
