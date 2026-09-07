# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mt_llmkit** is a Flutter plugin that enables running Large Language Models (LLMs) locally on Android and iOS using [llamadart](https://pub.dev/packages/llamadart) (`^0.8.22`). It provides real-time streaming inference, performance metrics, cloud AI chat providers, and a fully local RAG pipeline.

Requires Flutter `>=3.38.0` (llamadart 0.8.x) and iOS `16.4`+.

## Commands

```bash
# Install dependencies
flutter pub get

# Run all tests
flutter test

# Run a specific test file
flutter test test/src/core/performance_metrics_test.dart

# Run tests with coverage
flutter test --coverage

# Lint
flutter analyze

# Format
dart format lib/ test/ example/lib/

# Run the example app
cd example && flutter pub get && flutter run
```

## Architecture

### Public API

`lib/mt_llmkit.dart` is the single export file. It re-exports everything from `src/` plus, from `llamadart`: the content parts (`LlamaContentPart`, `LlamaTextContent`, `LlamaImageContent`, `LlamaAudioContent`), `GpuBackend`, `LoraAdapterConfig`, `GenerationGrammarTrigger`, the config knobs surfaced through `LlmConfig` (`FlashAttention`, `KvCacheType`, `ThinkingBudget`, `SpeculativeDecodingConfig`), and the `LlamaException` hierarchy.

### Class Hierarchy

```
LlmInterface (abstract interface)               ← lib/src/core/llm_interface.dart
  └─ LocalModel                                 ← lib/src/gguf/local_model.dart (public API)
       ├─ LlmModelIsolated  → Dart Isolate      ← lib/src/models/llm_model_isolated.dart
       └─ LlmModelStandard  → in-process        ← lib/src/models/llm_model_standard.dart
```

`LocalModel` selects the backend via `ModelBackend` enum (`isolate` | `inProcess`). Both share the same lifecycle: `loadModel(path)` → generate → `dispose()`.

### Three Generation Methods

1. `sendPrompt(prompt) → Stream<String>` — raw token stream
2. `sendPromptComplete(prompt) → Future<String>` — full response as a single string
3. `sendPromptStream(prompt) → Stream<StreamingChunk>` — **recommended**: live streaming + real-time metrics

All three take the same optional arguments:

- `systemPrompt` — sent as a real `system` message
- `attachments` (`List<LlamaContentPart>`) — `LlamaImageContent` for vision models (needs `mmprojPath`), `LlamaAudioContent` for audio-capable ones
- `overrides` (`GenerationOverrides`) — sampling for this request only, without reloading the model

`StreamingChunk` carries: `text`, `thinking` (reasoning channel — never mixed into `text`),
`metrics` (`PerformanceMetrics`), an `isFinal` flag, `finishReason` (set on the final chunk —
`'stop'` vs `'length'`; `isTruncated` is the shorthand for hitting the `nPredict` budget), and
`toolCalls` (final chunk only).

### Configuration

`LlmConfig` is an immutable config object (`lib/src/core/llm_config.dart`). Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `temp` | 0.72 | Sampling temperature |
| `nGpuLayers` | auto (999) | GPU layers offloaded; `0` forces CPU-only |
| `nCtx` | 8192 | Context window in tokens |
| `nBatch` | auto | `min(nCtx, 2048)`, resolved by llama.cpp |
| `nThreads` | auto | llama.cpp's thread heuristic |
| `topK` | 64 | Top-K sampling |
| `topP` | 0.95 | Top-P sampling |
| `penaltyRepeat` | 1.1 | Repetition penalty |
| `presencePenalty` | 0.0 | Presence penalty |
| `mmprojPath` | null | Path to mmproj GGUF for vision |
| `chatTemplate` | null | Custom Jinja chat template string; `null` uses the model's embedded template |
| `enableThinking` | true | Whether reasoning models may emit a thinking block |
| `thinkingBudget` | null | Caps tokens per reasoning block (`ThinkingBudget`) |
| `flashAttention` | auto | Required by a quantized KV cache |
| `cacheTypeK` / `cacheTypeV` | f16 | `q8_0` halves / `q4_0` quarters KV memory — the largest memory win on a phone |
| `speculativeDecoding` | null | `SpeculativeDecodingConfig` (n-gram, MTP, draft model, …) |

`LlmConfig.validate()` rejects combinations llama.cpp refuses (currently: a quantized KV cache
with flash attention disabled) and is called by both backends before loading, so the failure is
readable instead of surfacing from inside the worker isolate.

`lib/src/core/model_params_builder.dart` maps `LlmConfig` onto llamadart's `ModelParams` /
`GenerationParams` and builds the chat messages — one place to wire a new knob.

### Prompt Format

Override the model's built-in chat template via `LlmConfig.chatTemplate` (a raw Jinja/GGUF template string). When `null` the model file's embedded template is used automatically — no manual format selection required.

### System messages

Pass `systemPrompt` to any `sendPrompt*` call and it becomes a real `LlamaChatRole.system`
message, so the model's chat template places it where it expects instructions.
`RagPipeline` uses this for its instructions (`RagPipeline.defaultSystemPrompt`); its
`promptTemplate` now holds only `{context}` / `{question}`. Override either through
`RagEngine(systemPrompt:, promptTemplate:)`.

### Performance Metrics

`PerformanceMetrics` (`lib/src/core/performance_metrics.dart`) tracks `tokensGenerated`,
`durationMs`, `tokensPerSecond`, `msPerToken`, plus `promptTokens`, `promptEvalMs`, `evalMs`
and an `isExact` flag.

Metrics emitted **while streaming** are estimates (`isExact == false`): they count stream
chunks, and llamadart batches chunks according to `streamBatchTokenThreshold` /
`streamBatchByteThreshold`, so a chunk is not necessarily a token. The **final** chunk carries
llama.cpp's own counters, read through `LlamaEngine.getPerformanceContext()` — see
`lib/src/core/backend_perf.dart`. `tokensPerSecond` is then decode throughput and excludes
prompt ingestion.

### Tool calling and structured output

Declare tools per request through `GenerationOverrides.tools` (`LlmTool` — name, description,
`ToolParam` schema), with `toolChoice`, `parallelToolCalls` and `responseFormat`. Declaring tools
constrains decoding to their schema and lets llamadart parse the call out of whatever envelope
the model's chat template uses.

`LlmTool` deliberately has **no handler**: generation runs in a worker isolate, so your code
cannot (and should not) execute there. Completed calls arrive as data on
`StreamingChunk.toolCalls` (final chunk only) — llamadart streams a call's name and arguments as
fragments, and `ToolCallAccumulator` (`lib/src/core/tools.dart`) reassembles them so callers do
not have to.

### Engine operations beyond generation

`LocalModel` also exposes, on both backends:

- **Tokenization** — `tokenize`, `detokenize`, `countTokens`, `contextSize`, `metadata`
- **KV-cache state** — `supportsStatePersistence`, `saveState(path, tokens:)`,
  `loadState(path)`: resume a long conversation without paying prompt ingestion again
- **Runtime LoRA** — `setLora`, `removeLora`, `clearLoras` (previously load-time only)
- **Diagnostics** — `diagnostics()` → `ModelDiagnostics`: which backend actually won, how many
  layers really reached the GPU, GGUF file type, vision/audio support, VRAM

The isolate backend reaches these through a `call` message; both backends share the dispatch in
`lib/src/core/engine_rpc.dart`, so a new operation is wired once.

Not wired: `loadModelSource` / `loadModelFromUrl` and llamadart's download manager. Model
downloading needs app-level cache-directory decisions (and apps typically already have a
downloader with resume and notifications), so `loadModel` still takes a local path.

### Errors

llamadart's typed exceptions survive the worker isolate boundary: workers encode them with
`encodeError` and the main isolate rebuilds the same class with `decodeError`
(`lib/src/core/llm_errors.dart`). So `LlamaBackendInitializationException` ("GPU backend
failed, retry on CPU") stays distinguishable from `LlamaModelException` ("model file is
corrupt") and `LlamaUnsupportedException` (e.g. aLoRA adapters, hard-failing since llamadart
0.8.22). The hierarchy is re-exported from `lib/mt_llmkit.dart`.

### Teardown

`dispose()` is `Future<void>` everywhere (`LlmInterface`, `LocalModel`, both backends,
`RagEngine`) and **must be awaited**. Isolate-backed implementations wait for the worker to
acknowledge that llama.cpp released its native handles (5 s timeout) before killing the
isolate; `LocalModel.loadModel` awaits the previous model's teardown before loading the next.

### Cloud AI Providers

`AIChatProvider` interface in `lib/src/api/ai_chat_provider.dart`. Implementations:

- `OpenAIChatProvider`, `GeminiChatProvider`, `ClaudeChatProvider`, `MistralChatProvider`

Use `AIChatProviderFactory.create(AIChatProviderType)` to instantiate. Exceptions are in `lib/src/api/chat_exceptions.dart`.

### RAG Pipeline

`RagEngine` (`lib/src/rag/rag_engine.dart`) uses two standalone components:

- `LlamaEmbeddingProvider` — standalone embed isolate (CPU-only)
- `LocalModel` — generation isolate (GPU configured)

Both embedding paths use llamadart's native batch call (`embedBatch`) instead of one isolate
round-trip per chunk, size chunks with the model's tokenizer instead of a character count
(`lib/src/rag/embeddings/embed_worker_ops.dart` — a fixed 2000-character cut is not 512 tokens,
so long chunks used to overrun the encoder), and read the embedding width from GGUF metadata
rather than spending an inference pass on a probe string. `RagPipeline.embedBatchSize` (default
16) trades ingestion speed against progress granularity.

`RagPipeline` calls `sendPromptStream(augmentedPrompt, systemPrompt: systemPrompt)`, so the
instructions travel as a real `system` message and only the context + question go into the user
turn.

### Native Libraries

- Android: `.so` files downloaded and bundled by llamadart's Dart build hook (this repo ships no
  `jniLibs`); ABIs limited to `arm64-v8a` and `x86_64` in `android/build.gradle`
- iOS: llama.cpp XCFramework linked through **Swift Package Manager** by the companion package
  `llamadart_llama_cpp_flutter`, which the **app** must declare in its own `pubspec.yaml`
  (llamadart's `hook/build.dart` reads the consumer pubspec and then emits its asset as
  `LookupInProcess()` instead of a bundled dylib)
- Without that companion package llamadart falls back to native assets, and Flutter wraps the
  dylib in a `llamadart.framework` whose `Info.plist` hardcodes `MinimumOSVersion 13.0`
  (flutter/flutter#145104) while the binary needs 16.4 → App Store rejects with **ITMS-90208**.
  A correct release build contains **no** `llamadart.framework`, only `llama.framework` +
  `llamadart-llama-cpp-flutter.framework`. Verify only after `flutter clean` — Flutter never prunes
  `Runner.app/Frameworks`, so a stale `llamadart.framework` from a pre-migration build survives
  incremental builds and would still be shipped.

### iOS plugin packaging

Dual: Swift Package Manager (`ios/mt_llmkit/Package.swift`, product `mt-llmkit`) **and** CocoaPods
(`ios/mt_llmkit.podspec`) over the same sources in `ios/mt_llmkit/Sources/mt_llmkit/`. Both declare
iOS 16.4.

Gotcha: for the **example app** Flutter adds the plugin as a local SwiftPM package override, and
the override identity is the basename of the plugin's root directory. The checkout directory must
therefore be named `mt_llmkit`; from a directory named otherwise, `flutter build ios` fails with
`unable to override package 'mt_llmkit' because its identity '<dir>' doesn't match override's
identity (directory name) 'mt_llmkit'`. Consuming apps are unaffected (no override is added).

## Test Infrastructure

Tests live in:

```
test/
├── helpers/test_helpers.dart          ← TestHelpers, TestConfigBuilder, shared fixtures
├── src/api/                           ← cloud provider tests
├── src/core/                          ← LlmConfig, PerformanceMetrics tests
└── src/models/                        ← LlmModelBase/Isolated/Standard tests
```

`flutter analyze` must show **0 issues**; `flutter test` must pass all tests.
