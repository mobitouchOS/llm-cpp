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

`lib/mt_llmkit.dart` is the single export file. It re-exports everything from `src/` plus `LlamaImageContent`, `LlamaTextContent`, `LlamaContentPart`, `GpuBackend`, `LoraAdapterConfig`, `GenerationGrammarTrigger` from `llamadart`.

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

All three accept an optional `images` parameter for vision models.

`StreamingChunk` carries: `text`, `metrics` (`PerformanceMetrics`), and `isFinal` flag.

### Configuration

`LlmConfig` is an immutable config object (`lib/src/core/llm_config.dart`). Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `temp` | 0.72 | Sampling temperature |
| `nGpuLayers` | 64 | GPU layers offloaded |
| `nCtx` | 8192 | Context window in tokens |
| `nBatch` | 4096 | Batch size |
| `nThreads` | 6 | CPU threads |
| `topK` | 64 | Top-K sampling |
| `topP` | 0.95 | Top-P sampling |
| `penaltyRepeat` | 1.1 | Repetition penalty |
| `mmprojPath` | null | Path to mmproj GGUF for vision |
| `chatTemplate` | null | Custom Jinja chat template string; `null` uses the model's embedded template |

### Prompt Format

Override the model's built-in chat template via `LlmConfig.chatTemplate` (a raw Jinja/GGUF template string). When `null` the model file's embedded template is used automatically — no manual format selection required.

### System message support

`LlmModelIsolated` decodes `\x01SYS\x01`/`\x01USR\x01` markers in `sendPrompt*` calls to pass proper `[system, user]` messages to the underlying engine. `RagPipeline` uses this to inject the RAG system prompt.

### Performance Metrics

`PerformanceMetrics` (`lib/src/core/performance_metrics.dart`) tracks `tokensGenerated`, `durationMs`, `tokensPerSecond`, `msPerToken`. Counts are exact: each llamadart callback emits one token, incremented via `+= 1` in the streaming loop.

### Cloud AI Providers

`AIChatProvider` interface in `lib/src/api/ai_chat_provider.dart`. Implementations:

- `OpenAIChatProvider`, `GeminiChatProvider`, `ClaudeChatProvider`, `MistralChatProvider`

Use `AIChatProviderFactory.create(AIChatProviderType)` to instantiate. Exceptions are in `lib/src/api/chat_exceptions.dart`.

### RAG Pipeline

`RagEngine` (`lib/src/rag/rag_engine.dart`) uses two standalone components:

- `LlamaEmbeddingProvider` — standalone embed isolate (CPU-only)
- `LocalModel` — generation isolate (GPU configured)

`RagPipeline` injects system + user content as `\x01SYS\x01{system}\x01USR\x01{question}` via `sendPromptStream`.

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
