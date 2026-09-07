## Unreleased

### Changed

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
