# mt_llmkit

![Flutter](https://img.shields.io/badge/Flutter-%3E%3D3.38-02569B?logo=flutter)
![Dart](https://img.shields.io/badge/Dart-%3E%3D3.10.7-0175C2?logo=dart)
![Platform](https://img.shields.io/badge/platform-iOS%20%7C%20Android-lightgrey)
![Stability](https://img.shields.io/badge/stability-beta-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A Flutter plugin for running Large Language Models (LLMs) locally on Android and iOS using [llamadart](https://pub.dev/packages/llamadart) (which wraps llama.cpp). Also provides a unified interface for cloud AI chat providers (OpenAI, Gemini, Claude, Mistral) and a fully local RAG (Retrieval-Augmented Generation) pipeline.

---

## Table of Contents

1. [Installation](#installation)
   - [Platform requirements](#platform-requirements)
   - [iOS setup](#ios-setup)
2. [Local LLM Inference (GGUF)](#local-llm-inference-gguf)
   - [Quick start](#quick-start)
   - [Backends](#backends)
   - [Model lifecycle](#model-lifecycle)
   - [Configuration](#configuration)
   - [Generation methods](#generation-methods)
   - [Conversations](#conversations)
   - [Prompt format](#prompt-format)
   - [Performance metrics](#performance-metrics)
3. [Vision (Multimodal)](#vision-multimodal)
   - [Quick start](#quick-start-vision)
   - [Supported models](#supported-models)
   - [Image input](#image-input)
   - [Generation methods](#generation-methods-1)
4. [Cloud API Providers](#cloud-api-providers)
   - [Supported providers](#supported-providers)
   - [Basic usage](#basic-usage)
   - [Multi-turn conversations](#multi-turn-conversations)
   - [Streaming](#streaming)
   - [Provider-specific config](#provider-specific-config)
   - [Error handling](#error-handling)
5. [Local RAG Pipeline](#local-rag-pipeline)
   - [How it works](#how-it-works)
   - [Quick start](#quick-start-1)
   - [Document ingestion](#document-ingestion)
   - [Querying](#querying)
   - [Index persistence](#index-persistence)
   - [Advanced: custom prompt template](#advanced-custom-prompt-template)

---

## Installation

Add to your `pubspec.yaml`:

```yaml
dependencies:
  mt_llmkit: ^0.0.1-beta.1

  # iOS/macOS only: links llama.cpp's Apple XCFramework through Swift Package Manager.
  # Must be declared here, in the *app's* pubspec — llamadart's build hook reads it.
  llamadart_llama_cpp_flutter: ^0.0.17
```

Then run:

```bash
flutter pub get
```

Import the library:

```dart
import 'package:mt_llmkit/mt_llmkit.dart';
```

### Platform requirements

| | Minimum |
|---|---|
| iOS | **16.4** (`IPHONEOS_DEPLOYMENT_TARGET`, and `platform :ios, '16.4'` if you use CocoaPods) |
| Flutter | **3.38.0** (required by llamadart 0.8.x) |
| Android | arm64-v8a or x86_64 |

The iOS floor comes from llamadart's prebuilt llama.cpp runtime, which is built for iOS 16.4.
`ios/mt_llmkit.podspec` and `ios/mt_llmkit/Package.swift` both declare it, so the incompatibility
is reported at integration time rather than at runtime.

### iOS setup

On iOS the native runtime is linked as an XCFramework through **Swift Package Manager**, which is
enabled by default in Flutter 3.38+. Two things are required in your app:

1. `llamadart_llama_cpp_flutter` in the app's `pubspec.yaml` (see above).
2. Swift Package Manager **not** disabled — i.e. no
   `flutter: config: enable-swift-package-manager: false` in the app's `pubspec.yaml`.
   CocoaPods can stay: apps mixing SwiftPM packages with pods (Firebase and friends) are supported.

If either is missing, llamadart falls back to shipping its dylib as a Flutter *native asset*.
Flutter then wraps it in a framework whose `Info.plist` declares a hardcoded
`MinimumOSVersion` of `13.0` ([flutter/flutter#145104](https://github.com/flutter/flutter/issues/145104))
while the binary requires 16.4, and App Store validation rejects the upload:

```
ITMS-90208: Invalid Bundle - The bundle Runner.app/Frameworks/llamadart.framework
does not support the minimum OS Version specified in the Info.plist.
```

When migrating an existing app, run `flutter clean` first. Flutter does not prune
`Runner.app/Frameworks`, so a stale `llamadart.framework` from an earlier build stays in the bundle
— and would be uploaded — even though the new build no longer produces one.

You can confirm a correct setup on a release build — there must be **no** `llamadart.framework`:

```bash
ls build/ios/iphoneos/Runner.app/Frameworks
# App.framework  Flutter.framework  llama.framework
# llamadart-llama-cpp-flutter.framework  objective_c.framework
```

---

## Local LLM Inference (GGUF)

Run quantized GGUF models entirely on-device — no internet connection required.

### Quick start

```dart
final model = LocalModel(
  config: LlmConfig(temp: 0.7, nCtx: 2048, nGpuLayers: 4),
);

await model.loadModel('/path/to/model.gguf');

// Stream tokens as they are generated
model.sendPrompt('What is Flutter?').listen((token) {
  stdout.write(token);
});

model.dispose();
```

### Backends

`LocalModel` supports two backends controlled by the `backend` parameter:

| Backend | Class | When to use |
|---|---|---|
| `ModelBackend.isolate` *(default)* | `LlmModelIsolated` | Production. Runs in a Dart Isolate — no UI jank. Required when loading multiple models (e.g. RAG). |
| `ModelBackend.inProcess` | `LlmModelStandard` | Lighter startup cost. Runs on the calling thread. |

```dart
// Isolate backend (default)
final model = LocalModel(backend: ModelBackend.isolate);

// In-process backend
final model = LocalModel(backend: ModelBackend.inProcess);
```

### Model lifecycle

```dart
await model.loadModel(pathA);
await model.unload();          // frees the model, keeps the worker + backend
await model.loadModel(pathB);  // no isolate spawn, no llama.cpp re-init

await model.clean();           // drop the reused prompt prefix and its KV cache
await model.dispose();         // tear everything down
```

`loadModel` on a model that already has one loaded does the `unload` for you, so switching
models never respawns the worker isolate.

`clean()` works on **both** backends. llama.cpp has no call that forgets the prompt cache on the
spot, so the next generation runs with `reusePromptPrefix: false`, which clears context memory
and the cached prompt tokens before ingesting — nothing is recomputed until then, so calling it
is free.

### Configuration

All parameters are optional; sensible defaults are applied automatically.

```dart
final config = LlmConfig(
  nGpuLayers: 4,    // GPU layers offloaded (default: 64)
  nCtx: 2048,       // context window in tokens (default: 8192)
  nBatch: 512,      // batch size (default: 4096)
  nPredict: 1024,   // max tokens to generate (default: 8192)
  nThreads: 4,      // CPU threads (default: 6)
  temp: 0.7,        // temperature (default: 0.72)
  topK: 40,         // top-K sampling (default: 64)
  topP: 0.9,        // top-P sampling (default: 0.95)
  penaltyRepeat: 1.1, // repetition penalty (default: 1.1)
);
```

### Generation methods

Three methods are available on `LocalModel` (and any `LlmInterface` implementation):

| Method | Return type | Description |
|---|---|---|
| `sendPrompt(prompt)` | `Stream<String>` | Raw token stream. Lowest overhead. |
| `sendPromptComplete(prompt)` | `Future<String>` | Waits for the full response and returns it as a single string. |
| `sendPromptStream(prompt)` | `Stream<StreamingChunk>` | **Recommended.** Token stream with live performance metrics. |
| `sendPromptResult(prompt)` | `Future<GenerationResult>` | Everything the generation produced: text, reasoning, tool calls, finish reason, metrics. |
| `sendPromptStructured(prompt, output:)` | `Future<LlmStructuredResult<T>>` | JSON constrained to a schema, decoded into `T`. |

```dart
// 1. Raw token stream
model.sendPrompt('Hello').listen(stdout.write);

// 2. Full response at once
final response = await model.sendPromptComplete('Hello');
print(response);

// 3. Streaming with live metrics (recommended)
model.sendPromptStream('Hello').listen((chunk) {
  stdout.write(chunk.text);

  if (chunk.isFinal && chunk.metrics != null) {
    final m = chunk.metrics!;
    print('\n--- ${m.tokensGenerated} tokens, ${m.tokensPerSecond.toStringAsFixed(1)} t/s ---');
  }
});
```

`StreamingChunk` fields:

| Field | Type | Description |
|---|---|---|
| `text` | `String` | The generated text fragment. |
| `thinking` | `String?` | Reasoning, on its own chunks. Never mixed into `text`. |
| `isFinal` | `bool` | `true` on the last chunk of the response. |
| `finishReason` | `String?` | Final chunk only: `'stop'` or `'length'`. `isTruncated` is the shorthand. |
| `toolCalls` | `List<LlmToolCall>` | Final chunk only, when tools were declared. |
| `metrics` | `PerformanceMetrics?` | Available on every chunk; exact on the final one (`isExact`). |

`PerformanceMetrics` fields: `tokensGenerated`, `durationMs`, `tokensPerSecond`, `msPerToken`,
`promptTokens`, `promptEvalMs`, `evalMs`, `isExact`.

### Conversations

`sendPrompt*` is stateless — every call is a fresh exchange. For a chat, open a conversation:

```dart
final chat = await model.startConversation(
  systemPrompt: 'You are a concise assistant.',
);

await for (final chunk in chat.send('What is the capital of Poland?')) {
  stdout.write(chunk.text);
}

// The model sees the previous turn.
final follow = await chat.sendComplete('And its population?');
print(follow.text);

await chat.close();
```

History lives with the model and is trimmed as it approaches the context window. That trim is
**permanent** — the oldest turns are gone — so it is reported rather than silent:

```dart
chat.trims.listen((event) {
  print('Dropped ${event.dropped.length} older messages to make room.');
});
```

Pass `overflowPolicy: ContextOverflowPolicy.fail` to get a `LlmContextOverflowException` instead
of an answer built on a prompt the model could not fully see.

`chat.history()` returns `List<LlmChatMessage>` (JSON-serializable) and `chat.restore(...)` puts
it back, so a conversation can be persisted and resumed.

#### Tool calling

```dart
final tools = [
  const LlmTool(
    name: 'get_weather',
    description: 'Current weather for a city',
    parameters: [],
  ),
];

var turn = await chat.sendComplete(
  'What is the weather in Kraków?',
  overrides: GenerationOverrides(tools: tools),
);

while (turn.needsToolResults) {
  final results = [
    for (final call in turn.toolCalls)
      ToolResult.forCall(call, await runTool(call)),
  ];
  turn = await chat.submitToolResultsComplete(results);
}

print(turn.text);
```

Tools are declarations only — `LlmTool` carries no handler. Generation runs in a worker isolate,
so your tool code runs where your app's state and plugins actually are.

#### Structured output

```dart
final output = LlmStructuredOutput.jsonSchema(
  schema: {
    'type': 'object',
    'properties': {'city': {'type': 'string'}},
    'required': ['city'],
  },
  decoder: (json) => json['city'] as String,
);

final result = await model.sendPromptStructured(
  'Which city is the capital of Poland?',
  output: output,
);
print(result.value);
```

The schema constrains decoding, so the model cannot emit anything that would not parse. A schema
llamadart cannot turn into a grammar is rejected when you build the `LlmStructuredOutput`, not
mid-generation.

### Prompt format

Override the model's built-in chat template by passing a raw GGUF/Jinja template string via `LlmConfig.chatTemplate`. When `chatTemplate` is `null` (the default), the template embedded in the model file is used automatically.

```dart
final config = LlmConfig(
  chatTemplate: '<|user|>\n{prompt}<|end|>\n<|assistant|>\n', // custom override
);
```

### Performance metrics

`PerformanceMetrics` is updated incrementally with every `StreamingChunk`:

```dart
model.sendPromptStream('Explain Dart isolates in detail.').listen((chunk) {
  stdout.write(chunk.text);

  if (chunk.metrics != null) {
    final m = chunk.metrics!;
    // Update UI progress indicator
    print('${m.tokensGenerated} tokens | ${m.tokensPerSecond.toStringAsFixed(2)} t/s');
  }
});
```
---

## Vision (Multimodal)

`LocalModel` supports multimodal vision models (LLaVA, Gemma 3, Qwen VL, SmolVLM, etc.) that can analyse images alongside a text prompt. Vision requires two GGUF files: the main language model and a **multimodal projector** (`mmproj-*.gguf`).

### Quick start (vision) {#quick-start-vision}

```dart
final model = LocalModel(
  config: LlmConfig(
    mmprojPath: '/path/to/mmproj-model-f16-4B.gguf',
    nGpuLayers: 4,
    nCtx: 4096,
    nPredict: 512,
    temp: 0.3,
  ),
);

await model.loadModel('/path/to/gemma-3-4b-it-q4_0.gguf');

final image = LlamaImageContent(path: '/path/to/photo.jpg');

model.sendPromptStream(
  'Describe what you see in this image. <image>',
  images: [image],
).listen((chunk) {
  stdout.write(chunk.text);

  if (chunk.isFinal && chunk.metrics != null) {
    print('\n--- ${chunk.metrics!.tokensPerSecond.toStringAsFixed(1)} t/s ---');
  }
});
```

> **Important:** The prompt must contain one `<image>` placeholder per image passed in the list.

### Supported models

Any vision GGUF model that uses the `libmtmd` multimodal projection layer is supported. Tested models:

| Model | Notes |
|---|---|
| Gemma 3 (4B, 12B, 27B) | Recommended. Good accuracy, available in Q4 quantisation. |
| Qwen 2.5 VL | Strong OCR and document understanding. |
| LLaVA 1.5 / 1.6 | Classic CLIP-based architecture. |
| SmolVLM | Compact, fast, good for mobile devices. |

Each model has a corresponding `mmproj-*.gguf` file available on Hugging Face alongside the main model.

### Image input

`LlamaImageContent` is created by providing the image file path:

```dart
final image = LlamaImageContent(path: '/path/to/photo.jpg');
```

Supported formats: JPEG, PNG, and any format supported by the underlying `libmtmd` library.

### Generation methods (vision) {#generation-methods-1}

All three standard generation methods accept an optional `images` parameter:

| Method | Return type | Description |
|---|---|---|
| `sendPrompt(prompt, images: images)` | `Stream<String>` | Raw token stream. |
| `sendPromptComplete(prompt, images: images)` | `Future<String>` | Full response as a single string. |
| `sendPromptStream(prompt, images: images)` | `Stream<StreamingChunk>` | **Recommended.** Streaming with live performance metrics. |

All three methods throw `UnsupportedError` if `LlmConfig.mmprojPath` was not set.

```dart
// Full response at once
final response = await model.sendPromptComplete(
  'What objects are visible in this photo? <image>',
  images: [LlamaImageContent(path: '/path/to/photo.jpg')],
);
print(response);
```

---

## Cloud API Providers

`mt_llmkit` includes a unified `AIChatProvider` interface for four cloud LLM providers. All providers share the same API surface, making it easy to swap backends.

### Supported providers

| Provider | Enum value | Default model |
|---|---|---|
| OpenAI | `AIChatProviderType.openai` | `gpt-4o-mini` |
| Google Gemini | `AIChatProviderType.gemini` | `gemini-1.5-flash` |
| Anthropic Claude | `AIChatProviderType.claude` | `claude-haiku-4-5-20251001` |
| Mistral AI | `AIChatProviderType.mistral` | `mistral-small-latest` |

### Basic usage

Use `AIChatProviderFactory` to create a provider without importing the concrete class:

```dart
// Create and initialize in one step
final provider = await AIChatProviderFactory.createAndInitialize(
  AIChatProviderType.openai,
  {'apiKey': 'sk-...'},
);

final response = await provider.sendMessage('What is Flutter?');
print(response.message.content);
print('Tokens used: ${response.inputTokens} in / ${response.outputTokens} out');

await provider.dispose();
```

Or manage the lifecycle manually:

```dart
final provider = AIChatProviderFactory.create(AIChatProviderType.gemini);
await provider.initialize({'apiKey': 'AIza...'});

final response = await provider.sendMessage('Hello!');
print(response.message.content);

await provider.dispose();
```

### Multi-turn conversations

Build conversation history with `ChatMessage`:

```dart
final history = <ChatMessage>[
  ChatMessage.system('You are a concise assistant. Reply in three sentences max.'),
  ChatMessage.user('What is a Dart isolate?'),
  ChatMessage.assistant('A Dart isolate is an independent thread of execution...'),
];

// Continue the conversation
final r = await provider.sendMessage('Give me a code example.', history: history);
print(r.message.content);

// Append the reply to keep history growing
history.add(r.message);
```

For full control, pass a complete message list directly:

```dart
final response = await provider.sendChatMessages([
  ChatMessage.system('You are a poet.'),
  ChatMessage.user('Write a haiku about Flutter.'),
]);
print(response.message.content);
```

### Streaming

All providers support token streaming via `sendMessageStream`:

```dart
await for (final token in provider.sendMessageStream('Tell me a story.')) {
  stdout.write(token);
}
```

With conversation history:

```dart
final history = [ChatMessage.system('Reply only in Spanish.')];

await for (final token in provider.sendMessageStream('Hello!', history: history)) {
  stdout.write(token);
}
```

### Provider-specific config

#### OpenAI

```dart
await provider.initialize({
  'apiKey': 'sk-...',
  'model': 'gpt-4o',               // optional, default: gpt-4o-mini
  'baseUrl': 'https://...',        // optional, for Azure OpenAI or proxies
});
```

#### Google Gemini

```dart
await provider.initialize({
  'apiKey': 'AIza...',
  'model': 'gemini-1.5-pro',       // optional
});
```

#### Anthropic Claude

```dart
await provider.initialize({
  'apiKey': 'sk-ant-...',
  'model': 'claude-opus-4-6',      // optional
});
```

#### Mistral AI

```dart
await provider.initialize({
  'apiKey': '...',
  'model': 'mistral-large-latest', // optional
});
```

### Error handling

All providers throw typed exceptions from `chat_exceptions.dart`:

| Exception | Cause |
|---|---|
| `APIKeyException` | Invalid or missing API key (HTTP 401/403) |
| `NetworkException` | Transport error (timeout, DNS, connection reset) |
| `RateLimitException` | Quota exceeded (HTTP 429); contains `retryAfter` |
| `AIChatException` | Base class for any other API error |

Network and rate-limit errors are **automatically retried** up to 3 times with exponential back-off.

```dart
try {
  final response = await provider.sendMessage('Hello');
  print(response.message.content);
} on APIKeyException catch (e) {
  print('Check your API key: $e');
} on RateLimitException catch (e) {
  print('Rate limited. Retry after ${e.retryAfter?.inSeconds}s');
} on AIChatException catch (e) {
  print('API error: $e');
}
```

---

## Local RAG Pipeline

`RagEngine` provides a fully on-device Retrieval-Augmented Generation pipeline. Documents are chunked, embedded with a local embedding model, stored in an in-memory vector store, and retrieved at query time to ground the generation model's response.

### How it works

```
Ingestion:  Document → TextChunker → chunks → EmbeddingModel → VectorStore
Query:      question → EmbeddingModel → VectorStore.search() → prompt + context → GenerationModel
```

Both the embedding model and the generation model run inside a **single worker isolate**, avoiding threading issues that arise when multiple native model instances share global state — this is handled transparently by llamadart.

### Quick start

You need two GGUF models:
- A **generation model** (e.g. Llama 3, Mistral) for producing answers.
- An **embedding model** (e.g. nomic-embed-text) for vectorising text.

```dart
final rag = RagEngine(
  genModelPath: '/path/to/llama.gguf',
  embedModelPath: '/path/to/nomic-embed.gguf',
  genConfig: LlmConfig(temp: 0.3, nCtx: 4096, nGpuLayers: 4),
);

await rag.initialize();
```

### Document ingestion

Create a `Document` from text or PDF content and ingest it into the vector store:

```dart
// From plain text
final doc = Document.fromText(
  'Flutter is Google\'s UI toolkit for building natively compiled applications...',
  source: 'flutter_intro.txt',
);

// From extracted PDF text (use a PDF parser to extract the string first)
final pdfDoc = Document.fromPdf(
  extractedText,
  source: '/path/to/manual.pdf',
  pageCount: 42,
);

// Ingest — stream progress for UI updates
await for (final progress in rag.ingestDocument(doc)) {
  print('${progress.embeddedChunks}/${progress.totalChunks} — ${progress.currentPreview}');
  // progress.fraction gives 0.0–1.0 for a progress bar
}

print('Indexed: ${rag.indexedSize} chunks across ${rag.documentIds.length} documents');
```

Manage the index:

```dart
// Remove a single document (all its chunks)
await rag.removeDocument(doc.id);

// Clear everything
await rag.clearIndex();
```

### Querying

```dart
// Stream the generated answer
await for (final chunk in rag.query('What is Flutter?')) {
  stdout.write(chunk.text);

  if (chunk.isFinal && chunk.metrics != null) {
    print('\n${chunk.metrics!.tokensPerSecond.toStringAsFixed(1)} t/s');
  }
}
```

Optional query parameters:

```dart
rag.query(
  'Explain the rendering pipeline.',
  topK: 3,              // number of context chunks (default: 5)
  minSimilarity: 0.35,  // minimum cosine similarity 0.0–1.0 (default: 0.25)
)
```

Retrieve relevant chunks without generating an answer (useful for inspection):

```dart
final results = await rag.findRelevant('rendering pipeline', topK: 3);
for (final r in results) {
  print('${(r.similarity * 100).toStringAsFixed(0)}% — ${r.chunk.text.substring(0, 80)}');
}
```

### Index persistence

Pass `indexPath` to automatically save and restore the vector index between sessions:

```dart
final dir = await getApplicationDocumentsDirectory();

final rag = RagEngine(
  genModelPath: '/path/to/llama.gguf',
  embedModelPath: '/path/to/nomic-embed.gguf',
  indexPath: '${dir.path}/rag_index.json',  // auto-save on every change
);

await rag.initialize(); // loads existing index if the file exists
```

When `indexPath` is `null`, the store is in-memory only and data is lost when the app restarts.

### Advanced: custom prompt template

The default template instructs the model to answer only from the provided context. Override it if you need different behaviour:

```dart
final rag = RagEngine(
  genModelPath: '...',
  embedModelPath: '...',
  promptTemplate:
    'You are a helpful assistant. Use the context below to answer.\n\n'
    'CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:',
);
```

The template must contain `{context}` and `{question}` placeholders.

### Cleanup

```dart
rag.dispose(); // releases the worker isolate and both models; safe to call twice
```
