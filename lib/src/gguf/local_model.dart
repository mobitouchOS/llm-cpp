// lib/src/gguf/local_model.dart

import 'package:llamadart/llamadart.dart' show LlamaContentPart;

import '../core/generation_overrides.dart';
import '../core/llm_config.dart';
import '../core/llm_interface.dart';
import '../core/model_diagnostics.dart';
import '../core/structured_output.dart';
import '../core/streaming_result.dart';
import '../models/llm_model_base.dart';
import '../models/llm_model_isolated.dart';
import '../models/llm_model_standard.dart';

/// Backend used by [LocalModel].
///
/// - [isolate]: runs in a Dart Isolate — **recommended**, no UI blocking.
/// - [inProcess]: runs on the calling thread — lighter startup.
enum ModelBackend { isolate, inProcess }

/// Plugin for running local GGUF models.
///
/// Implements [LlmInterface] and selects the appropriate backend internally:
/// [LlmModelIsolated] (isolate, default) or [LlmModelStandard] (in-process).
class LocalModel implements LlmInterface {
  final ModelBackend backend;
  final LlmConfig config;

  LlmModelBase? _model;

  LocalModel({
    this.backend = ModelBackend.isolate,
    this.config = const LlmConfig(),
  });

  @override
  Future<void> loadModel(String localPath) async {
    final existing = _model;
    if (existing != null && !existing.isDisposed) {
      // Swap the model inside the live backend: the worker isolate and the
      // llama.cpp backend stay up, and the previous model's native handles are
      // freed before the next one starts allocating.
      await existing.unload();
      await existing.loadModel(localPath);
      return;
    }

    _model = backend == ModelBackend.isolate
        ? LlmModelIsolated(config)
        : LlmModelStandard(config);
    await _model!.loadModel(localPath);
  }

  /// Frees the model and its context but keeps the backend — and, on
  /// [ModelBackend.isolate], the worker isolate — alive, so the next
  /// [loadModel] skips isolate spawn and llama.cpp initialization.
  Future<void> unload() async {
    final model = _model;
    if (model == null || model.isDisposed) return;
    await model.unload();
  }

  @override
  Future<void> dispose() async {
    final model = _model;
    _model = null;
    await model?.dispose();
  }

  @override
  Stream<String> sendPrompt(
    String prompt, {
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) {
    _ensureInitialized();
    return _model!.sendPrompt(
      prompt,
      systemPrompt: systemPrompt,
      attachments: attachments,
      overrides: overrides,
    );
  }

  @override
  Future<String> sendPromptComplete(
    String prompt, {
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) {
    _ensureInitialized();
    return _model!.sendPromptComplete(
      prompt,
      systemPrompt: systemPrompt,
      attachments: attachments,
      overrides: overrides,
    );
  }

  @override
  Stream<StreamingChunk> sendPromptStream(
    String prompt, {
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) {
    _ensureInitialized();
    return _model!.sendPromptStream(
      prompt,
      systemPrompt: systemPrompt,
      attachments: attachments,
      overrides: overrides,
    );
  }

  // ── Structured output ────────────────────────────────────────────────────

  /// Generates JSON constrained to [output]'s schema and decodes it into `T`.
  ///
  /// The schema constrains decoding, so the model cannot emit anything that
  /// would not parse. A schema llamadart cannot turn into a grammar is
  /// rejected when you build the [LlmStructuredOutput], not mid-generation.
  ///
  /// ```dart
  /// final output = LlmStructuredOutput.jsonSchema(
  ///   schema: {
  ///     'type': 'object',
  ///     'properties': {'city': {'type': 'string'}},
  ///     'required': ['city'],
  ///   },
  ///   decoder: (json) => json['city'] as String,
  /// );
  /// final result = await model.sendPromptStructured(
  ///   'Which city is the capital of Poland?',
  ///   output: output,
  /// );
  /// print(result.value);
  /// ```
  Future<LlmStructuredResult<T>> sendPromptStructured<T>(
    String prompt, {
    required LlmStructuredOutput<T> output,
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) {
    _ensureInitialized();
    return _model!.sendPromptStructured(
      prompt,
      output: output,
      systemPrompt: systemPrompt,
      attachments: attachments,
      overrides: overrides,
    );
  }

  // ── Tokenization ─────────────────────────────────────────────────────────

  /// Tokenizes [text] with the loaded model's tokenizer.
  Future<List<int>> tokenize(String text, {bool addSpecial = true}) {
    _ensureInitialized();
    return _model!.tokenize(text, addSpecial: addSpecial);
  }

  /// Turns token ids back into text.
  Future<String> detokenize(List<int> tokens, {bool special = false}) {
    _ensureInitialized();
    return _model!.detokenize(tokens, special: special);
  }

  /// Number of tokens [text] occupies — the honest way to check what fits in
  /// the context window, instead of counting characters.
  Future<int> countTokens(String text) {
    _ensureInitialized();
    return _model!.countTokens(text);
  }

  /// Context window the model was actually loaded with.
  Future<int> contextSize() {
    _ensureInitialized();
    return _model!.contextSize();
  }

  /// GGUF metadata key/value pairs.
  Future<Map<String, String>> metadata() {
    _ensureInitialized();
    return _model!.metadata();
  }

  // ── KV-cache state ───────────────────────────────────────────────────────

  /// Whether the KV cache can be saved and restored on this backend.
  Future<bool> get supportsStatePersistence {
    _ensureInitialized();
    return _model!.supportsStatePersistence;
  }

  /// Writes the current KV cache to [path] together with [tokens], the
  /// sequence that produced it (see [tokenize]).
  Future<bool> saveState(String path, {required List<int> tokens}) {
    _ensureInitialized();
    return _model!.saveState(path, tokens: tokens);
  }

  /// Restores a KV cache written by [saveState] and returns the tokens it was
  /// produced from — resuming a long conversation without paying for prompt
  /// ingestion again.
  Future<List<int>> loadState(String path, {int? tokenCapacity}) {
    _ensureInitialized();
    return _model!.loadState(path, tokenCapacity: tokenCapacity);
  }

  // ── LoRA ─────────────────────────────────────────────────────────────────

  /// Attaches a LoRA adapter at runtime, without reloading the model.
  Future<void> setLora(String path, {double scale = 1.0}) {
    _ensureInitialized();
    return _model!.setLora(path, scale: scale);
  }

  /// Detaches a previously attached LoRA adapter.
  Future<void> removeLora(String path) {
    _ensureInitialized();
    return _model!.removeLora(path);
  }

  /// Detaches every LoRA adapter.
  Future<void> clearLoras() {
    _ensureInitialized();
    return _model!.clearLoras();
  }

  // ── Diagnostics ──────────────────────────────────────────────────────────

  /// What the backend actually resolved: which backend won, how many layers
  /// really reached the GPU, whether vision and audio are usable.
  Future<ModelDiagnostics> diagnostics() {
    _ensureInitialized();
    return _model!.diagnostics();
  }

  @override
  bool get isGenerating => _model?.isGenerating ?? false;

  @override
  bool get isInitialized => _model?.isInitialized ?? false;

  /// Drops the reused prompt prefix and the KV cache it stands for, so the
  /// next generation starts from a clean context.
  ///
  /// llama.cpp has no call that forgets the cache on the spot: the next
  /// generation runs with `reusePromptPrefix: false`, which clears context
  /// memory and the cached prompt tokens before ingesting its prompt. Nothing
  /// is recomputed until then, so calling this is free.
  ///
  /// Works on both backends.
  Future<void> clean({bool resetConversations = true}) {
    _ensureInitialized();
    return _model!.clean(resetConversations: resetConversations);
  }

  void _ensureInitialized() {
    if (_model == null || !_model!.isInitialized) {
      throw StateError(
        'LocalModel is not initialized. Call loadModel() first.',
      );
    }
  }
}
