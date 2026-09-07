// lib/src/models/llm_model_base.dart
import 'package:flutter/foundation.dart';

import 'package:llamadart/llamadart.dart' show LlamaContentPart;

import '../core/generation_overrides.dart';
import '../core/llm_interface.dart';
import '../core/model_diagnostics.dart';
import '../core/structured_output.dart';

abstract class LlmModelBase implements LlmInterface {
  // ── Lifecycle ────────────────────────────────────────────────────────────

  /// Frees the model and its context but keeps the backend — and, for the
  /// isolate backend, the worker isolate — alive, so the next [loadModel]
  /// skips isolate spawn and llama.cpp initialization. Switching models is
  /// the case this exists for.
  Future<void> unload();

  /// Drops the reused prompt prefix and the KV cache it stands for.
  ///
  /// llama.cpp has no call that forgets the cache on the spot. The next
  /// generation runs with `reusePromptPrefix: false`, which clears context
  /// memory and the cached prompt tokens before ingesting its prompt —
  /// nothing is recomputed until then, so calling this is free.
  Future<void> clean({bool resetConversations = true});

  // ── Structured output ────────────────────────────────────────────────────

  /// Generates JSON constrained to [output]'s schema and decodes it.
  ///
  /// Implemented over [sendPromptStream]: only the response format crosses to
  /// the worker, and decoding happens on this isolate.
  Future<LlmStructuredResult<T>> sendPromptStructured<T>(
    String prompt, {
    required LlmStructuredOutput<T> output,
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) => sendPromptStream(
    prompt,
    systemPrompt: systemPrompt,
    attachments: attachments,
    overrides: (overrides ?? const GenerationOverrides()).copyWith(
      responseFormat: output.responseFormat,
    ),
  ).parseStructured(output);

  // ── Tokenization ─────────────────────────────────────────────────────────

  /// Tokenizes [text] with the loaded model's tokenizer.
  Future<List<int>> tokenize(String text, {bool addSpecial = true});

  /// Turns token ids back into text.
  Future<String> detokenize(List<int> tokens, {bool special = false});

  /// Number of tokens [text] occupies. Use this rather than counting
  /// characters when deciding what fits in the context window.
  Future<int> countTokens(String text);

  /// Context window the model was actually loaded with.
  Future<int> contextSize();

  /// GGUF metadata key/value pairs.
  Future<Map<String, String>> metadata();

  // ── KV-cache state ───────────────────────────────────────────────────────

  /// Whether this backend can save and restore the KV cache.
  Future<bool> get supportsStatePersistence;

  /// Writes the current KV cache to [path] along with [tokens], the sequence
  /// it was produced from.
  Future<bool> saveState(String path, {required List<int> tokens});

  /// Restores a KV cache written by [saveState] and returns the token
  /// sequence it came from. Skips re-evaluating the prompt, which is the
  /// difference between a multi-second resume and an instant one.
  Future<List<int>> loadState(String path, {int? tokenCapacity});

  // ── LoRA ─────────────────────────────────────────────────────────────────

  /// Attaches a LoRA adapter to the loaded model.
  Future<void> setLora(String path, {double scale = 1.0});

  /// Detaches a previously attached LoRA adapter.
  Future<void> removeLora(String path);

  /// Detaches every LoRA adapter.
  Future<void> clearLoras();

  // ── Diagnostics ──────────────────────────────────────────────────────────

  /// What the backend actually resolved for this model.
  Future<ModelDiagnostics> diagnostics();

  bool _isInitialized = false;
  bool _isDisposed = false;
  bool _isGenerating = false;

  @override
  bool get isInitialized => _isInitialized;
  bool get isDisposed => _isDisposed;

  @override
  bool get isGenerating => _isGenerating;

  @protected
  void markAsInitialized() {
    _isInitialized = true;
  }

  /// Marks the model as unloaded while keeping the instance usable: the
  /// backend and its worker isolate stay alive, so the next [loadModel] skips
  /// spawning and backend initialization.
  @protected
  void markAsUnloaded() {
    _isInitialized = false;
    _isGenerating = false;
  }

  @protected
  void markAsDisposed() {
    _isDisposed = true;
    _isInitialized = false;
    _isGenerating = false;
  }

  @protected
  void markGenerationStart() {
    _isGenerating = true;
  }

  @protected
  void markGenerationEnd() {
    _isGenerating = false;
  }

  @protected
  void checkInitialized() {
    if (!_isInitialized) {
      throw StateError('Model not initialized. Call loadModel() first.');
    }
    if (_isDisposed) {
      throw StateError('Model has been disposed.');
    }
  }

  @protected
  void checkNotDisposed() {
    if (_isDisposed) {
      throw StateError('Model has been disposed.');
    }
  }
}
