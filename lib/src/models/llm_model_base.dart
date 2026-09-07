// lib/src/models/llm_model_base.dart
import 'package:flutter/foundation.dart';

import '../core/llm_interface.dart';
import '../core/model_diagnostics.dart';

abstract class LlmModelBase implements LlmInterface {
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
