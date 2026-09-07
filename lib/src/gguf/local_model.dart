// lib/src/gguf/local_model.dart

import 'package:llamadart/llamadart.dart' show LlamaImageContent;

import '../core/llm_config.dart';
import '../core/llm_interface.dart';
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
    // Awaited: the previous model's native handles must be freed before the
    // next one starts allocating.
    await _model?.dispose();
    _model = backend == ModelBackend.isolate
        ? LlmModelIsolated(config)
        : LlmModelStandard(config);
    await _model!.loadModel(localPath);
  }

  @override
  Future<void> dispose() async {
    final model = _model;
    _model = null;
    await model?.dispose();
  }

  @override
  Stream<String> sendPrompt(String prompt, {List<LlamaImageContent>? images}) {
    _ensureInitialized();
    return _model!.sendPrompt(prompt, images: images);
  }

  @override
  Future<String> sendPromptComplete(
    String prompt, {
    List<LlamaImageContent>? images,
  }) {
    _ensureInitialized();
    return _model!.sendPromptComplete(prompt, images: images);
  }

  @override
  Stream<StreamingChunk> sendPromptStream(
    String prompt, {
    List<LlamaImageContent>? images,
  }) {
    _ensureInitialized();
    return _model!.sendPromptStream(prompt, images: images);
  }

  @override
  bool get isGenerating => _model?.isGenerating ?? false;

  @override
  bool get isInitialized => _model?.isInitialized ?? false;

  /// Resets the conversation context without reloading the model.
  ///
  /// Only supported with [ModelBackend.inProcess].
  /// Throws [UnsupportedError] with [ModelBackend.isolate].
  @override
  void clean() {
    _ensureInitialized();
    if (backend == ModelBackend.inProcess) {
      _model!.clean();
    } else {
      throw UnsupportedError('clean() requires ModelBackend.inProcess.');
    }
  }

  void _ensureInitialized() {
    if (_model == null || !_model!.isInitialized) {
      throw StateError(
        'LocalModel is not initialized. Call loadModel() first.',
      );
    }
  }
}
