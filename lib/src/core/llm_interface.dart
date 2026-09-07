import 'package:llamadart/llamadart.dart' show LlamaImageContent;

import 'streaming_result.dart';

abstract interface class LlmInterface {
  Future<void> loadModel(String localPath);

  /// Sends a prompt and returns a stream of tokens.
  ///
  /// Pass [images] to enable vision (requires the model to have been loaded
  /// with a multimodal projector via `mmprojPath` in [LlmConfig]).
  Stream<String> sendPrompt(String prompt, {List<LlamaImageContent>? images});

  /// Sends a prompt and waits for the complete response.
  ///
  /// Pass [images] to enable vision.
  Future<String> sendPromptComplete(
    String prompt, {
    List<LlamaImageContent>? images,
  });

  /// Sends a prompt and returns a stream of [StreamingChunk] with live
  /// performance metrics. **Recommended** method for UI use.
  ///
  /// Pass [images] to enable vision.
  Stream<StreamingChunk> sendPromptStream(
    String prompt, {
    List<LlamaImageContent>? images,
  });

  /// Whether generation is currently in progress.
  bool get isGenerating;

  /// Whether the model is loaded and ready for generation.
  bool get isInitialized;

  /// Releases the model and any worker isolate backing it.
  ///
  /// Awaiting this matters: llama.cpp frees native handles during teardown,
  /// and loading another model before that finishes races those handles.
  Future<void> dispose();

  void clean();
}
