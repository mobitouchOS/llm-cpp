import 'package:llamadart/llamadart.dart' show LlamaContentPart;

import 'generation_overrides.dart';
import 'streaming_result.dart';

abstract interface class LlmInterface {
  Future<void> loadModel(String localPath);

  /// Sends a prompt and returns a stream of tokens.
  ///
  /// [systemPrompt] is passed as a real `system` message, so the model's chat
  /// template can place it where it belongs instead of it being smuggled into
  /// the user turn.
  ///
  /// [attachments] carries non-text content — [LlamaImageContent] for vision
  /// (requires the model to have been loaded with a multimodal projector via
  /// `mmprojPath` in [LlmConfig]) and [LlamaAudioContent] for audio-capable
  /// models.
  ///
  /// [overrides] adjusts sampling for this request only.
  Stream<String> sendPrompt(
    String prompt, {
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  });

  /// Sends a prompt and waits for the complete response.
  Future<String> sendPromptComplete(
    String prompt, {
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  });

  /// Sends a prompt and returns a stream of [StreamingChunk] with live
  /// performance metrics. **Recommended** method for UI use.
  Stream<StreamingChunk> sendPromptStream(
    String prompt, {
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
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
}
