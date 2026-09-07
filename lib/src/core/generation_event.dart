// lib/src/core/generation_event.dart

/// One message from a worker isolate: either a token, or the terminal event
/// carrying the generation's finish reason and llama.cpp's perf counters.
///
/// Internal to the plugin — not exported from `mt_llmkit.dart`.
class GenerationEvent {
  final String text;

  /// Reasoning text, when the model emitted a thinking block. A token event
  /// carries either [text] or [thinking], never both.
  final String? thinking;

  final bool isFinal;
  final String? finishReason;
  final Map<String, dynamic>? perf;

  const GenerationEvent({
    required this.text,
    this.thinking,
    this.isFinal = false,
    this.finishReason,
    this.perf,
  });
}
