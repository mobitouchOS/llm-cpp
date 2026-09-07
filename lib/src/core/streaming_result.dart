// lib/src/core/streaming_result.dart

import 'performance_metrics.dart';

/// Streaming chunk with optional performance metrics
class StreamingChunk {
  /// Text chunk
  final String text;

  /// Reasoning text, for models that emit a thinking block.
  ///
  /// llamadart routes reasoning to its own channel, so it never appears in
  /// [text]. A chunk carries one or the other. Set `enableThinking: false` in
  /// [LlmConfig] to stop paying decode time for reasoning you do not show.
  final String? thinking;

  /// Current performance metrics (calculated so far)
  final PerformanceMetrics? metrics;

  /// Whether this is the last chunk
  final bool isFinal;

  /// Why generation stopped, as reported by llama.cpp — `'stop'` for a clean
  /// end (EOS or a stop sequence) and `'length'` when the token budget ran
  /// out. Only set on the final chunk, and null when the backend does not
  /// report one.
  final String? finishReason;

  /// Whether generation stopped because it hit the token budget
  /// (`nPredict`) rather than finishing on its own.
  bool get isTruncated => finishReason == 'length';

  StreamingChunk({
    required this.text,
    this.thinking,
    this.metrics,
    this.isFinal = false,
    this.finishReason,
  });

  @override
  String toString() {
    return 'StreamingChunk(text: "$text", thinking: $thinking, '
        'metrics: $metrics, isFinal: $isFinal, finishReason: $finishReason)';
  }
}
