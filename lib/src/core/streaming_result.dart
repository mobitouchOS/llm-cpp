// lib/src/core/streaming_result.dart

import 'performance_metrics.dart';
import 'tools.dart';

/// Streaming chunk with optional performance metrics
class StreamingChunk {
  /// Text chunk
  final String text;

  /// Reasoning text, for models that emit a thinking block.
  ///
  /// llamadart routes reasoning to its own channel, so it never appears in
  /// [text], and both backends emit it as its own chunk — a chunk carries one
  /// channel or the other. Set `enableThinking: false` in [LlmConfig] to stop
  /// paying decode time for reasoning you do not show.
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

  /// Tool calls the model requested, reassembled from their streamed
  /// fragments. Only set on the final chunk, and empty unless tools were
  /// declared through [GenerationOverrides.tools].
  final List<LlmToolCall> toolCalls;

  /// Whether generation stopped because it hit the token budget
  /// (`nPredict`) rather than finishing on its own.
  bool get isTruncated => finishReason == 'length';

  StreamingChunk({
    required this.text,
    this.thinking,
    this.metrics,
    this.isFinal = false,
    this.finishReason,
    this.toolCalls = const [],
  });

  @override
  String toString() {
    return 'StreamingChunk(text: "$text", thinking: $thinking, '
        'metrics: $metrics, isFinal: $isFinal, finishReason: $finishReason, '
        'toolCalls: $toolCalls)';
  }
}
