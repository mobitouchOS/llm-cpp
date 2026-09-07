// lib/src/core/performance_metrics.dart

/// Performance metrics for LLM text generation
class PerformanceMetrics {
  /// Number of tokens generated
  final int tokensGenerated;

  /// Total time taken for generation in milliseconds
  final int durationMs;

  /// Tokens per second (t/s)
  final double tokensPerSecond;

  /// Average time per token in milliseconds
  final double msPerToken;

  /// Start timestamp
  final DateTime startTime;

  /// End timestamp
  final DateTime endTime;

  /// Number of prompt tokens evaluated (prefill). Null unless the backend
  /// reported its own counters — see [isExact].
  final int? promptTokens;

  /// Time llama.cpp spent evaluating the prompt, in milliseconds. Null unless
  /// the backend reported its own counters.
  final double? promptEvalMs;

  /// Time llama.cpp spent generating tokens, in milliseconds. Null unless the
  /// backend reported its own counters.
  final double? evalMs;

  /// Whether these numbers come from llama.cpp's own counters.
  ///
  /// Live metrics emitted while streaming are estimates: llamadart batches the
  /// stream according to `streamBatchTokenThreshold` / `streamBatchByteThreshold`,
  /// so one chunk is not necessarily one token. The final chunk carries exact
  /// numbers whenever the backend exposes them.
  final bool isExact;

  PerformanceMetrics({
    required this.tokensGenerated,
    required this.durationMs,
    required this.tokensPerSecond,
    required this.msPerToken,
    required this.startTime,
    required this.endTime,
    this.promptTokens,
    this.promptEvalMs,
    this.evalMs,
    this.isExact = false,
  });

  /// Creates performance metrics from generation data.
  ///
  /// [tokenCount] is a chunk count, which only equals the token count while
  /// stream batching is disabled — the result is an estimate ([isExact] is
  /// false). Prefer [PerformanceMetrics.fromBackendPerf] when the backend
  /// reports its own counters.
  factory PerformanceMetrics.fromGeneration({
    required int tokenCount,
    required DateTime startTime,
    required DateTime endTime,
  }) {
    final duration = endTime.difference(startTime);
    final durationMs = duration.inMilliseconds;
    final tokensPerSecond = durationMs > 0
        ? (tokenCount * 1000) / durationMs
        : 0.0;
    final msPerToken = tokenCount > 0 ? durationMs / tokenCount : 0.0;

    return PerformanceMetrics(
      tokensGenerated: tokenCount,
      durationMs: durationMs,
      tokensPerSecond: tokensPerSecond,
      msPerToken: msPerToken,
      startTime: startTime,
      endTime: endTime,
    );
  }

  /// Creates exact metrics from llama.cpp's own performance counters.
  ///
  /// [evalTokens] and [evalMs] cover generation only, so [tokensPerSecond] is
  /// decode throughput and excludes prompt ingestion. When [evalMs] is missing
  /// or zero, throughput falls back to wall-clock time.
  factory PerformanceMetrics.fromBackendPerf({
    required int evalTokens,
    required DateTime startTime,
    required DateTime endTime,
    int? promptTokens,
    double? promptEvalMs,
    double? evalMs,
  }) {
    final wallClockMs = endTime.difference(startTime).inMilliseconds;
    final measuredMs = (evalMs != null && evalMs > 0)
        ? evalMs
        : wallClockMs.toDouble();

    return PerformanceMetrics(
      tokensGenerated: evalTokens,
      durationMs: wallClockMs,
      tokensPerSecond: measuredMs > 0 ? (evalTokens * 1000) / measuredMs : 0.0,
      msPerToken: evalTokens > 0 ? measuredMs / evalTokens : 0.0,
      startTime: startTime,
      endTime: endTime,
      promptTokens: promptTokens,
      promptEvalMs: promptEvalMs,
      evalMs: evalMs,
      isExact: true,
    );
  }

  /// Duration of generation
  Duration get duration => Duration(milliseconds: durationMs);

  @override
  String toString() {
    return 'PerformanceMetrics('
        'tokens: $tokensGenerated, '
        'duration: ${durationMs}ms, '
        't/s: ${tokensPerSecond.toStringAsFixed(2)}, '
        'ms/token: ${msPerToken.toStringAsFixed(2)}, '
        'exact: $isExact'
        ')';
  }

  /// Converts to JSON-compatible map
  Map<String, dynamic> toJson() {
    return {
      'tokensGenerated': tokensGenerated,
      'durationMs': durationMs,
      'tokensPerSecond': tokensPerSecond,
      'msPerToken': msPerToken,
      'startTime': startTime.toIso8601String(),
      'endTime': endTime.toIso8601String(),
      if (promptTokens != null) 'promptTokens': promptTokens,
      if (promptEvalMs != null) 'promptEvalMs': promptEvalMs,
      if (evalMs != null) 'evalMs': evalMs,
      'isExact': isExact,
    };
  }

  /// Creates from JSON map
  factory PerformanceMetrics.fromJson(Map<String, dynamic> json) {
    return PerformanceMetrics(
      tokensGenerated: json['tokensGenerated'] as int,
      durationMs: json['durationMs'] as int,
      tokensPerSecond: (json['tokensPerSecond'] as num).toDouble(),
      msPerToken: (json['msPerToken'] as num).toDouble(),
      startTime: DateTime.parse(json['startTime'] as String),
      endTime: DateTime.parse(json['endTime'] as String),
      promptTokens: json['promptTokens'] as int?,
      promptEvalMs: (json['promptEvalMs'] as num?)?.toDouble(),
      evalMs: (json['evalMs'] as num?)?.toDouble(),
      isExact: json['isExact'] as bool? ?? false,
    );
  }
}
