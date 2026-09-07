// lib/src/core/backend_perf.dart

import 'package:llamadart/llamadart.dart';

import 'performance_metrics.dart';

/// Reads llama.cpp's own performance counters as a plain, isolate-sendable map.
///
/// Returns null when the backend does not expose them (`getPerformanceContext`
/// is optional) or when reading them fails — metrics are diagnostics, and must
/// never break a generation that already produced its output.
Future<Map<String, dynamic>?> readBackendPerf(LlamaEngine engine) async {
  try {
    final perf = await engine.getPerformanceContext();
    if (perf == null) return null;
    return {
      'evalTokens': perf.evalTokens,
      'promptTokens': perf.promptEvalTokens,
      'promptEvalMs': perf.promptEvalMs,
      'evalMs': perf.evalMs,
    };
  } catch (_) {
    return null;
  }
}

/// Builds the final metrics for a generation.
///
/// Prefers the backend counters in [perf]; falls back to the chunk-count
/// estimate when they are unavailable.
PerformanceMetrics finalMetrics({
  required Map<String, dynamic>? perf,
  required int fallbackTokenCount,
  required DateTime startTime,
  required DateTime endTime,
}) {
  final evalTokens = perf?['evalTokens'] as int?;
  if (evalTokens == null) {
    return PerformanceMetrics.fromGeneration(
      tokenCount: fallbackTokenCount,
      startTime: startTime,
      endTime: endTime,
    );
  }

  return PerformanceMetrics.fromBackendPerf(
    evalTokens: evalTokens,
    startTime: startTime,
    endTime: endTime,
    promptTokens: perf?['promptTokens'] as int?,
    promptEvalMs: (perf?['promptEvalMs'] as num?)?.toDouble(),
    evalMs: (perf?['evalMs'] as num?)?.toDouble(),
  );
}
