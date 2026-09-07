// lib/src/core/generation_result.dart

import 'performance_metrics.dart';
import 'tools.dart';

/// Everything one generation produced.
///
/// `sendPromptComplete` returns only the answer text, which silently discards
/// the reasoning a thinking model spent decode time on, the tool calls it
/// requested, and whether it was cut off. This carries all of it.
class GenerationResult {
  /// The answer text.
  final String text;

  /// Reasoning text, when the model emitted a thinking block. Never part of
  /// [text] — llamadart keeps the two channels separate.
  final String? thinking;

  /// Tool calls the model requested, if tools were declared.
  final List<LlmToolCall> toolCalls;

  /// Why generation stopped: `'stop'` for a clean end, `'length'` when the
  /// token budget ran out.
  final String? finishReason;

  final PerformanceMetrics? metrics;

  /// Whether the reasoning block was cut short by
  /// [LlmConfig.thinkingBudget].
  ///
  /// llamadart does not report this: the budget is enforced by a native
  /// sampler that forces the closing tag, and no event, flag or distinct
  /// finish reason reaches Dart. So this is derived, and how reliable it is
  /// depends on how the budget was configured — see [thinkingTokens].
  final bool thinkingTruncated;

  /// Token count of [thinking], measured when a budget was set without a
  /// `forcedMessage`.
  ///
  /// With `ThinkingBudget.forcedMessage` set, [thinkingTruncated] is exact:
  /// the forced text either terminates the reasoning or it does not. Without
  /// it, truncation is inferred from this count reaching the budget, which a
  /// model that happens to stop right at the limit would also trigger. Set
  /// `forcedMessage` when you need to trust the flag.
  final int? thinkingTokens;

  const GenerationResult({
    required this.text,
    this.thinking,
    this.toolCalls = const [],
    this.finishReason,
    this.metrics,
    this.thinkingTruncated = false,
    this.thinkingTokens,
  });

  /// Whether generation stopped because it hit the token budget.
  bool get isTruncated => finishReason == 'length';

  /// Whether the model asked for tools to be run.
  bool get needsToolResults => toolCalls.isNotEmpty;

  @override
  String toString() =>
      'GenerationResult(text: ${text.length} chars, '
      'thinking: ${thinking?.length ?? 0} chars, '
      'toolCalls: ${toolCalls.length}, finishReason: $finishReason)';
}
