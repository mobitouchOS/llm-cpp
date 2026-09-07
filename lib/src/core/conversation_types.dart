// lib/src/core/conversation_types.dart

import 'chat_message.dart';
import 'performance_metrics.dart';
import 'streaming_result.dart';
import 'tools.dart';

/// What to do when a conversation no longer fits the model's context window.
enum ContextOverflowPolicy {
  /// Match llamadart: drop the oldest turns, and if it still does not fit,
  /// send the oversized prompt anyway and report it through
  /// [ConversationChunk.fitContext].
  allow,

  /// Fail the turn with [LlmContextOverflowException] instead of answering
  /// from a prompt the model could not fully see.
  fail,
}

/// Thrown when a turn's prompt does not fit the context window and the
/// conversation was opened with [ContextOverflowPolicy.fail].
class LlmContextOverflowException implements Exception {
  final String message;

  LlmContextOverflowException(this.message);

  @override
  String toString() => 'LlmContextOverflowException: $message';
}

/// Turns that were permanently deleted from a conversation's history to make
/// the prompt fit.
///
/// llamadart trims silently — it removes the oldest turns and logs a warning.
/// Without this an app would only notice by seeing the model forget things.
class ContextTrimEvent {
  /// The messages that are gone.
  final List<LlmChatMessage> dropped;

  /// Whether the prompt fitted *after* the trim. False means llamadart gave
  /// up and sent an oversized prompt.
  final bool fitContext;

  const ContextTrimEvent({required this.dropped, required this.fitContext});

  @override
  String toString() =>
      'ContextTrimEvent(${dropped.length} messages dropped, '
      'fitContext: $fitContext)';
}

/// A streamed chunk of a conversation turn.
class ConversationChunk extends StreamingChunk {
  /// False when the rendered prompt did not fit the context window even after
  /// older turns were dropped. Only meaningful on the final chunk.
  final bool fitContext;

  /// Turns permanently deleted from history to make this request fit. Only
  /// populated on the final chunk.
  final List<LlmChatMessage> dropped;

  ConversationChunk({
    required super.text,
    super.thinking,
    super.metrics,
    super.isFinal,
    super.finishReason,
    super.toolCalls,
    this.fitContext = true,
    this.dropped = const [],
  });
}

/// A finished conversation turn.
class ConversationTurn {
  final String text;

  /// Reasoning the model emitted, when it emitted any.
  final String? thinking;

  /// Tool calls the model is waiting on. Answer them with
  /// `Conversation.submitToolResults`.
  final List<LlmToolCall> toolCalls;

  final String? finishReason;
  final PerformanceMetrics? metrics;

  /// False when the prompt did not fit the context window even after trimming.
  final bool fitContext;

  /// Turns permanently deleted from history to make this request fit.
  final List<LlmChatMessage> dropped;

  const ConversationTurn({
    required this.text,
    this.thinking,
    this.toolCalls = const [],
    this.finishReason,
    this.metrics,
    this.fitContext = true,
    this.dropped = const [],
  });

  /// Whether the model asked for tools to be run before it can continue.
  bool get needsToolResults => toolCalls.isNotEmpty;

  /// Whether generation stopped because it hit the token budget.
  bool get isTruncated => finishReason == 'length';

  @override
  String toString() =>
      'ConversationTurn(${text.length} chars, '
      'toolCalls: ${toolCalls.length}, fitContext: $fitContext)';
}
