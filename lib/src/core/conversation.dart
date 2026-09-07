// lib/src/core/conversation.dart

import 'dart:async';

import 'package:llamadart/llamadart.dart' show LlamaContentPart;

import 'backend_perf.dart';
import 'chat_message.dart';
import 'conversation_types.dart';
import 'generation_event.dart';
import 'generation_overrides.dart';
import 'performance_metrics.dart';
import 'session_ops.dart';

/// How a [Conversation] reaches the session that holds its history.
///
/// The session lives next to the engine — inside the worker isolate for
/// [ModelBackend.isolate], in this isolate for [ModelBackend.inProcess] — so
/// this is the seam the two backends differ at, and nothing else about a
/// conversation is backend-specific.
abstract interface class ConversationTransport {
  Stream<GenerationEvent> turn(String id, TurnRequest request);
  Future<List<LlmChatMessage>> history(String id);
  Future<void> restore(String id, List<LlmChatMessage> messages);
  Future<void> reset(String id, {bool keepSystemPrompt});
  Future<void> setSystemPrompt(String id, String? value);
  Future<void> cancel(String id);
  Future<void> close(String id);
}

/// A multi-turn conversation with one loaded model.
///
/// The model sees the whole exchange, so follow-up questions work without the
/// app concatenating prompts by hand. History lives with the engine and is
/// trimmed as it approaches the context window — see [trims], which reports
/// what was dropped, and [ContextOverflowPolicy].
///
/// ```dart
/// final chat = await model.startConversation(
///   systemPrompt: 'You are a concise assistant.',
/// );
///
/// await for (final chunk in chat.send('What is the capital of Poland?')) {
///   stdout.write(chunk.text);
/// }
/// final follow = await chat.sendComplete('And its population?');
/// print(follow.text);
///
/// await chat.close();
/// ```
///
/// Turns are serialized across every conversation on a model: llama.cpp chat
/// generation runs on one context, so two conversations cannot generate at the
/// same time — a second turn waits rather than interleaving.
class Conversation {
  /// Opaque identifier for the session behind this conversation.
  final String id;

  final ConversationTransport _transport;
  final StreamController<ContextTrimEvent> _trims =
      StreamController<ContextTrimEvent>.broadcast();

  bool _closed = false;

  Conversation(this.id, this._transport);

  /// Fires whenever a turn permanently deleted history to make the prompt fit.
  ///
  /// Broadcast, so an app that only reads the text of a turn still finds out
  /// its oldest exchanges are gone — otherwise the only symptom is the model
  /// mysteriously forgetting.
  Stream<ContextTrimEvent> get trims => _trims.stream;

  /// Whether [close] has been called.
  bool get isClosed => _closed;

  /// Sends a user turn and streams the reply.
  Stream<ConversationChunk> send(
    String message, {
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) => _turn(
    TurnRequest(
      message: message,
      attachments: attachments ?? const [],
      overrides: overrides,
    ),
  );

  /// Sends a user turn and waits for the whole reply.
  Future<ConversationTurn> sendComplete(
    String message, {
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) => _collect(send(message, attachments: attachments, overrides: overrides));

  /// Answers the tool calls from the previous turn and lets the model continue.
  ///
  /// Each result becomes its own message: llamadart's message format keeps
  /// only the first tool result per message, so batching them would silently
  /// drop all but one.
  Stream<ConversationChunk> submitToolResults(
    List<ToolResult> results, {
    GenerationOverrides? overrides,
  }) => _turn(TurnRequest(toolResults: results, overrides: overrides));

  /// Answers tool calls and waits for the whole continuation.
  Future<ConversationTurn> submitToolResultsComplete(
    List<ToolResult> results, {
    GenerationOverrides? overrides,
  }) => _collect(submitToolResults(results, overrides: overrides));

  /// The conversation as the model currently sees it, oldest first.
  ///
  /// Trimmed turns are gone from here too — that is what [trims] reports.
  Future<List<LlmChatMessage>> history() {
    _checkOpen();
    return _transport.history(id);
  }

  /// Replaces the history, for resuming a stored conversation.
  Future<void> restore(List<LlmChatMessage> messages) {
    _checkOpen();
    return _transport.restore(id, messages);
  }

  /// Clears the history. Keeps the system prompt unless told otherwise.
  Future<void> reset({bool keepSystemPrompt = true}) {
    _checkOpen();
    return _transport.reset(id, keepSystemPrompt: keepSystemPrompt);
  }

  /// Replaces the system prompt for every later turn.
  Future<void> setSystemPrompt(String? value) {
    _checkOpen();
    return _transport.setSystemPrompt(id, value);
  }

  /// Stops the turn in progress.
  ///
  /// The partial reply stays in history, so the next turn continues from a
  /// coherent exchange rather than two user messages in a row.
  Future<void> cancel() {
    _checkOpen();
    return _transport.cancel(id);
  }

  /// Releases the session. The conversation is unusable afterwards.
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    await _trims.close();
    await _transport.close(id);
  }

  // ── Internals ────────────────────────────────────────────────────────────

  Stream<ConversationChunk> _turn(TurnRequest request) async* {
    _checkOpen();

    final startTime = DateTime.now();
    var tokenCount = 0;

    await for (final event in _transport.turn(id, request)) {
      if (event.isFinal) {
        if (event.dropped.isNotEmpty && !_trims.isClosed) {
          _trims.add(
            ContextTrimEvent(
              dropped: event.dropped,
              fitContext: event.fitContext,
            ),
          );
        }

        yield ConversationChunk(
          text: '',
          metrics: finalMetrics(
            perf: event.perf,
            fallbackTokenCount: tokenCount,
            startTime: startTime,
            endTime: DateTime.now(),
          ),
          isFinal: true,
          finishReason: event.finishReason,
          toolCalls: event.toolCalls,
          fitContext: event.fitContext,
          dropped: event.dropped,
        );
        continue;
      }

      tokenCount += 1;
      yield ConversationChunk(
        text: event.text,
        thinking: event.thinking,
        metrics: PerformanceMetrics.fromGeneration(
          tokenCount: tokenCount,
          startTime: startTime,
          endTime: DateTime.now(),
        ),
      );
    }
  }

  Future<ConversationTurn> _collect(Stream<ConversationChunk> stream) async {
    final text = StringBuffer();
    final thinking = StringBuffer();
    var turn = const ConversationTurn(text: '');

    await for (final chunk in stream) {
      text.write(chunk.text);
      if (chunk.thinking != null) thinking.write(chunk.thinking);
      if (chunk.isFinal) {
        turn = ConversationTurn(
          text: '',
          toolCalls: chunk.toolCalls,
          finishReason: chunk.finishReason,
          metrics: chunk.metrics,
          fitContext: chunk.fitContext,
          dropped: chunk.dropped,
        );
      }
    }

    return ConversationTurn(
      text: text.toString(),
      thinking: thinking.isEmpty ? null : thinking.toString(),
      toolCalls: turn.toolCalls,
      finishReason: turn.finishReason,
      metrics: turn.metrics,
      fitContext: turn.fitContext,
      dropped: turn.dropped,
    );
  }

  void _checkOpen() {
    if (_closed) {
      throw StateError('Conversation "$id" has been closed.');
    }
  }
}

/// Reaches a [SessionRegistry] living in this isolate.
class LocalConversationTransport implements ConversationTransport {
  final SessionRegistry registry;

  LocalConversationTransport(this.registry);

  @override
  Stream<GenerationEvent> turn(String id, TurnRequest request) =>
      registry.turn(id, request);

  @override
  Future<List<LlmChatMessage>> history(String id) async => registry.history(id);

  @override
  Future<void> restore(String id, List<LlmChatMessage> messages) async =>
      registry.restore(id, messages);

  @override
  Future<void> reset(String id, {bool keepSystemPrompt = true}) async =>
      registry.reset(id, keepSystemPrompt: keepSystemPrompt);

  @override
  Future<void> setSystemPrompt(String id, String? value) async =>
      registry.setSystemPrompt(id, value);

  @override
  Future<void> cancel(String id) async => registry.cancel(id);

  @override
  Future<void> close(String id) async => registry.close(id);
}
