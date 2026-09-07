// lib/src/core/session_ops.dart
//
// Multi-turn conversation state. Shared by both backends: the in-process one
// calls this directly, the isolate one reaches it through the worker port, so
// there is one implementation of the tricky parts — context-trim detection,
// cancellation repair, tool-result fan-out and turn serialization.

import 'dart:async';

import 'package:llamadart/llamadart.dart';

import 'chat_message.dart';
import 'conversation_types.dart';
import 'generation_event.dart';
import 'generation_overrides.dart';
import 'llm_config.dart';
import 'tools.dart';

/// The slice of llamadart's `ChatSession` this registry depends on.
///
/// Narrow on purpose: it is the seam that lets every rule below be tested
/// without an engine, an isolate or a model file.
abstract interface class ChatSessionLike {
  String? get systemPrompt;
  set systemPrompt(String? value);

  int? get maxContextTokens;
  set maxContextTokens(int? value);

  List<LlamaChatMessage> get history;

  /// False when the rendered prompt did not fit even after trimming. llamadart
  /// sends it anyway.
  bool get lastRequestFitContext;

  void addMessage(LlamaChatMessage message);

  void reset({bool keepSystemPrompt});

  Stream<LlamaCompletionChunk> create(
    List<LlamaContentPart> parts, {
    GenerationParams? params,
    List<ToolDefinition>? tools,
    ToolChoice? toolChoice,
    bool parallelToolCalls,
    bool enableThinking,
    bool continuesPreviousTurn,
  });
}

/// Adapts llamadart's [ChatSession] to [ChatSessionLike].
class RealChatSession implements ChatSessionLike {
  final ChatSession _session;

  RealChatSession(this._session);

  @override
  String? get systemPrompt => _session.systemPrompt;
  @override
  set systemPrompt(String? value) => _session.systemPrompt = value;

  @override
  int? get maxContextTokens => _session.maxContextTokens;
  @override
  set maxContextTokens(int? value) => _session.maxContextTokens = value;

  @override
  List<LlamaChatMessage> get history => _session.history;

  @override
  bool get lastRequestFitContext => _session.lastRequestFitContext;

  @override
  void addMessage(LlamaChatMessage message) => _session.addMessage(message);

  @override
  void reset({bool keepSystemPrompt = true}) =>
      _session.reset(keepSystemPrompt: keepSystemPrompt);

  @override
  Stream<LlamaCompletionChunk> create(
    List<LlamaContentPart> parts, {
    GenerationParams? params,
    List<ToolDefinition>? tools,
    ToolChoice? toolChoice,
    bool parallelToolCalls = false,
    bool enableThinking = true,
    bool continuesPreviousTurn = false,
  }) => _session.create(
    parts,
    params: params,
    tools: tools,
    toolChoice: toolChoice,
    parallelToolCalls: parallelToolCalls,
    enableThinking: enableThinking,
    continuesPreviousTurn: continuesPreviousTurn,
  );
}

/// How a conversation was opened.
class SessionOptions {
  final String? systemPrompt;
  final int? maxContextTokens;
  final ContextOverflowPolicy overflowPolicy;
  final bool keepThinkingInHistory;

  const SessionOptions({
    this.systemPrompt,
    this.maxContextTokens,
    this.overflowPolicy = ContextOverflowPolicy.allow,
    this.keepThinkingInHistory = false,
  });
}

/// One turn's input.
class TurnRequest {
  /// The user's message. Null for a continuation (after tool results).
  final String? message;

  final List<LlamaContentPart> attachments;

  /// Answers to the previous turn's tool calls, appended before generating.
  final List<ToolResult> toolResults;

  final GenerationOverrides? overrides;

  const TurnRequest({
    this.message,
    this.attachments = const [],
    this.toolResults = const [],
    this.overrides,
  });
}

class _SessionEntry {
  final ChatSessionLike session;
  final SessionOptions options;
  StreamSubscription<LlamaCompletionChunk>? subscription;

  _SessionEntry(this.session, this.options);
}

/// Holds every open conversation for one loaded model.
class SessionRegistry {
  /// Builds a session. Injected so tests can supply a fake.
  final ChatSessionLike Function(SessionOptions options) _createSession;

  /// Base generation parameters from [LlmConfig], applied under any overrides.
  final GenerationParams Function() _baseParams;

  /// Whether reasoning is on by default for this model.
  final bool Function() _enableThinkingDefault;

  /// Reads llama.cpp's own perf counters for the finished turn.
  final Future<Map<String, dynamic>?> Function() _readPerf;

  /// Runs a turn to completion without letting anything else generate.
  ///
  /// llama.cpp chat generation runs on sequence 0 of a single context, so
  /// turns cannot overlap — `maxParallelSequences` sizes the KV cache for
  /// batched embeddings, not for chat. The backend supplies this so
  /// conversation turns and one-shot prompts share one queue instead of
  /// racing on two.
  final Future<void> Function(Future<void> Function() action) _serialize;

  final _sessions = <String, _SessionEntry>{};

  SessionRegistry({
    required ChatSessionLike Function(SessionOptions options) createSession,
    required GenerationParams Function() baseParams,
    required bool Function() enableThinkingDefault,
    required Future<Map<String, dynamic>?> Function() readPerf,
    required Future<void> Function(Future<void> Function() action) serialize,
  }) : _createSession = createSession,
       _baseParams = baseParams,
       _enableThinkingDefault = enableThinkingDefault,
       _readPerf = readPerf,
       _serialize = serialize;

  bool get isEmpty => _sessions.isEmpty;

  void open(
    String id,
    SessionOptions options, {
    List<LlmChatMessage>? history,
  }) {
    final session = _createSession(options)
      ..systemPrompt = options.systemPrompt
      ..maxContextTokens = options.maxContextTokens;
    _sessions[id] = _SessionEntry(session, options);
    if (history != null) restore(id, history);
  }

  void close(String id) {
    final entry = _sessions.remove(id);
    entry?.subscription?.cancel();
  }

  /// Closes every conversation. Used when the model is unloaded or the whole
  /// context is cleaned.
  void closeAll() {
    for (final id in _sessions.keys.toList()) {
      close(id);
    }
  }

  void resetAll({bool keepSystemPrompt = true}) {
    for (final entry in _sessions.values) {
      entry.session.reset(keepSystemPrompt: keepSystemPrompt);
    }
  }

  _SessionEntry _entry(String id) {
    final entry = _sessions[id];
    if (entry == null) {
      throw StateError('Conversation "$id" is closed.');
    }
    return entry;
  }

  List<LlmChatMessage> history(String id) =>
      _entry(id).session.history.map(fromLlamaMessage).toList();

  void restore(String id, List<LlmChatMessage> messages) {
    final session = _entry(id).session..reset(keepSystemPrompt: true);
    for (final message in messages) {
      // A system message added to history is silently dropped from the prompt
      // by llamadart; the systemPrompt field is the only one that reaches the
      // model, so route it there instead of losing it.
      if (message.role == LlmChatRole.system) {
        session.systemPrompt = message.text;
        continue;
      }
      session.addMessage(toLlamaMessage(message));
    }
  }

  void reset(String id, {bool keepSystemPrompt = true}) =>
      _entry(id).session.reset(keepSystemPrompt: keepSystemPrompt);

  void setSystemPrompt(String id, String? value) =>
      _entry(id).session.systemPrompt = value;

  void cancel(String id) {
    final entry = _sessions[id];
    entry?.subscription?.cancel();
    entry?.subscription = null;
  }

  /// Runs one turn, queued behind any other turn on this model.
  Stream<GenerationEvent> turn(String id, TurnRequest request) {
    final controller = StreamController<GenerationEvent>();
    // Queued eagerly so two turns started back to back keep their order.
    unawaited(_serialize(() => _runTurn(id, request, controller)));
    controller.onCancel = () => cancel(id);
    return controller.stream;
  }

  Future<void> _runTurn(
    String id,
    TurnRequest request,
    StreamController<GenerationEvent> controller,
  ) async {
    if (controller.isClosed) return;

    final _SessionEntry entry;
    try {
      entry = _entry(id);
    } catch (error, stackTrace) {
      controller.addError(error, stackTrace);
      await controller.close();
      return;
    }

    final session = entry.session;
    final overrides = request.overrides;

    // One message per tool result: llamadart's message JSON keeps only the
    // first tool result in a message, so several results have to be several
    // messages or the extras vanish silently.
    for (final result in request.toolResults) {
      session.addMessage(toLlamaMessage(LlmChatMessage.tool(result)));
    }

    final parts = <LlamaContentPart>[
      if (request.message != null && request.message!.isNotEmpty)
        LlamaTextContent(request.message!),
      ...request.attachments,
    ];

    final before = List<LlamaChatMessage>.of(session.history);
    final text = StringBuffer();
    final thinking = StringBuffer();
    final toolCalls = ToolCallAccumulator();
    String? finishReason;
    var checkedFit = false;
    var failedFit = false;

    final finished = Completer<void>();
    void settle() {
      if (!finished.isCompleted) finished.complete();
    }

    /// Fails the turn before the caller consumes output built on a prompt that
    /// llamadart already knows did not fit.
    bool overflowed() {
      if (checkedFit) return failedFit;
      checkedFit = true;
      failedFit =
          !session.lastRequestFitContext &&
          entry.options.overflowPolicy == ContextOverflowPolicy.fail;
      return failedFit;
    }

    entry.subscription = session
        .create(
          parts,
          params: overrides?.applyTo(_baseParams()) ?? _baseParams(),
          tools: overrides?.tools?.map((t) => t.toToolDefinition()).toList(),
          toolChoice: overrides?.toolChoice,
          parallelToolCalls: overrides?.parallelToolCalls ?? false,
          enableThinking: overrides?.enableThinking ?? _enableThinkingDefault(),
          continuesPreviousTurn: request.toolResults.isNotEmpty,
        )
        .listen(
          (chunk) {
            if (overflowed()) {
              entry.subscription?.cancel();
              entry.subscription = null;
              if (!controller.isClosed) {
                controller.addError(
                  LlmContextOverflowException(
                    'The conversation no longer fits the context window, even '
                    'after dropping older turns.',
                  ),
                );
              }
              settle();
              return;
            }

            final choice = chunk.choices.firstOrNull;
            finishReason = choice?.finishReason ?? finishReason;

            final deltas = choice?.delta.toolCalls;
            if (deltas != null) toolCalls.add(deltas);

            final content = choice?.delta.content;
            if (content != null) {
              text.write(content);
              if (!controller.isClosed) {
                controller.add(GenerationEvent(text: content));
              }
            }

            final reasoning = choice?.delta.thinking;
            if (reasoning != null) {
              thinking.write(reasoning);
              if (!controller.isClosed) {
                controller.add(GenerationEvent(text: '', thinking: reasoning));
              }
            }
          },
          onDone: settle,
          onError: (Object error, StackTrace stackTrace) {
            // ChatSession appends its assistant message only after a clean
            // finish, so an error leaves a user turn with no reply.
            _repairHistory(session, text, thinking);
            if (!controller.isClosed) controller.addError(error, stackTrace);
            settle();
          },
        );

    await finished.future;

    final wasCancelled =
        entry.subscription == null && !controller.isClosed && !failedFit;
    entry.subscription = null;

    if (failedFit) {
      await controller.close();
      return;
    }

    if (wasCancelled) {
      _repairHistory(session, text, thinking);
      await controller.close();
      return;
    }

    if (!entry.options.keepThinkingInHistory) {
      _stripThinking(session);
    }

    if (!controller.isClosed) {
      controller.add(
        GenerationEvent(
          text: '',
          isFinal: true,
          finishReason: finishReason,
          toolCalls: toolCalls.build(),
          perf: await _readPerf(),
          fitContext: session.lastRequestFitContext,
          dropped: _diffRemoved(before, session.history),
        ),
      );
    }
    await controller.close();
  }

  /// Appends the assistant turn llamadart skipped.
  ///
  /// `ChatSession.create` adds it after its last yield, so cancelling or
  /// erroring leaves the user message in history with no reply — and the next
  /// turn would render two user messages in a row. There is no removal API, so
  /// the repair is to append what was produced, even if that is nothing.
  void _repairHistory(
    ChatSessionLike session,
    StringBuffer text,
    StringBuffer thinking,
  ) {
    session.addMessage(
      LlamaChatMessage.withContent(
        role: LlamaChatRole.assistant,
        content: [
          if (thinking.isNotEmpty) LlamaThinkingContent(thinking.toString()),
          LlamaTextContent(text.toString()),
        ],
      ),
    );
  }

  /// Drops reasoning from stored history.
  ///
  /// Whether a model's chat template renders `reasoning_content` at all is
  /// model-dependent, and re-sending reasoning costs context on every later
  /// turn. `ChatSession` keeps its history private, so the only way to rewrite
  /// it is reset + re-add. Histories are small; this is cheap.
  void _stripThinking(ChatSessionLike session) {
    final history = session.history;
    if (!history.any((m) => m.parts.any((p) => p is LlamaThinkingContent))) {
      return;
    }

    final rebuilt = [
      for (final message in history)
        LlamaChatMessage.withContent(
          role: message.role,
          content: [
            ...message.parts.whereType<LlamaContentPart>().where(
              (p) => p is! LlamaThinkingContent,
            ),
          ],
          continuesPreviousTurn: message.continuesPreviousTurn,
        ),
    ];

    session.reset(keepSystemPrompt: true);
    for (final message in rebuilt) {
      session.addMessage(message);
    }
  }

  /// Messages that were in history before the turn and are gone after it.
  ///
  /// `ChatSession._enforceContextLimit` only removes entries and `create` only
  /// appends, both order-preserving, so a linear identity merge finds exactly
  /// what was dropped. If a future llamadart rewrites history instead, the
  /// merge stops matching — report nothing rather than claim the whole
  /// conversation was deleted.
  static List<LlmChatMessage> _diffRemoved(
    List<LlamaChatMessage> before,
    List<LlamaChatMessage> after,
  ) {
    if (before.isEmpty) return const [];

    final removed = <LlamaChatMessage>[];
    var index = 0;
    for (final message in before) {
      if (index < after.length && identical(after[index], message)) {
        index++;
      } else {
        removed.add(message);
      }
    }

    if (removed.length == before.length && after.isNotEmpty) {
      return const []; // Identities no longer line up; do not guess.
    }
    return removed.map(fromLlamaMessage).toList();
  }
}

/// Routes a non-streaming session operation by name.
///
/// Mirrors `dispatchEngineCall`: one place to add an operation, and the
/// isolate backend reaches it through the worker port.
Object? dispatchSessionCall(
  SessionRegistry registry,
  String method,
  Map<String, dynamic> args,
) {
  final id = args['id'] as String;
  switch (method) {
    case 'open':
      registry.open(
        id,
        args['options'] as SessionOptions,
        history: (args['history'] as List?)?.cast<LlmChatMessage>(),
      );
      return null;
    case 'history':
      return registry.history(id);
    case 'restore':
      registry.restore(id, (args['history'] as List).cast<LlmChatMessage>());
      return null;
    case 'reset':
      registry.reset(
        id,
        keepSystemPrompt: args['keepSystemPrompt'] as bool? ?? true,
      );
      return null;
    case 'setSystemPrompt':
      registry.setSystemPrompt(id, args['value'] as String?);
      return null;
    case 'cancel':
      registry.cancel(id);
      return null;
    case 'close':
      registry.close(id);
      return null;
    default:
      throw UnsupportedError('Unknown session call: $method');
  }
}
