// lib/src/models/llm_model_isolated.dart
import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:llamadart/llamadart.dart';

import '../core/backend_perf.dart';
import '../core/chat_message.dart';
import '../core/conversation.dart';
import '../core/conversation_types.dart';
import '../core/generation_event.dart';
import '../core/generation_overrides.dart';
import '../core/llm_config.dart';
import '../core/engine_rpc.dart';
import '../core/llm_errors.dart';
import '../core/model_diagnostics.dart';
import '../core/model_params_builder.dart';
import '../core/session_ops.dart';
import '../core/performance_metrics.dart';
import '../core/streaming_result.dart';
import '../core/tools.dart';
import 'llm_model_base.dart';

/// How long [LlmModelIsolated.dispose] waits for the worker to confirm that
/// llama.cpp released its native handles before the isolate is killed.
const Duration _disposeTimeout = Duration(seconds: 5);

// ── Worker Isolate entry point ─────────────────────────────────────────────

Future<void> _llamaIsolateWorkerMain(Map<String, dynamic> args) async {
  final LlmConfig config = args['config'] as LlmConfig;
  final SendPort mainPort = args['sendPort'] as SendPort;

  // The worker starts with an engine but no model: loading is a message, so a
  // model can be swapped with `unload` + `load` without respawning the isolate
  // and re-initializing the llama.cpp backend.
  final engine = LlamaEngine(LlamaBackend());
  final baseParams = buildGenerationParams(config);

  final receivePort = ReceivePort();
  mainPort.send({'type': 'ready', 'port': receivePort.sendPort});

  StreamSubscription<LlamaCompletionChunk>? genSubscription;
  // Releases the queue slot held by the running generation. Cancellation does
  // not fire onDone, so cancelling has to settle this explicitly.
  void Function()? genComplete;

  // llama.cpp has no "forget the prompt cache now" call. `clean` raises this
  // flag and the next generation runs with `reusePromptPrefix: false`, which
  // clears context memory and the cached prompt tokens before ingesting.
  var pendingPrefixInvalidation = false;

  // llamadart's model lifecycle mutex is non-reentrant: overlapping load /
  // unload / dispose throw LlamaStateException. Generation joins the same
  // queue so a load never lands mid-stream.
  var queue = Future<void>.value();
  Future<T> serialized<T>(Future<T> Function() action) {
    final completer = Completer<T>();
    queue = queue.then((_) async {
      try {
        completer.complete(await action());
      } catch (error, stackTrace) {
        completer.completeError(error, stackTrace);
      }
    });
    return completer.future;
  }

  // Conversations live here, next to the engine: ChatSession holds a
  // LlamaEngine, so it cannot be sent across the port.
  final sessions = SessionRegistry(
    createSession: (options) => RealChatSession(
      ChatSession(
        engine,
        maxContextTokens: options.maxContextTokens,
        systemPrompt: options.systemPrompt,
      ),
    ),
    baseParams: () => baseParams,
    enableThinkingDefault: () => config.enableThinkingDefault,
    readPerf: () => readBackendPerf(engine),
    // Conversation turns share the worker's queue with one-shot generation and
    // with model lifecycle operations.
    serialize: (action) => serialized(action),
  );

  // Stops an in-flight generation *outside* the queue, so unload and dispose
  // interrupt it instead of waiting behind it.
  Future<void> abortGeneration() async {
    final sub = genSubscription;
    genSubscription = null;
    if (sub == null) return;
    engine.cancelGeneration();
    await sub.cancel();
    genComplete?.call();
  }

  await for (final message in receivePort) {
    if (message is! Map<String, dynamic>) continue;

    switch (message['type'] as String?) {
      case 'load':
        final replyPort = message['replyPort'] as SendPort;
        try {
          await serialized(() async {
            await engine.loadModel(
              message['modelPath'] as String,
              modelParams: buildModelParams(config),
            );
            if (config.mmprojPath != null) {
              await engine.loadMultimodalProjector(config.mmprojPath!);
            }
          });
          replyPort.send({'type': 'ok'});
        } catch (e) {
          replyPort.send({'type': 'error', ...encodeError(e)});
        }

      case 'unload':
        final replyPort = message['replyPort'] as SendPort;
        await abortGeneration();
        sessions.closeAll();
        try {
          await serialized(engine.unloadModel);
          pendingPrefixInvalidation = false;
          replyPort.send({'type': 'ok'});
        } catch (e) {
          replyPort.send({'type': 'error', ...encodeError(e)});
        }

      case 'clean':
        pendingPrefixInvalidation = true;
        if (message['resetConversations'] as bool? ?? true) {
          sessions.resetAll();
        }
        (message['replyPort'] as SendPort?)?.send({'type': 'ok'});

      case 'session':
        final replyPort = message['replyPort'] as SendPort;
        try {
          replyPort.send({
            'type': 'ok',
            'value': dispatchSessionCall(
              sessions,
              message['method'] as String,
              (message['args'] as Map?)?.cast<String, dynamic>() ?? const {},
            ),
          });
        } catch (e) {
          replyPort.send({'type': 'error', ...encodeError(e)});
        }

      case 'session_turn':
        final streamPort = message['streamPort'] as SendPort;
        sessions
            .turn(message['id'] as String, message['request'] as TurnRequest)
            .listen(
              (event) => streamPort.send({'type': 'event', 'event': event}),
              onDone: () => streamPort.send({'type': 'done'}),
              onError: (Object e) =>
                  streamPort.send({'type': 'error', ...encodeError(e)}),
            );

      case 'generate':
        final prompt = message['prompt'] as String;
        final systemPrompt = message['systemPrompt'] as String?;
        final attachments = (message['attachments'] as List?)
            ?.cast<LlamaContentPart>();
        final overrides = message['overrides'] as GenerationOverrides?;
        final streamPort = message['streamPort'] as SendPort;

        var params = overrides?.applyTo(baseParams) ?? baseParams;
        if (pendingPrefixInvalidation) {
          params = params.copyWith(reusePromptPrefix: false);
          pendingPrefixInvalidation = false;
        }
        final enableThinking =
            overrides?.enableThinking ?? config.enableThinkingDefault;

        // Not awaited: the queued action outlives this message handler, so the
        // worker keeps servicing `cancel` and `unload` while tokens stream.
        unawaited(
          serialized(() async {
            String? finishReason;
            final toolCalls = ToolCallAccumulator();
            final finished = Completer<void>();

            void complete() {
              if (!finished.isCompleted) finished.complete();
            }

            genComplete = complete;

            genSubscription = engine
                .create(
                  buildMessages(
                    prompt,
                    systemPrompt: systemPrompt,
                    attachments: attachments,
                  ),
                  params: params,
                  enableThinking: enableThinking,
                  tools: overrides?.tools
                      ?.map((t) => t.toToolDefinition())
                      .toList(),
                  toolChoice: overrides?.toolChoice,
                  parallelToolCalls: overrides?.parallelToolCalls ?? false,
                  responseFormat: overrides?.responseFormat,
                )
                .listen(
                  (chunk) {
                    final choice = chunk.choices.firstOrNull;
                    finishReason = choice?.finishReason ?? finishReason;

                    final deltas = choice?.delta.toolCalls;
                    if (deltas != null) toolCalls.add(deltas);

                    final text = choice?.delta.content;
                    if (text != null) {
                      streamPort.send({'type': 'token', 'text': text});
                    }
                    final thinking = choice?.delta.thinking;
                    if (thinking != null) {
                      streamPort.send({'type': 'token', 'thinking': thinking});
                    }
                  },
                  onDone: () async {
                    genSubscription = null;
                    streamPort.send({
                      'type': 'done',
                      if (finishReason != null) 'finishReason': finishReason,
                      'perf': await readBackendPerf(engine),
                      if (!toolCalls.isEmpty)
                        'toolCalls': toolCalls
                            .build()
                            .map((c) => c.toMap())
                            .toList(),
                    });
                    complete();
                  },
                  onError: (Object e) {
                    genSubscription = null;
                    streamPort.send({'type': 'error', ...encodeError(e)});
                    complete();
                  },
                );

            await finished.future;
            genComplete = null;
          }),
        );

      case 'call':
        final replyPort = message['replyPort'] as SendPort;
        try {
          final value = await dispatchEngineCall(
            engine,
            message['method'] as String,
            (message['args'] as Map?)?.cast<String, dynamic>() ?? const {},
          );
          replyPort.send({'type': 'ok', 'value': value});
        } catch (e) {
          replyPort.send({'type': 'error', ...encodeError(e)});
        }

      case 'cancel':
        await abortGeneration();

      case 'dispose':
        await abortGeneration();
        sessions.closeAll();
        await engine.dispose();
        (message['replyPort'] as SendPort?)?.send({'type': 'disposed'});
        receivePort.close();
        return;
    }
  }
}

// ── LlmModelIsolated ───────────────────────────────────────────────────────

class LlmModelIsolated extends LlmModelBase {
  @override
  final LlmConfig config;
  Isolate? _isolate;
  SendPort? _workerPort;

  LlmModelIsolated(this.config);

  @override
  Future<void> loadModel(String localPath) async {
    checkNotDisposed();
    config.validate();

    if (!File(localPath).existsSync()) {
      throw FileSystemException('File not found', localPath);
    }

    await _ensureWorker();
    await _request('load', {
      'modelPath': localPath,
    }, context: 'Model load failed');
    markAsInitialized();
  }

  /// Spawns the worker isolate on first use. The worker comes up with a
  /// llama.cpp backend but no model, so later loads reuse it.
  Future<void> _ensureWorker() async {
    if (_workerPort != null) return;

    final initPort = ReceivePort();
    _isolate = await Isolate.spawn(_llamaIsolateWorkerMain, {
      'config': config,
      'sendPort': initPort.sendPort,
    }, debugName: 'mt_llmkit_LlamaWorker');

    final initMsg = await initPort.first as Map<String, dynamic>;
    initPort.close();
    _workerPort = initMsg['port'] as SendPort;
  }

  /// Sends one request-reply message to the worker.
  Future<void> _request(
    String type,
    Map<String, dynamic> args, {
    required String context,
  }) async {
    final replyPort = ReceivePort();
    _workerPort!.send({...args, 'type': type, 'replyPort': replyPort.sendPort});

    final reply = (await replyPort.first as Map).cast<String, dynamic>();
    replyPort.close();

    if (reply['type'] == 'error') {
      throw decodeError(reply, context: context);
    }
  }

  int _nextSessionId = 0;

  @override
  Future<Conversation> startConversation({
    String? systemPrompt,
    int? maxContextTokens,
    List<LlmChatMessage>? history,
    ContextOverflowPolicy overflowPolicy = ContextOverflowPolicy.allow,
    bool keepThinkingInHistory = false,
  }) async {
    checkInitialized();
    final id = 's${_nextSessionId++}';
    await _sessionCall('open', {
      'id': id,
      'options': SessionOptions(
        systemPrompt: systemPrompt,
        maxContextTokens: maxContextTokens,
        overflowPolicy: overflowPolicy,
        keepThinkingInHistory: keepThinkingInHistory,
      ),
      if (history != null) 'history': history,
    });
    return Conversation(id, _IsolateConversationTransport(this));
  }

  /// One request-reply session operation on the worker's registry.
  Future<Object?> _sessionCall(String method, Map<String, dynamic> args) async {
    final replyPort = ReceivePort();
    _workerPort!.send({
      'type': 'session',
      'method': method,
      'args': args,
      'replyPort': replyPort.sendPort,
    });

    final reply = (await replyPort.first as Map).cast<String, dynamic>();
    replyPort.close();

    if (reply['type'] == 'error') {
      throw decodeError(reply, context: 'Conversation "$method" failed');
    }
    return reply['value'];
  }

  /// Bridges one conversation turn from the worker's registry.
  Stream<GenerationEvent> _sessionTurn(String id, TurnRequest request) {
    final controller = StreamController<GenerationEvent>();
    final replyPort = ReceivePort();

    _workerPort!.send({
      'type': 'session_turn',
      'id': id,
      'request': request,
      'streamPort': replyPort.sendPort,
    });

    final sub = replyPort.listen((dynamic msg) {
      if (msg is! Map<String, dynamic>) return;
      switch (msg['type'] as String?) {
        case 'event':
          if (!controller.isClosed) {
            controller.add(msg['event'] as GenerationEvent);
          }
        case 'done':
          replyPort.close();
          if (!controller.isClosed) controller.close();
        case 'error':
          replyPort.close();
          if (!controller.isClosed) {
            controller.addError(decodeError(msg, context: 'Conversation turn'));
            controller.close();
          }
      }
    });

    controller.onCancel = () {
      sub.cancel();
      replyPort.close();
    };

    return controller.stream;
  }

  @override
  Future<void> unload() async {
    checkNotDisposed();
    if (_workerPort == null) return;
    await _request('unload', const {}, context: 'Model unload failed');
    markAsUnloaded();
  }

  @override
  Future<void> clean({bool resetConversations = true}) async {
    checkInitialized();
    await _request('clean', {
      'resetConversations': resetConversations,
    }, context: 'clean() failed');
  }

  // Bridges worker SendPort messages into a stream of events. The terminal
  // event carries the generation's finish reason and backend perf counters.
  // No generation tracking — callers manage isGenerating state.
  Stream<GenerationEvent> _rawWorkerStream(Map<String, dynamic> message) {
    final controller = StreamController<GenerationEvent>();
    final replyPort = ReceivePort();
    var finished = false;

    _workerPort!.send({...message, 'streamPort': replyPort.sendPort});

    final sub = replyPort.listen((dynamic msg) {
      if (msg is! Map<String, dynamic>) return;
      switch (msg['type'] as String?) {
        case 'token':
          if (!controller.isClosed) {
            controller.add(
              GenerationEvent(
                text: msg['text'] as String? ?? '',
                thinking: msg['thinking'] as String?,
              ),
            );
          }
        case 'done':
          finished = true;
          replyPort.close();
          if (!controller.isClosed) {
            controller.add(
              GenerationEvent(
                text: '',
                isFinal: true,
                finishReason: msg['finishReason'] as String?,
                perf: (msg['perf'] as Map?)?.cast<String, dynamic>(),
                toolCalls: [
                  for (final call in (msg['toolCalls'] as List?) ?? const [])
                    LlmToolCall.fromMap((call as Map).cast<String, dynamic>()),
                ],
              ),
            );
            controller.close();
          }
        case 'error':
          finished = true;
          replyPort.close();
          if (!controller.isClosed) {
            controller.addError(decodeError(msg, context: 'Worker error'));
            controller.close();
          }
      }
    });

    controller.onCancel = () {
      sub.cancel();
      replyPort.close();
      // Only interrupt a generation that is still running: cancelGeneration()
      // is engine-wide, so cancelling after a clean finish would abort an
      // unrelated request when maxParallelSequences > 1.
      if (!finished) _workerPort?.send({'type': 'cancel'});
    };

    return controller.stream;
  }

  // Wraps _rawWorkerStream with isGenerating tracking for sendPrompt().
  Stream<GenerationEvent> _trackedStream(Map<String, dynamic> message) async* {
    markGenerationStart();
    try {
      yield* _rawWorkerStream(message);
    } finally {
      markGenerationEnd();
    }
  }

  Map<String, dynamic> _buildMessage(
    String prompt, {
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) => {
    'type': 'generate',
    'prompt': prompt,
    if (systemPrompt != null) 'systemPrompt': systemPrompt,
    if (attachments != null && attachments.isNotEmpty)
      'attachments': attachments,
    if (overrides != null) 'overrides': overrides,
  };

  @override
  Stream<String> sendPrompt(
    String prompt, {
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) {
    checkInitialized();
    return _trackedStream(
      _buildMessage(
        prompt,
        systemPrompt: systemPrompt,
        attachments: attachments,
        overrides: overrides,
      ),
    ).where((e) => !e.isFinal && e.text.isNotEmpty).map((e) => e.text);
  }

  @override
  Future<String> sendPromptComplete(
    String prompt, {
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) async {
    checkInitialized();
    markGenerationStart();
    try {
      final buffer = StringBuffer();
      await for (final event in _rawWorkerStream(
        _buildMessage(
          prompt,
          systemPrompt: systemPrompt,
          attachments: attachments,
          overrides: overrides,
        ),
      )) {
        buffer.write(event.text);
      }
      return buffer.toString();
    } finally {
      markGenerationEnd();
    }
  }

  @override
  Stream<StreamingChunk> sendPromptStream(
    String prompt, {
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) async* {
    checkInitialized();

    final startTime = DateTime.now();
    int totalTokenCount = 0;

    markGenerationStart();
    try {
      await for (final event in _rawWorkerStream(
        _buildMessage(
          prompt,
          systemPrompt: systemPrompt,
          attachments: attachments,
          overrides: overrides,
        ),
      )) {
        if (event.isFinal) {
          yield StreamingChunk(
            text: '',
            metrics: finalMetrics(
              perf: event.perf,
              fallbackTokenCount: totalTokenCount,
              startTime: startTime,
              endTime: DateTime.now(),
            ),
            isFinal: true,
            finishReason: event.finishReason,
            toolCalls: event.toolCalls,
          );
          continue;
        }

        totalTokenCount += 1;
        yield StreamingChunk(
          text: event.text,
          thinking: event.thinking,
          metrics: PerformanceMetrics.fromGeneration(
            tokenCount: totalTokenCount,
            startTime: startTime,
            endTime: DateTime.now(),
          ),
          isFinal: false,
        );
      }
    } finally {
      markGenerationEnd();
    }
  }

  @override
  Future<void> dispose() async {
    final workerPort = _workerPort;
    _workerPort = null;

    if (workerPort != null) {
      // Wait for the worker to confirm llama.cpp released its handles before
      // killing the isolate — killing mid-teardown races token release,
      // worker shutdown and engine deletion.
      final ackPort = ReceivePort();
      workerPort.send({'type': 'dispose', 'replyPort': ackPort.sendPort});
      try {
        await ackPort.first.timeout(_disposeTimeout);
      } catch (_) {
        // A worker that is wedged or already gone must not block teardown.
      } finally {
        ackPort.close();
      }
    }

    _isolate?.kill(priority: Isolate.beforeNextEvent);
    _isolate = null;
    markAsDisposed();
  }

  /// Round-trips one non-streaming engine call through the worker.
  Future<Object?> _call(String method, [Map<String, dynamic> args = const {}]) {
    checkInitialized();
    final replyPort = ReceivePort();
    _workerPort!.send({
      'type': 'call',
      'method': method,
      'args': args,
      'replyPort': replyPort.sendPort,
    });
    return replyPort.first.then((dynamic response) {
      replyPort.close();
      final map = (response as Map).cast<String, dynamic>();
      if (map['type'] == 'error') {
        throw decodeError(map, context: 'Engine call "$method" failed');
      }
      return map['value'];
    });
  }

  @override
  Future<List<int>> tokenize(String text, {bool addSpecial = true}) async =>
      ((await _call('tokenize', {'text': text, 'addSpecial': addSpecial}))
              as List)
          .cast<int>();

  @override
  Future<String> detokenize(List<int> tokens, {bool special = false}) async =>
      (await _call('detokenize', {'tokens': tokens, 'special': special}))
          as String;

  @override
  Future<int> countTokens(String text) async =>
      (await _call('countTokens', {'text': text})) as int;

  @override
  Future<int> contextSize() async => (await _call('contextSize')) as int;

  @override
  Future<Map<String, String>> metadata() async =>
      ((await _call('metadata')) as Map).cast<String, String>();

  @override
  Future<bool> get supportsStatePersistence async =>
      (await _call('supportsStatePersistence')) as bool;

  @override
  Future<bool> saveState(String path, {required List<int> tokens}) async =>
      (await _call('saveState', {'path': path, 'tokens': tokens})) as bool;

  @override
  Future<List<int>> loadState(String path, {int? tokenCapacity}) async =>
      ((await _call('loadState', {
                'path': path,
                if (tokenCapacity != null) 'tokenCapacity': tokenCapacity,
              }))
              as List)
          .cast<int>();

  @override
  Future<void> setLora(String path, {double scale = 1.0}) async =>
      _call('setLora', {'path': path, 'scale': scale});

  @override
  Future<void> removeLora(String path) async =>
      _call('removeLora', {'path': path});

  @override
  Future<void> clearLoras() async => _call('clearLoras');

  @override
  Future<ModelDiagnostics> diagnostics() async =>
      (await _call('diagnostics')) as ModelDiagnostics;
}

/// Reaches the [SessionRegistry] living in the worker isolate.
class _IsolateConversationTransport implements ConversationTransport {
  final LlmModelIsolated _model;

  _IsolateConversationTransport(this._model);

  @override
  Stream<GenerationEvent> turn(String id, TurnRequest request) =>
      _model._sessionTurn(id, request);

  @override
  Future<List<LlmChatMessage>> history(String id) async =>
      ((await _model._sessionCall('history', {'id': id})) as List)
          .cast<LlmChatMessage>();

  @override
  Future<void> restore(String id, List<LlmChatMessage> messages) async =>
      _model._sessionCall('restore', {'id': id, 'history': messages});

  @override
  Future<void> reset(String id, {bool keepSystemPrompt = true}) async => _model
      ._sessionCall('reset', {'id': id, 'keepSystemPrompt': keepSystemPrompt});

  @override
  Future<void> setSystemPrompt(String id, String? value) async =>
      _model._sessionCall('setSystemPrompt', {'id': id, 'value': value});

  @override
  Future<void> cancel(String id) async =>
      _model._sessionCall('cancel', {'id': id});

  @override
  Future<void> close(String id) async =>
      _model._sessionCall('close', {'id': id});
}
