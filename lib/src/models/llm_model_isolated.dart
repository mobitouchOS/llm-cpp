// lib/src/models/llm_model_isolated.dart
import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:llamadart/llamadart.dart';

import '../core/backend_perf.dart';
import '../core/generation_event.dart';
import '../core/llm_config.dart';
import '../core/llm_errors.dart';
import '../core/performance_metrics.dart';
import '../core/streaming_result.dart';
import 'llm_model_base.dart';

/// How long [LlmModelIsolated.dispose] waits for the worker to confirm that
/// llama.cpp released its native handles before the isolate is killed.
const Duration _disposeTimeout = Duration(seconds: 5);

// ── Worker Isolate entry point ─────────────────────────────────────────────

Future<void> _llamaIsolateWorkerMain(Map<String, dynamic> args) async {
  final String modelPath = args['modelPath'] as String;
  final String? mmprojPath = args['mmprojPath'] as String?;
  final int contextSize = args['contextSize'] as int;
  final int gpuLayers = args['gpuLayers'] as int;
  final int batchSize = args['batchSize'] as int;
  final int numberOfThreads = args['numberOfThreads'] as int;
  final int numberOfThreadsBatch = args['numberOfThreadsBatch'] as int;
  final int microBatchSize = args['microBatchSize'] as int;
  final int maxParallelSequences = args['maxParallelSequences'] as int;
  final String? chatTemplate = args['chatTemplate'] as String?;
  final List<LoraAdapterConfig> loras = (args['loras'] as List)
      .cast<Map>()
      .map(
        (m) => LoraAdapterConfig(
          path: m['path'] as String,
          scale: (m['scale'] as num).toDouble(),
        ),
      )
      .toList();
  final int maxTokens = args['maxTokens'] as int;
  final double temp = (args['temp'] as num).toDouble();
  final int topK = args['topK'] as int;
  final double topP = (args['topP'] as num).toDouble();
  final double minP = (args['minP'] as num).toDouble();
  final double penalty = (args['penalty'] as num).toDouble();
  final int? seed = args['seed'] as int?;
  final List<String> stopSequences = (args['stopSequences'] as List)
      .cast<String>();
  final String? grammar = args['grammar'] as String?;
  final bool grammarLazy = args['grammarLazy'] as bool;
  final List<GenerationGrammarTrigger> grammarTriggers =
      (args['grammarTriggers'] as List)
          .cast<Map>()
          .map(
            (m) => GenerationGrammarTrigger(
              type: m['type'] as int,
              value: m['value'] as String,
              token: m['token'] as int?,
            ),
          )
          .toList();
  final List<String> preservedTokens = (args['preservedTokens'] as List)
      .cast<String>();
  final String grammarRoot = args['grammarRoot'] as String;
  final bool reusePromptPrefix = args['reusePromptPrefix'] as bool;
  final int streamBatchTokenThreshold =
      args['streamBatchTokenThreshold'] as int;
  final int streamBatchByteThreshold = args['streamBatchByteThreshold'] as int;
  final String gpuBackendName = args['gpuBackend'] as String;
  final GpuBackend gpuBackend = GpuBackend.values.firstWhere(
    (e) => e.name == gpuBackendName,
    orElse: () => GpuBackend.auto,
  );
  final SendPort mainPort = args['sendPort'] as SendPort;

  final engine = LlamaEngine(LlamaBackend());
  try {
    await engine.loadModel(
      modelPath,
      modelParams: ModelParams(
        contextSize: contextSize,
        gpuLayers: gpuLayers,
        batchSize: batchSize,
        numberOfThreads: numberOfThreads,
        numberOfThreadsBatch: numberOfThreadsBatch,
        microBatchSize: microBatchSize,
        maxParallelSequences: maxParallelSequences,
        loras: loras,
        chatTemplate: chatTemplate,
        preferredBackend: gpuBackend,
      ),
    );
    if (mmprojPath != null) {
      await engine.loadMultimodalProjector(mmprojPath);
    }
  } catch (e) {
    mainPort.send({'type': 'error', ...encodeError(e)});
    return;
  }

  final genParams = GenerationParams(
    maxTokens: maxTokens,
    temp: temp,
    topK: topK,
    topP: topP,
    minP: minP,
    penalty: penalty,
    seed: seed,
    stopSequences: stopSequences,
    grammar: grammar,
    grammarLazy: grammarLazy,
    grammarTriggers: grammarTriggers,
    preservedTokens: preservedTokens,
    grammarRoot: grammarRoot,
    reusePromptPrefix: reusePromptPrefix,
    streamBatchTokenThreshold: streamBatchTokenThreshold,
    streamBatchByteThreshold: streamBatchByteThreshold,
  );

  final receivePort = ReceivePort();
  mainPort.send({'type': 'ready', 'port': receivePort.sendPort});

  StreamSubscription<LlamaCompletionChunk>? genSubscription;

  await for (final message in receivePort) {
    if (message is! Map<String, dynamic>) continue;

    switch (message['type'] as String?) {
      case 'generate':
        final prompt = message['prompt'] as String;
        final images = (message['images'] as List?)?.cast<LlamaImageContent>();
        final streamPort = message['streamPort'] as SendPort;

        final msg = (images != null && images.isNotEmpty)
            ? LlamaChatMessage.withContent(
                role: LlamaChatRole.user,
                content: [LlamaTextContent(prompt), ...images],
              )
            : LlamaChatMessage.fromText(role: LlamaChatRole.user, text: prompt);

        String? finishReason;

        genSubscription = engine
            .create([msg], params: genParams)
            .listen(
              (chunk) {
                final choice = chunk.choices.firstOrNull;
                finishReason = choice?.finishReason ?? finishReason;
                final text = choice?.delta.content;
                if (text != null) {
                  streamPort.send({'type': 'token', 'text': text});
                }
              },
              onDone: () async {
                genSubscription = null;
                streamPort.send({
                  'type': 'done',
                  if (finishReason != null) 'finishReason': finishReason,
                  'perf': await readBackendPerf(engine),
                });
              },
              onError: (Object e) {
                genSubscription = null;
                streamPort.send({'type': 'error', ...encodeError(e)});
              },
            );

      case 'cancel':
        genSubscription?.cancel();
        genSubscription = null;
        engine.cancelGeneration();

      case 'dispose':
        await genSubscription?.cancel();
        genSubscription = null;
        await engine.dispose();
        (message['replyPort'] as SendPort?)?.send({'type': 'disposed'});
        receivePort.close();
        return;
    }
  }
}

// ── LlmModelIsolated ───────────────────────────────────────────────────────

class LlmModelIsolated extends LlmModelBase {
  final LlmConfig config;
  Isolate? _isolate;
  SendPort? _workerPort;

  LlmModelIsolated(this.config);

  @override
  Future<void> loadModel(String localPath) async {
    checkNotDisposed();

    if (!File(localPath).existsSync()) {
      throw FileSystemException('File not found', localPath);
    }

    final initPort = ReceivePort();
    _isolate = await Isolate.spawn(_llamaIsolateWorkerMain, {
      'modelPath': localPath,
      'mmprojPath': config.mmprojPath,
      'contextSize': config.nCtxDefault,
      'gpuLayers': config.nGpuLayersDefault,
      'batchSize': config.nBatchDefault,
      'numberOfThreads': config.nThreadsDefault,
      'numberOfThreadsBatch': config.numberOfThreadsBatchDefault,
      'microBatchSize': config.microBatchSizeDefault,
      'maxParallelSequences': config.maxParallelSequencesDefault,
      'chatTemplate': config.chatTemplate,
      'loras': config.lorasDefault
          .map((l) => {'path': l.path, 'scale': l.scale})
          .toList(),
      'maxTokens': config.nPredictDefault,
      'temp': config.tempDefault,
      'topK': config.topKDefault,
      'topP': config.topPDefault,
      'minP': config.minPDefault,
      'penalty': config.penaltyRepeatDefault,
      'seed': config.seed,
      'stopSequences': config.stopSequencesDefault,
      'grammar': config.grammar,
      'grammarLazy': config.grammarLazyDefault,
      'grammarTriggers': config.grammarTriggersDefault
          .map(
            (t) => {
              'type': t.type,
              'value': t.value,
              if (t.token != null) 'token': t.token,
            },
          )
          .toList(),
      'preservedTokens': config.preservedTokensDefault,
      'grammarRoot': config.grammarRootDefault,
      'reusePromptPrefix': config.reusePromptPrefixDefault,
      'streamBatchTokenThreshold': config.streamBatchTokenThresholdDefault,
      'streamBatchByteThreshold': config.streamBatchByteThresholdDefault,
      'gpuBackend': config.gpuBackendDefault.name,
      'sendPort': initPort.sendPort,
    }, debugName: 'llmcpp_LlamaWorker');

    final initMsg = await initPort.first as Map<String, dynamic>;
    initPort.close();

    if (initMsg['type'] == 'error') {
      _isolate?.kill();
      _isolate = null;
      throw decodeError(initMsg, context: 'LlmModelIsolated init failed');
    }

    _workerPort = initMsg['port'] as SendPort;
    markAsInitialized();
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
            controller.add(GenerationEvent(text: msg['text'] as String));
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
    List<LlamaImageContent>? images,
  }) => {
    'type': 'generate',
    'prompt': prompt,
    if (images != null && images.isNotEmpty) 'images': images,
  };

  @override
  Stream<String> sendPrompt(String prompt, {List<LlamaImageContent>? images}) {
    checkInitialized();
    return _trackedStream(
      _buildMessage(prompt, images: images),
    ).where((e) => !e.isFinal).map((e) => e.text);
  }

  @override
  Future<String> sendPromptComplete(
    String prompt, {
    List<LlamaImageContent>? images,
  }) async {
    checkInitialized();
    markGenerationStart();
    try {
      final buffer = StringBuffer();
      await for (final event in _rawWorkerStream(
        _buildMessage(prompt, images: images),
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
    List<LlamaImageContent>? images,
  }) async* {
    checkInitialized();

    final startTime = DateTime.now();
    int totalTokenCount = 0;

    markGenerationStart();
    try {
      await for (final event in _rawWorkerStream(
        _buildMessage(prompt, images: images),
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
          );
          continue;
        }

        totalTokenCount += 1;
        yield StreamingChunk(
          text: event.text,
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

  @override
  void clean() {
    checkInitialized();
    // create() is stateless in llamadart — no context to reset
  }
}
