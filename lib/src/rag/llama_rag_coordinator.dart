// lib/src/rag/llama_rag_coordinator.dart
//
// ARCHITECTURE — single isolate for both models
//
// llamadart resolves the NativeCallable isolate-crash issue internally.
// We still use one isolate for both models to:
// 1. Keep the UI thread free during inference
// 2. Avoid any residual resource contention between two LlamaEngine instances

import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:llamadart/llamadart.dart';

import '../core/backend_perf.dart';
import '../core/generation_event.dart';
import '../core/llm_config.dart';
import '../core/llm_errors.dart';
import '../core/llm_interface.dart';
import '../core/performance_metrics.dart';
import '../core/streaming_result.dart';
import 'embeddings/embedding_provider.dart';

/// How long [LlamaRagCoordinator.dispose] waits for the worker to confirm that
/// both engines released their native handles before the isolate is killed.
const Duration _disposeTimeout = Duration(seconds: 5);

// ── Worker isolate ────────────────────────────────────────────────────────────

Future<void> _llamaRagWorkerMain(Map<String, dynamic> args) async {
  final String embedModelPath = args['embedModelPath'] as String;
  final String genModelPath = args['genModelPath'] as String;
  final int embedContextSize = args['embedContextSize'] as int;
  final int genContextSize = args['genContextSize'] as int;
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
  final GpuBackend genGpuBackend = GpuBackend.values.firstWhere(
    (e) => e.name == gpuBackendName,
    orElse: () => GpuBackend.auto,
  );
  final int genGpuLayers = args['genGpuLayers'] as int;
  final SendPort mainPort = args['sendPort'] as SendPort;

  final embedEngine = LlamaEngine(LlamaBackend());
  try {
    await embedEngine.loadModel(
      embedModelPath,
      modelParams: ModelParams(
        contextSize: embedContextSize,
        gpuLayers: 0,
        batchSize: batchSize,
        numberOfThreads: numberOfThreads,
        preferredBackend: GpuBackend.cpu,
      ),
    );
  } catch (e) {
    mainPort.send({'type': 'error', 'phase': 'embed_init', ...encodeError(e)});
    return;
  }

  final genEngine = LlamaEngine(LlamaBackend());
  try {
    await genEngine.loadModel(
      genModelPath,
      modelParams: ModelParams(
        contextSize: genContextSize,
        gpuLayers: genGpuLayers,
        batchSize: batchSize,
        numberOfThreads: numberOfThreads,
        numberOfThreadsBatch: numberOfThreadsBatch,
        microBatchSize: microBatchSize,
        maxParallelSequences: maxParallelSequences,
        loras: loras,
        chatTemplate: chatTemplate,
        preferredBackend: genGpuBackend,
      ),
    );
  } catch (e) {
    await embedEngine.dispose();
    mainPort.send({'type': 'error', 'phase': 'gen_init', ...encodeError(e)});
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
      case 'embed':
        final text = message['text'] as String;
        final replyPort = message['replyPort'] as SendPort;
        try {
          final embedding = await embedEngine.embed(text);
          replyPort.send({'type': 'ok', 'embedding': embedding});
        } catch (e) {
          replyPort.send({'type': 'error', ...encodeError(e)});
        }

      case 'generate':
        final prompt = message['prompt'] as String;
        final streamPort = message['streamPort'] as SendPort;
        String? finishReason;

        genSubscription = genEngine
            .create([
              LlamaChatMessage.fromText(role: LlamaChatRole.user, text: prompt),
            ], params: genParams)
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
                  'perf': await readBackendPerf(genEngine),
                });
              },
              onError: (Object e) {
                genSubscription = null;
                streamPort.send({'type': 'error', ...encodeError(e)});
              },
            );

      case 'stop_generate':
        genSubscription?.cancel();
        genSubscription = null;
        genEngine.cancelGeneration();

      case 'dispose':
        await genSubscription?.cancel();
        genSubscription = null;
        await embedEngine.dispose();
        await genEngine.dispose();
        (message['replyPort'] as SendPort?)?.send({'type': 'disposed'});
        receivePort.close();
        return;
    }
  }
}

// ── Private provider implementations ─────────────────────────────────────────

class _CoordEmbeddingProvider implements EmbeddingProvider {
  final SendPort _workerPort;
  final int _dimensions;
  bool _isInitialized = true;

  _CoordEmbeddingProvider(this._workerPort, {required int dimensions})
    : _dimensions = dimensions;

  @override
  Future<void> initialize(Map<String, dynamic> config) async {
    _isInitialized = true;
  }

  @override
  Future<List<double>> embed(String text) async {
    final replyPort = ReceivePort();
    _workerPort.send({
      'type': 'embed',
      'text': _truncate(text),
      'replyPort': replyPort.sendPort,
    });
    final response = await replyPort.first as Map<String, dynamic>;
    replyPort.close();
    if (response['type'] == 'error') {
      throw decodeError(response, context: 'Embedding error');
    }
    final raw = response['embedding'] as List;
    return raw.map((e) => (e as num).toDouble()).toList();
  }

  @override
  Future<List<List<double>>> embedBatch(List<String> texts) async {
    final results = <List<double>>[];
    for (final text in texts) {
      results.add(await embed(text));
    }
    return results;
  }

  @override
  Future<void> dispose() async {
    _isInitialized = false;
  }

  @override
  int get dimensions => _dimensions;

  @override
  bool get isInitialized => _isInitialized;

  String _truncate(String text, {int maxChars = 2000}) {
    if (text.length <= maxChars) return text;
    final truncated = text.substring(0, maxChars);
    final lastSpace = truncated.lastIndexOf(' ');
    return lastSpace > maxChars * 0.8
        ? truncated.substring(0, lastSpace)
        : truncated;
  }
}

class _CoordPlugin implements LlmInterface {
  final SendPort _workerPort;
  bool _isGenerating = false;

  _CoordPlugin(this._workerPort);

  @override
  bool get isInitialized => true;

  @override
  bool get isGenerating => _isGenerating;

  @override
  Future<void> loadModel(String localPath) async {}

  @override
  Stream<String> sendPrompt(String prompt, {List<LlamaImageContent>? images}) =>
      _events(
        prompt,
        images: images,
      ).where((e) => !e.isFinal).map((e) => e.text);

  Stream<GenerationEvent> _events(
    String prompt, {
    List<LlamaImageContent>? images,
  }) {
    if (images != null && images.isNotEmpty) {
      throw UnsupportedError('Vision is not supported in the RAG pipeline.');
    }
    final controller = StreamController<GenerationEvent>();
    final replyPort = ReceivePort();
    var finished = false;

    _workerPort.send({
      'type': 'generate',
      'prompt': prompt,
      'streamPort': replyPort.sendPort,
    });

    _isGenerating = true;
    final sub = replyPort.listen((dynamic message) {
      if (message is! Map<String, dynamic>) return;
      switch (message['type'] as String?) {
        case 'token':
          if (!controller.isClosed) {
            controller.add(GenerationEvent(text: message['text'] as String));
          }
        case 'done':
          finished = true;
          replyPort.close();
          _isGenerating = false;
          if (!controller.isClosed) {
            controller.add(
              GenerationEvent(
                text: '',
                isFinal: true,
                finishReason: message['finishReason'] as String?,
                perf: (message['perf'] as Map?)?.cast<String, dynamic>(),
              ),
            );
            controller.close();
          }
        case 'error':
          finished = true;
          replyPort.close();
          _isGenerating = false;
          if (!controller.isClosed) {
            controller.addError(
              decodeError(message, context: 'Generation error'),
            );
            controller.close();
          }
      }
    });

    controller.onCancel = () {
      sub.cancel();
      replyPort.close();
      _isGenerating = false;
      if (!finished) _workerPort.send({'type': 'stop_generate'});
    };

    return controller.stream;
  }

  @override
  Future<String> sendPromptComplete(
    String prompt, {
    List<LlamaImageContent>? images,
  }) async {
    final buffer = StringBuffer();
    await for (final token in sendPrompt(prompt, images: images)) {
      buffer.write(token);
    }
    return buffer.toString();
  }

  @override
  Stream<StreamingChunk> sendPromptStream(
    String prompt, {
    List<LlamaImageContent>? images,
  }) async* {
    final startTime = DateTime.now();
    int totalTokenCount = 0;

    await for (final event in _events(prompt, images: images)) {
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
  }

  @override
  Future<void> dispose() async {}

  @override
  void clean() {}
}

// ── LlamaRagCoordinator ───────────────────────────────────────────────────────

class LlamaRagCoordinator {
  Isolate? _isolate;
  SendPort? _workerPort;
  late final EmbeddingProvider _embeddingProvider;
  late final LlmInterface _generationPlugin;

  LlamaRagCoordinator._();

  static Future<LlamaRagCoordinator> create({
    required String embedModelPath,
    required String genModelPath,
    LlmConfig genConfig = const LlmConfig(),
    int embedNCtx = 512,
  }) async {
    if (!File(embedModelPath).existsSync()) {
      throw FileSystemException(
        'Embedding model does not exist',
        embedModelPath,
      );
    }
    if (!File(genModelPath).existsSync()) {
      throw FileSystemException(
        'Generation model does not exist',
        genModelPath,
      );
    }

    final coordinator = LlamaRagCoordinator._();
    await coordinator._init(embedModelPath, genModelPath, genConfig, embedNCtx);
    return coordinator;
  }

  Future<void> _init(
    String embedModelPath,
    String genModelPath,
    LlmConfig genConfig,
    int embedNCtx,
  ) async {
    final initPort = ReceivePort();

    _isolate = await Isolate.spawn(_llamaRagWorkerMain, {
      'embedModelPath': embedModelPath,
      'genModelPath': genModelPath,
      'embedContextSize': embedNCtx,
      'genContextSize': genConfig.nCtxDefault,
      'batchSize': genConfig.nBatchDefault,
      'numberOfThreads': genConfig.nThreadsDefault,
      'numberOfThreadsBatch': genConfig.numberOfThreadsBatchDefault,
      'microBatchSize': genConfig.microBatchSizeDefault,
      'maxParallelSequences': genConfig.maxParallelSequencesDefault,
      'chatTemplate': genConfig.chatTemplate,
      'loras': genConfig.lorasDefault
          .map((l) => {'path': l.path, 'scale': l.scale})
          .toList(),
      'maxTokens': genConfig.nPredictDefault,
      'temp': genConfig.tempDefault,
      'topK': genConfig.topKDefault,
      'topP': genConfig.topPDefault,
      'minP': genConfig.minPDefault,
      'penalty': genConfig.penaltyRepeatDefault,
      'seed': genConfig.seed,
      'stopSequences': genConfig.stopSequencesDefault,
      'grammar': genConfig.grammar,
      'grammarLazy': genConfig.grammarLazyDefault,
      'grammarTriggers': genConfig.grammarTriggersDefault
          .map(
            (t) => {
              'type': t.type,
              'value': t.value,
              if (t.token != null) 'token': t.token,
            },
          )
          .toList(),
      'preservedTokens': genConfig.preservedTokensDefault,
      'grammarRoot': genConfig.grammarRootDefault,
      'reusePromptPrefix': genConfig.reusePromptPrefixDefault,
      'genGpuLayers': genConfig.nGpuLayersDefault,
      'streamBatchTokenThreshold': genConfig.streamBatchTokenThresholdDefault,
      'streamBatchByteThreshold': genConfig.streamBatchByteThresholdDefault,
      'gpuBackend': genConfig.gpuBackendDefault.name,
      'sendPort': initPort.sendPort,
    }, debugName: 'llmcpp_RagWorker');

    final initMsg = await initPort.first as Map<String, dynamic>;
    initPort.close();

    if (initMsg['type'] == 'error') {
      _isolate?.kill();
      _isolate = null;
      throw decodeError(
        initMsg,
        context: 'LlamaRagCoordinator init failed (${initMsg['phase']})',
      );
    }

    _workerPort = initMsg['port'] as SendPort;

    final tempProvider = _CoordEmbeddingProvider(_workerPort!, dimensions: 0);
    final probeVec = await tempProvider.embed('dim_probe');

    _embeddingProvider = _CoordEmbeddingProvider(
      _workerPort!,
      dimensions: probeVec.length,
    );
    _generationPlugin = _CoordPlugin(_workerPort!);
  }

  EmbeddingProvider get embeddingProvider => _embeddingProvider;
  LlmInterface get generationPlugin => _generationPlugin;
  bool get isReady => _workerPort != null;

  Future<void> dispose() async {
    final workerPort = _workerPort;
    _workerPort = null;

    if (workerPort != null) {
      // Wait for both engines to confirm teardown rather than guessing at a
      // delay — killing the isolate mid-teardown races native handle release.
      final ackPort = ReceivePort();
      workerPort.send({'type': 'dispose', 'replyPort': ackPort.sendPort});
      try {
        await ackPort.first.timeout(_disposeTimeout);
      } catch (_) {
        // A wedged or already-gone worker must not block teardown.
      } finally {
        ackPort.close();
      }
    }

    _isolate?.kill(priority: Isolate.beforeNextEvent);
    _isolate = null;
  }
}
