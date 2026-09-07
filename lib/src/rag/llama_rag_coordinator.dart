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
import '../core/generation_overrides.dart';
import '../core/llm_config.dart';
import '../core/model_params_builder.dart';
import '../core/llm_errors.dart';
import '../core/llm_interface.dart';
import '../core/performance_metrics.dart';
import '../core/streaming_result.dart';
import 'embeddings/embed_worker_ops.dart';
import 'embeddings/embedding_provider.dart';

/// How long [LlamaRagCoordinator.dispose] waits for the worker to confirm that
/// both engines released their native handles before the isolate is killed.
const Duration _disposeTimeout = Duration(seconds: 5);

// ── Worker isolate ────────────────────────────────────────────────────────────

Future<void> _llamaRagWorkerMain(Map<String, dynamic> args) async {
  final String embedModelPath = args['embedModelPath'] as String;
  final String genModelPath = args['genModelPath'] as String;
  final int embedContextSize = args['embedContextSize'] as int;
  final LlmConfig genConfig = args['genConfig'] as LlmConfig;
  final SendPort mainPort = args['sendPort'] as SendPort;

  final embedEngine = LlamaEngine(LlamaBackend());
  try {
    await embedEngine.loadModel(
      embedModelPath,
      modelParams: ModelParams(
        contextSize: embedContextSize,
        gpuLayers: 0,
        batchSize: genConfig.nBatchDefault,
        numberOfThreads: genConfig.nThreadsDefault,
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
      modelParams: buildModelParams(genConfig),
    );
  } catch (e) {
    await embedEngine.dispose();
    mainPort.send({'type': 'error', 'phase': 'gen_init', ...encodeError(e)});
    return;
  }

  final baseParams = buildGenerationParams(genConfig);

  final receivePort = ReceivePort();
  mainPort.send({
    'type': 'ready',
    'port': receivePort.sendPort,
    'dimensions': await embeddingDimensionsFromMetadata(embedEngine),
  });

  StreamSubscription<LlamaCompletionChunk>? genSubscription;

  await for (final message in receivePort) {
    if (message is! Map<String, dynamic>) continue;

    switch (message['type'] as String?) {
      case 'embed':
        final replyPort = message['replyPort'] as SendPort;
        try {
          final text = await fitToContext(
            embedEngine,
            message['text'] as String,
            embedContextSize,
          );
          final embedding = await embedEngine.embed(text);
          replyPort.send({'type': 'ok', 'embedding': embedding});
        } catch (e) {
          replyPort.send({'type': 'error', ...encodeError(e)});
        }

      case 'embed_batch':
        final replyPort = message['replyPort'] as SendPort;
        try {
          final texts = [
            for (final text in (message['texts'] as List).cast<String>())
              await fitToContext(embedEngine, text, embedContextSize),
          ];
          final embeddings = await embedEngine.embedBatch(texts);
          replyPort.send({'type': 'ok', 'embeddings': embeddings});
        } catch (e) {
          replyPort.send({'type': 'error', ...encodeError(e)});
        }

      case 'generate':
        final prompt = message['prompt'] as String;
        final systemPrompt = message['systemPrompt'] as String?;
        final overrides = message['overrides'] as GenerationOverrides?;
        final streamPort = message['streamPort'] as SendPort;
        String? finishReason;

        genSubscription = genEngine
            .create(
              buildMessages(prompt, systemPrompt: systemPrompt),
              params: overrides?.applyTo(baseParams) ?? baseParams,
              enableThinking:
                  overrides?.enableThinking ?? genConfig.enableThinkingDefault,
            )
            .listen(
              (chunk) {
                final choice = chunk.choices.firstOrNull;
                finishReason = choice?.finishReason ?? finishReason;
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
      'text': text,
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
    if (texts.isEmpty) return const [];

    final replyPort = ReceivePort();
    _workerPort.send({
      'type': 'embed_batch',
      'texts': texts,
      'replyPort': replyPort.sendPort,
    });

    final response = await replyPort.first as Map<String, dynamic>;
    replyPort.close();
    if (response['type'] == 'error') {
      throw decodeError(response, context: 'Embedding error');
    }

    return [
      for (final vector in response['embeddings'] as List)
        [for (final value in vector as List) (value as num).toDouble()],
    ];
  }

  @override
  Future<void> dispose() async {
    _isInitialized = false;
  }

  @override
  int get dimensions => _dimensions;

  @override
  bool get isInitialized => _isInitialized;
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
  Stream<String> sendPrompt(
    String prompt, {
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) => _events(
    prompt,
    systemPrompt: systemPrompt,
    attachments: attachments,
    overrides: overrides,
  ).where((e) => !e.isFinal && e.text.isNotEmpty).map((e) => e.text);

  Stream<GenerationEvent> _events(
    String prompt, {
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) {
    if (attachments != null && attachments.isNotEmpty) {
      throw UnsupportedError(
        'Attachments are not supported in the RAG pipeline.',
      );
    }
    final controller = StreamController<GenerationEvent>();
    final replyPort = ReceivePort();
    var finished = false;

    _workerPort.send({
      'type': 'generate',
      'prompt': prompt,
      if (systemPrompt != null) 'systemPrompt': systemPrompt,
      if (overrides != null) 'overrides': overrides,
      'streamPort': replyPort.sendPort,
    });

    _isGenerating = true;
    final sub = replyPort.listen((dynamic message) {
      if (message is! Map<String, dynamic>) return;
      switch (message['type'] as String?) {
        case 'token':
          if (!controller.isClosed) {
            controller.add(
              GenerationEvent(
                text: message['text'] as String? ?? '',
                thinking: message['thinking'] as String?,
              ),
            );
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
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) async {
    final buffer = StringBuffer();
    await for (final token in sendPrompt(
      prompt,
      systemPrompt: systemPrompt,
      attachments: attachments,
      overrides: overrides,
    )) {
      buffer.write(token);
    }
    return buffer.toString();
  }

  @override
  Stream<StreamingChunk> sendPromptStream(
    String prompt, {
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) async* {
    final startTime = DateTime.now();
    int totalTokenCount = 0;

    await for (final event in _events(
      prompt,
      systemPrompt: systemPrompt,
      attachments: attachments,
      overrides: overrides,
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
        thinking: event.thinking,
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

    genConfig.validate();

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
      'genConfig': genConfig,
      'sendPort': initPort.sendPort,
    }, debugName: 'mt_llmkit_RagWorker');

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

    // Prefer the width the model declares in GGUF metadata; only spend an
    // inference pass on a probe when it does not declare one.
    var dimensions = initMsg['dimensions'] as int?;
    if (dimensions == null) {
      final probe = _CoordEmbeddingProvider(_workerPort!, dimensions: 0);
      dimensions = (await probe.embed('dim_probe')).length;
    }

    _embeddingProvider = _CoordEmbeddingProvider(
      _workerPort!,
      dimensions: dimensions,
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
