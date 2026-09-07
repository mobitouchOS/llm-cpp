// lib/src/rag/embeddings/llama_embedding_provider.dart

import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:llamadart/llamadart.dart';

import '../../core/llm_errors.dart';
import 'embed_worker_ops.dart';
import 'embedding_provider.dart';

/// How long [LlamaEmbeddingProvider.dispose] waits for the worker to confirm
/// that llama.cpp released its native handles before the isolate is killed.
const Duration _disposeTimeout = Duration(seconds: 5);

// ── Worker isolate entry point ─────────────────────────────────────────────

Future<void> _llamaEmbedWorkerMain(Map<String, dynamic> args) async {
  final String modelPath = args['modelPath'] as String;
  final int contextSize = args['contextSize'] as int;
  final int batchSize = args['batchSize'] as int;
  final int numberOfThreads = args['numberOfThreads'] as int;
  final SendPort mainPort = args['sendPort'] as SendPort;

  final engine = LlamaEngine(LlamaBackend());
  try {
    await engine.loadModel(
      modelPath,
      modelParams: ModelParams(
        contextSize: contextSize,
        gpuLayers: 0,
        batchSize: batchSize,
        numberOfThreads: numberOfThreads,
        preferredBackend: GpuBackend.cpu,
      ),
    );
  } catch (e) {
    mainPort.send({'type': 'error', ...encodeError(e)});
    return;
  }

  final receivePort = ReceivePort();
  mainPort.send({
    'type': 'ready',
    'port': receivePort.sendPort,
    'dimensions': await embeddingDimensionsFromMetadata(engine),
  });

  await for (final message in receivePort) {
    if (message is! Map<String, dynamic>) continue;

    switch (message['type'] as String?) {
      case 'embed':
        final replyPort = message['replyPort'] as SendPort;
        try {
          final text = await fitToContext(
            engine,
            message['text'] as String,
            contextSize,
          );
          final embedding = await engine.embed(text);
          replyPort.send({'type': 'ok', 'embedding': embedding});
        } catch (e) {
          replyPort.send({'type': 'error', ...encodeError(e)});
        }

      case 'embed_batch':
        final replyPort = message['replyPort'] as SendPort;
        try {
          final texts = [
            for (final text in (message['texts'] as List).cast<String>())
              await fitToContext(engine, text, contextSize),
          ];
          // One native batch call instead of one isolate round-trip per chunk.
          final embeddings = await engine.embedBatch(texts);
          replyPort.send({'type': 'ok', 'embeddings': embeddings});
        } catch (e) {
          replyPort.send({'type': 'error', ...encodeError(e)});
        }

      case 'dispose':
        await engine.dispose();
        (message['replyPort'] as SendPort?)?.send({'type': 'disposed'});
        receivePort.close();
        return;
    }
  }
}

// ── LlamaEmbeddingProvider ─────────────────────────────────────────────────

/// Embedding provider that runs a GGUF embedding model in a dedicated isolate.
///
/// Unlike [LlamaRagCoordinator], each instance creates its own isolated worker
/// so the embed and generation models never share an isolate boundary.
///
/// ## Usage
///
/// ```dart
/// final provider = LlamaEmbeddingProvider(
///   modelPath: '/path/to/nomic-embed-text.gguf',
/// );
/// await provider.load();
///
/// final vector = await provider.embed('What is the capital of Poland?');
/// print('Dimensions: ${vector.length}');
///
/// await provider.dispose();
/// ```
class LlamaEmbeddingProvider implements EmbeddingProvider {
  final String modelPath;

  /// Context window for the embedding model (default 512 tokens).
  final int embedNCtx;

  /// Batch size for the embedding model (default 512).
  final int batchSize;

  /// Number of CPU threads for embedding inference (default 4).
  final int numberOfThreads;

  Isolate? _isolate;
  SendPort? _workerPort;
  int _dimensions = 0;
  bool _isInitialized = false;

  LlamaEmbeddingProvider({
    required this.modelPath,
    this.embedNCtx = 512,
    this.batchSize = 512,
    this.numberOfThreads = 4,
  });

  @override
  bool get isInitialized => _isInitialized;

  @override
  int get dimensions => _dimensions;

  /// Initializes from a config map (satisfies [EmbeddingProvider] contract).
  ///
  /// Delegates to [load]. The `config` map is ignored; use constructor
  /// parameters instead.
  @override
  Future<void> initialize(Map<String, dynamic> config) => load();

  /// Loads the embedding model in a dedicated worker isolate.
  ///
  /// Idempotent — subsequent calls are no-ops.
  Future<void> load() async {
    if (_isInitialized) return;

    if (!File(modelPath).existsSync()) {
      throw FileSystemException('Embedding model not found', modelPath);
    }

    final initPort = ReceivePort();
    _isolate = await Isolate.spawn(_llamaEmbedWorkerMain, {
      'modelPath': modelPath,
      'contextSize': embedNCtx,
      'batchSize': batchSize,
      'numberOfThreads': numberOfThreads,
      'sendPort': initPort.sendPort,
    }, debugName: 'llmcpp_EmbedWorker');

    final initMsg = await initPort.first as Map<String, dynamic>;
    initPort.close();

    if (initMsg['type'] == 'error') {
      _isolate?.kill();
      _isolate = null;
      throw decodeError(initMsg, context: 'LlamaEmbeddingProvider init failed');
    }

    _workerPort = initMsg['port'] as SendPort;

    // The model usually declares its embedding width in GGUF metadata; only
    // fall back to an inference pass when it does not.
    final declared = initMsg['dimensions'] as int?;
    _dimensions = declared ?? (await embed('probe')).length;
    _isInitialized = true;
  }

  @override
  Future<List<double>> embed(String text) async {
    if (_workerPort == null) {
      throw StateError(
        'LlamaEmbeddingProvider is not initialized. Call load() first.',
      );
    }

    final replyPort = ReceivePort();
    _workerPort!.send({
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
    if (_workerPort == null) {
      throw StateError(
        'LlamaEmbeddingProvider is not initialized. Call load() first.',
      );
    }
    if (texts.isEmpty) return const [];

    final replyPort = ReceivePort();
    _workerPort!.send({
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
    final workerPort = _workerPort;
    _workerPort = null;

    if (workerPort != null) {
      // Wait for the worker's confirmation rather than guessing at a delay —
      // killing the isolate mid-teardown races native handle release.
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
    _isInitialized = false;
    _dimensions = 0;
  }
}
