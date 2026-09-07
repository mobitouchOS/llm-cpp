// lib/src/models/llm_model_standard.dart
import 'dart:async';
import 'dart:io';

import 'package:llamadart/llamadart.dart';

import '../core/backend_perf.dart';
import '../core/llm_config.dart';
import '../core/performance_metrics.dart';
import '../core/streaming_result.dart';
import 'llm_model_base.dart';

class LlmModelStandard extends LlmModelBase {
  final LlmConfig config;
  LlamaEngine? _engine;

  LlmModelStandard(this.config);

  ModelParams get _modelParams => ModelParams(
    contextSize: config.nCtxDefault,
    gpuLayers: config.nGpuLayersDefault,
    batchSize: config.nBatchDefault,
    numberOfThreads: config.nThreadsDefault,
    numberOfThreadsBatch: config.numberOfThreadsBatchDefault,
    microBatchSize: config.microBatchSizeDefault,
    maxParallelSequences: config.maxParallelSequencesDefault,
    loras: config.lorasDefault,
    chatTemplate: config.chatTemplate,
    preferredBackend: config.gpuBackendDefault,
  );

  GenerationParams get _genParams => GenerationParams(
    maxTokens: config.nPredictDefault,
    temp: config.tempDefault,
    topK: config.topKDefault,
    topP: config.topPDefault,
    minP: config.minPDefault,
    penalty: config.penaltyRepeatDefault,
    seed: config.seed,
    stopSequences: config.stopSequencesDefault,
    grammar: config.grammar,
    grammarLazy: config.grammarLazyDefault,
    grammarTriggers: config.grammarTriggersDefault,
    preservedTokens: config.preservedTokensDefault,
    grammarRoot: config.grammarRootDefault,
    reusePromptPrefix: config.reusePromptPrefixDefault,
    streamBatchTokenThreshold: config.streamBatchTokenThresholdDefault,
    streamBatchByteThreshold: config.streamBatchByteThresholdDefault,
  );

  LlamaChatMessage _buildMessage(
    String prompt,
    List<LlamaImageContent>? images,
  ) {
    if (images != null && images.isNotEmpty) {
      return LlamaChatMessage.withContent(
        role: LlamaChatRole.user,
        content: [LlamaTextContent(prompt), ...images],
      );
    }
    return LlamaChatMessage.fromText(role: LlamaChatRole.user, text: prompt);
  }

  @override
  Future<void> loadModel(String localPath) async {
    checkNotDisposed();

    if (!File(localPath).existsSync()) {
      throw FileSystemException('File not found', localPath);
    }

    _engine = LlamaEngine(LlamaBackend());
    await _engine!.loadModel(localPath, modelParams: _modelParams);

    if (config.mmprojPath != null) {
      await _engine!.loadMultimodalProjector(config.mmprojPath!);
    }

    markAsInitialized();
  }

  @override
  Stream<String> sendPrompt(String prompt, {List<LlamaImageContent>? images}) {
    checkInitialized();
    return _bufferedStream(prompt, images: images);
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
      await for (final chunk in _engine!.create([
        _buildMessage(prompt, images),
      ], params: _genParams)) {
        final text = chunk.choices.firstOrNull?.delta.content;
        if (text != null) buffer.write(text);
      }
      return buffer.toString();
    } finally {
      markGenerationEnd();
    }
  }

  Stream<String> _bufferedStream(
    String prompt, {
    List<LlamaImageContent>? images,
  }) async* {
    if (_engine == null) return;

    final buffer = StringBuffer();
    final stopwatch = Stopwatch()..start();
    const yieldInterval = Duration(milliseconds: 50);

    markGenerationStart();
    try {
      await for (final chunk in _engine!.create([
        _buildMessage(prompt, images),
      ], params: _genParams)) {
        final text = chunk.choices.firstOrNull?.delta.content;
        if (text == null) continue;

        buffer.write(text);

        if (stopwatch.elapsed >= yieldInterval) {
          if (buffer.isNotEmpty) {
            yield buffer.toString();
            buffer.clear();
          }
          stopwatch.reset();
          await Future.delayed(Duration.zero);
        }
      }

      if (buffer.isNotEmpty) {
        yield buffer.toString();
      }
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
      String? finishReason;

      await for (final chunk in _engine!.create([
        _buildMessage(prompt, images),
      ], params: _genParams)) {
        final choice = chunk.choices.firstOrNull;
        finishReason = choice?.finishReason ?? finishReason;

        final text = choice?.delta.content;
        if (text == null) continue;

        totalTokenCount += 1;

        yield StreamingChunk(
          text: text,
          metrics: PerformanceMetrics.fromGeneration(
            tokenCount: totalTokenCount,
            startTime: startTime,
            endTime: DateTime.now(),
          ),
          isFinal: false,
        );
      }

      yield StreamingChunk(
        text: '',
        metrics: finalMetrics(
          perf: await readBackendPerf(_engine!),
          fallbackTokenCount: totalTokenCount,
          startTime: startTime,
          endTime: DateTime.now(),
        ),
        isFinal: true,
        finishReason: finishReason,
      );
    } finally {
      markGenerationEnd();
    }
  }

  @override
  Future<void> dispose() async {
    final engine = _engine;
    _engine = null;
    await engine?.dispose();
    markAsDisposed();
  }

  @override
  void clean() {
    checkInitialized();
    // create() is stateless in llamadart — no context to reset
  }
}
