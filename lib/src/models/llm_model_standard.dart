// lib/src/models/llm_model_standard.dart
import 'dart:async';
import 'dart:io';

import 'package:llamadart/llamadart.dart';

import '../core/backend_perf.dart';
import '../core/generation_overrides.dart';
import '../core/llm_config.dart';
import '../core/model_params_builder.dart';
import '../core/performance_metrics.dart';
import '../core/streaming_result.dart';
import 'llm_model_base.dart';

class LlmModelStandard extends LlmModelBase {
  final LlmConfig config;
  LlamaEngine? _engine;

  LlmModelStandard(this.config);

  Stream<LlamaCompletionChunk> _create(
    String prompt,
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  ) {
    final base = buildGenerationParams(config);
    return _engine!.create(
      buildMessages(
        prompt,
        systemPrompt: systemPrompt,
        attachments: attachments,
      ),
      params: overrides?.applyTo(base) ?? base,
      enableThinking: overrides?.enableThinking ?? config.enableThinkingDefault,
    );
  }

  @override
  Future<void> loadModel(String localPath) async {
    checkNotDisposed();
    config.validate();

    if (!File(localPath).existsSync()) {
      throw FileSystemException('File not found', localPath);
    }

    _engine = LlamaEngine(LlamaBackend());
    await _engine!.loadModel(localPath, modelParams: buildModelParams(config));

    if (config.mmprojPath != null) {
      await _engine!.loadMultimodalProjector(config.mmprojPath!);
    }

    markAsInitialized();
  }

  @override
  Stream<String> sendPrompt(
    String prompt, {
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) {
    checkInitialized();
    return _bufferedStream(prompt, systemPrompt, attachments, overrides);
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
      await for (final chunk in _create(
        prompt,
        systemPrompt,
        attachments,
        overrides,
      )) {
        final text = chunk.choices.firstOrNull?.delta.content;
        if (text != null) buffer.write(text);
      }
      return buffer.toString();
    } finally {
      markGenerationEnd();
    }
  }

  Stream<String> _bufferedStream(
    String prompt,
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  ) async* {
    if (_engine == null) return;

    final buffer = StringBuffer();
    final stopwatch = Stopwatch()..start();
    const yieldInterval = Duration(milliseconds: 50);

    markGenerationStart();
    try {
      await for (final chunk in _create(
        prompt,
        systemPrompt,
        attachments,
        overrides,
      )) {
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
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) async* {
    checkInitialized();

    final startTime = DateTime.now();
    int totalTokenCount = 0;

    markGenerationStart();
    try {
      String? finishReason;

      await for (final chunk in _create(
        prompt,
        systemPrompt,
        attachments,
        overrides,
      )) {
        final choice = chunk.choices.firstOrNull;
        finishReason = choice?.finishReason ?? finishReason;

        final text = choice?.delta.content;
        final thinking = choice?.delta.thinking;
        if (text == null && thinking == null) continue;

        totalTokenCount += 1;

        yield StreamingChunk(
          text: text ?? '',
          thinking: thinking,
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
