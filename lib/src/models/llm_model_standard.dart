// lib/src/models/llm_model_standard.dart
import 'dart:async';
import 'dart:io';

import 'package:llamadart/llamadart.dart';

import '../core/backend_perf.dart';
import '../core/engine_rpc.dart';
import '../core/generation_overrides.dart';
import '../core/llm_config.dart';
import '../core/model_diagnostics.dart';
import '../core/model_params_builder.dart';
import '../core/performance_metrics.dart';
import '../core/streaming_result.dart';
import '../core/tools.dart';
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
      tools: overrides?.tools?.map((t) => t.toToolDefinition()).toList(),
      toolChoice: overrides?.toolChoice,
      parallelToolCalls: overrides?.parallelToolCalls ?? false,
      responseFormat: overrides?.responseFormat,
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
      final toolCalls = ToolCallAccumulator();

      await for (final chunk in _create(
        prompt,
        systemPrompt,
        attachments,
        overrides,
      )) {
        final choice = chunk.choices.firstOrNull;
        finishReason = choice?.finishReason ?? finishReason;

        final deltas = choice?.delta.toolCalls;
        if (deltas != null) toolCalls.add(deltas);

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
        toolCalls: toolCalls.build(),
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

  Future<Object?> _call(String method, [Map<String, dynamic> args = const {}]) {
    checkInitialized();
    return dispatchEngineCall(_engine!, method, args);
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

  @override
  void clean() {
    checkInitialized();
    // create() is stateless in llamadart — no context to reset
  }
}
