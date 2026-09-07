import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:mt_llmkit/mt_llmkit.dart';
import 'package:mt_llmkit/src/models/llm_model_base.dart';
import 'package:mt_llmkit/src/models/llm_model_isolated.dart';
import 'package:mt_llmkit/src/models/llm_model_standard.dart';

void main() {
  group('Integration Tests - Complete Workflow', () {
    test('should create, configure and dispose standard model', () async {
      const config = LlmConfig(nGpuLayers: 32, nCtx: 4096, temp: 0.7);

      final model = LlmModelStandard(config);

      expect(model.isInitialized, false);
      expect(model.isDisposed, false);

      await model.dispose();

      expect(model.isDisposed, true);
    });

    test('should create, configure and dispose isolated model', () async {
      const config = LlmConfig(nGpuLayers: 32, nCtx: 4096, temp: 0.7);

      final model = LlmModelIsolated(config);

      expect(model.isInitialized, false);
      expect(model.isDisposed, false);

      await model.dispose();

      expect(model.isDisposed, true);
    });

    test('should handle multiple models simultaneously', () async {
      final model1 = LlmModelStandard(const LlmConfig());
      final model2 = LlmModelIsolated(const LlmConfig());

      expect(model1.isInitialized, false);
      expect(model2.isInitialized, false);

      await model1.dispose();
      await model2.dispose();

      expect(model1.isDisposed, true);
      expect(model2.isDisposed, true);
    });

    test('should work with different configs', () async {
      final models = [
        LlmModelStandard(const LlmConfig()),
        LlmModelStandard(const LlmConfig(nGpuLayers: 0, temp: 0.5)),
        LlmModelStandard(const LlmConfig(nCtx: 4096, topP: 0.8)),
        LlmModelIsolated(const LlmConfig()),
        LlmModelIsolated(const LlmConfig(nGpuLayers: 0, temp: 0.5)),
        LlmModelIsolated(const LlmConfig(nCtx: 4096, topP: 0.8)),
      ];

      for (final model in models) {
        expect(model, isNotNull);
        expect(model.isInitialized, false);
        await model.dispose();
        expect(model.isDisposed, true);
      }
    });

    test('should respect configuration parameters', () async {
      const lowResource = LlmConfig(
        nGpuLayers: 0,
        nCtx: 512,
        nBatch: 128,
        nThreads: 2,
      );

      const highResource = LlmConfig(
        nGpuLayers: 64,
        nCtx: 8192,
        nBatch: 4096,
        nThreads: 8,
      );

      final lowModel = LlmModelStandard(lowResource);
      final highModel = LlmModelStandard(highResource);

      expect(lowModel, isNotNull);
      expect(highModel, isNotNull);

      await lowModel.dispose();
      await highModel.dispose();
    });
  });

  group('Integration Tests - Error Handling', () {
    test('should handle file not found gracefully', () {
      final model = LlmModelStandard(const LlmConfig());

      expect(
        () => model.loadModel('/definitely/not/existing/model.gguf'),
        throwsA(isA<FileSystemException>()),
      );
    });

    test('should prevent operations on disposed model', () async {
      final model = LlmModelStandard(const LlmConfig());
      await model.dispose();

      expect(() => model.loadModel('/test.gguf'), throwsStateError);
      expect(() => model.sendPrompt('test'), throwsStateError);
      expect(() => model.clean(), throwsStateError);
    });

    test('should prevent operations before initialization', () {
      final model = LlmModelStandard(const LlmConfig());

      expect(() => model.sendPrompt('test'), throwsStateError);
      expect(() => model.clean(), throwsStateError);
    });
  });

  group('Integration Tests - LlmInterface Compliance', () {
    test('all models should implement LlmInterface', () {
      final standard = LlmModelStandard(const LlmConfig());
      final isolated = LlmModelIsolated(const LlmConfig());

      expect(standard, isA<LlmInterface>());
      expect(isolated, isA<LlmInterface>());
    });

    test('all models should extend LlmModelBase', () {
      final standard = LlmModelStandard(const LlmConfig());
      final isolated = LlmModelIsolated(const LlmConfig());

      expect(standard, isA<LlmModelBase>());
      expect(isolated, isA<LlmModelBase>());
    });

    test('all models should have consistent state management', () async {
      final models = <LlmModelBase>[
        LlmModelStandard(const LlmConfig()),
        LlmModelIsolated(const LlmConfig()),
      ];

      for (final model in models) {
        // Initial state
        expect(model.isInitialized, false);
        expect(model.isDisposed, false);

        // After dispose
        await model.dispose();
        expect(model.isInitialized, false);
        expect(model.isDisposed, true);
      }
    });
  });

  group('Integration Tests - Configuration Variations', () {
    test('should work with minimal configuration', () async {
      const config = LlmConfig(nCtx: 128, nBatch: 32, nThreads: 1);

      final model = LlmModelStandard(config);
      expect(model, isNotNull);
      await model.dispose();
    });

    test('should work with maximum configuration', () async {
      const config = LlmConfig(
        nGpuLayers: 99,
        nCtx: 32768,
        nBatch: 8192,
        nPredict: 32768,
        nThreads: 16,
        temp: 2.0,
        topK: 100,
        topP: 1.0,
        penaltyRepeat: 2.0,
      );

      final model = LlmModelStandard(config);
      expect(model, isNotNull);
      await model.dispose();
    });

    test('should handle edge case values', () async {
      const config = LlmConfig(
        nGpuLayers: 0,
        nCtx: 1,
        nBatch: 1,
        nPredict: 1,
        nThreads: 1,
        temp: 0.0,
        topK: 1,
        topP: 0.0,
        penaltyRepeat: 0.0,
      );

      final model = LlmModelStandard(config);
      expect(model, isNotNull);
      await model.dispose();
    });
  });
}
