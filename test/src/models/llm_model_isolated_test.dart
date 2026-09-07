import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:mt_llmkit/mt_llmkit.dart';
import 'package:mt_llmkit/src/models/llm_model_base.dart';
import 'package:mt_llmkit/src/models/llm_model_isolated.dart';
import 'package:mt_llmkit/src/models/llm_model_standard.dart';

void main() {
  group('LlmModelIsolated - State Management', () {
    late LlmModelIsolated model;

    setUp(() {
      model = LlmModelIsolated(const LlmConfig());
    });

    tearDown(() async {
      if (model.isInitialized && !model.isDisposed) {
        await model.dispose();
      }
    });

    test('should initialize with correct default state', () {
      expect(model.isInitialized, false);
      expect(model.isDisposed, false);
    });

    test('should throw FileSystemException when model file does not exist', () {
      expect(
        () => model.loadModel('/nonexistent/model.gguf'),
        throwsA(isA<FileSystemException>()),
      );
    });

    test(
      'should throw StateError when sending prompt before initialization',
      () {
        expect(() => model.sendPrompt('test'), throwsStateError);
      },
    );

    test('should throw StateError when cleaning before initialization', () {
      expect(() => model.clean(), throwsStateError);
    });

    test('should mark as disposed after dispose', () async {
      await model.dispose();

      expect(model.isInitialized, false);
      expect(model.isDisposed, true);
    });

    test('should throw StateError when loading after dispose', () async {
      await model.dispose();

      expect(() => model.loadModel('/test/model.gguf'), throwsStateError);
    });
  });

  group('LlmModelIsolated - Configuration', () {
    test('should use default configuration', () {
      final model = LlmModelIsolated(const LlmConfig());

      expect(model, isNotNull);
      expect(model.isInitialized, false);
    });

    test('should accept custom configuration', () {
      const config = LlmConfig(
        nGpuLayers: 32,
        nCtx: 4096,
        nBatch: 2048,
        temp: 0.8,
      );

      final model = LlmModelIsolated(config);

      expect(model, isNotNull);
      expect(model.isInitialized, false);
    });

    test('should accept config with all numeric params', () async {
      const config = LlmConfig(nGpuLayers: 4, temp: 0.5);

      final model = LlmModelIsolated(config);

      expect(model, isNotNull);
      await model.dispose();
    });
  });

  group('LlmModelIsolated - Lifecycle', () {
    late LlmModelIsolated model;

    setUp(() {
      model = LlmModelIsolated(const LlmConfig());
    });

    tearDown(() async {
      if (model.isInitialized && !model.isDisposed) {
        await model.dispose();
      }
    });

    test('should handle dispose without initialization', () async {
      await expectLater(model.dispose(), completes);
      expect(model.isDisposed, true);
    });

    test('should not allow operations after dispose', () async {
      await model.dispose();

      expect(() => model.loadModel('/test.gguf'), throwsStateError);
      expect(() => model.sendPrompt('test'), throwsStateError);
      expect(() => model.clean(), throwsStateError);
    });
  });

  group('LlmModelIsolated vs LlmModelStandard', () {
    test('both should implement same interface', () {
      final isolated = LlmModelIsolated(const LlmConfig());
      final standard = LlmModelStandard(const LlmConfig());

      expect(isolated, isA<LlmInterface>());
      expect(standard, isA<LlmInterface>());
      expect(isolated, isA<LlmModelBase>());
      expect(standard, isA<LlmModelBase>());
    });

    test('both should have same lifecycle methods', () {
      final isolated = LlmModelIsolated(const LlmConfig());
      final standard = LlmModelStandard(const LlmConfig());

      // Verify they have same methods
      expect(isolated.loadModel, isA<Function>());
      expect(standard.loadModel, isA<Function>());
      expect(isolated.sendPrompt, isA<Function>());
      expect(standard.sendPrompt, isA<Function>());
      expect(isolated.dispose, isA<Function>());
      expect(standard.dispose, isA<Function>());
      expect(isolated.clean, isA<Function>());
      expect(standard.clean, isA<Function>());
    });
  });
}
