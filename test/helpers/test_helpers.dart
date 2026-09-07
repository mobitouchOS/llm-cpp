import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:mt_llmkit/mt_llmkit.dart';
import 'package:mt_llmkit/src/models/llm_model_base.dart';
import 'package:mt_llmkit/src/models/llm_model_isolated.dart';
import 'package:mt_llmkit/src/models/llm_model_standard.dart';

/// Helper functions for tests
class TestHelpers {
  /// Creates a temporary standard model for testing
  static LlmModelStandard createTestStandardModel({LlmConfig? config}) {
    return LlmModelStandard(config ?? const LlmConfig());
  }

  /// Creates a temporary isolated model for testing
  static LlmModelIsolated createTestIsolatedModel({LlmConfig? config}) {
    return LlmModelIsolated(config ?? const LlmConfig());
  }

  /// Creates minimal configuration for fast tests
  static const LlmConfig minimalConfig = LlmConfig(
    nGpuLayers: 0,
    nCtx: 128,
    nBatch: 32,
    nThreads: 1,
  );

  /// Creates maximal configuration
  static const LlmConfig maximalConfig = LlmConfig(
    nGpuLayers: 99,
    nCtx: 32768,
    nBatch: 8192,
    nPredict: 32768,
    nThreads: 16,
  );

  /// Checks if model has valid state
  static void expectValidState(
    LlmModelBase model, {
    required bool shouldBeInitialized,
    required bool shouldBeDisposed,
  }) {
    expect(
      model.isInitialized,
      shouldBeInitialized,
      reason: 'Model should ${shouldBeInitialized ? '' : 'not '}be initialized',
    );
    expect(
      model.isDisposed,
      shouldBeDisposed,
      reason: 'Model should ${shouldBeDisposed ? '' : 'not '}be disposed',
    );
  }

  /// Cleanup function for tests
  static Future<void> cleanupModel(LlmModelBase model) async {
    if (model.isInitialized && !model.isDisposed) {
      await model.dispose();
    }
  }
}

/// Matcher for checking StateError
final throwsStateError = throwsA(isA<StateError>());

/// Matcher for checking FileSystemException
final throwsFileSystemException = throwsA(isA<FileSystemException>());

/// Extension methods for tests
extension LlmModelTestExtensions on LlmModelBase {
  /// Checks if model is in valid initial state
  bool get isInInitialState => !isInitialized && !isDisposed;

  /// Checks if model is ready to use
  bool get isReady => isInitialized && !isDisposed;

  /// Checks if model is destroyed
  bool get isDestroyed => isDisposed;
}

/// Mock for streaming tests
class MockStreamController {
  final List<String> emittedValues = [];
  bool isClosed = false;

  void add(String value) {
    if (isClosed) throw StateError('Stream is closed');
    emittedValues.add(value);
  }

  void close() {
    isClosed = true;
  }

  Stream<String> get stream async* {
    for (final value in emittedValues) {
      yield value;
    }
  }
}

/// Builder pattern for configuration in tests
class TestConfigBuilder {
  int? _nGpuLayers;
  int? _nCtx;
  int? _nBatch;
  int? _nPredict;
  int? _nThreads;
  double? _temp;
  int? _topK;
  double? _topP;
  double? _penaltyRepeat;

  TestConfigBuilder withGpuLayers(int layers) {
    _nGpuLayers = layers;
    return this;
  }

  TestConfigBuilder withContext(int ctx) {
    _nCtx = ctx;
    return this;
  }

  TestConfigBuilder withBatch(int batch) {
    _nBatch = batch;
    return this;
  }

  TestConfigBuilder withPredict(int predict) {
    _nPredict = predict;
    return this;
  }

  TestConfigBuilder withThreads(int threads) {
    _nThreads = threads;
    return this;
  }

  TestConfigBuilder withTemperature(double temp) {
    _temp = temp;
    return this;
  }

  TestConfigBuilder withTopK(int topK) {
    _topK = topK;
    return this;
  }

  TestConfigBuilder withTopP(double topP) {
    _topP = topP;
    return this;
  }

  TestConfigBuilder withPenaltyRepeat(double penalty) {
    _penaltyRepeat = penalty;
    return this;
  }

  LlmConfig build() {
    return LlmConfig(
      nGpuLayers: _nGpuLayers,
      nCtx: _nCtx,
      nBatch: _nBatch,
      nPredict: _nPredict,
      nThreads: _nThreads,
      temp: _temp,
      topK: _topK,
      topP: _topP,
      penaltyRepeat: _penaltyRepeat,
    );
  }
}

void main() {
  group('TestHelpers', () {
    test('should create standard model', () {
      final model = TestHelpers.createTestStandardModel();
      expect(model, isNotNull);
      expect(model, isA<LlmModelStandard>());
      TestHelpers.cleanupModel(model);
    });

    test('should create isolated model', () {
      final model = TestHelpers.createTestIsolatedModel();
      expect(model, isNotNull);
      expect(model, isA<LlmModelIsolated>());
      TestHelpers.cleanupModel(model);
    });

    test('should validate model state', () async {
      final model = TestHelpers.createTestStandardModel();

      TestHelpers.expectValidState(
        model,
        shouldBeInitialized: false,
        shouldBeDisposed: false,
      );

      await model.dispose();

      TestHelpers.expectValidState(
        model,
        shouldBeInitialized: false,
        shouldBeDisposed: true,
      );
    });
  });

  group('TestConfigBuilder', () {
    test('should build config with fluent API', () {
      final config = TestConfigBuilder()
          .withGpuLayers(32)
          .withContext(4096)
          .withBatch(2048)
          .withThreads(4)
          .withTemperature(0.8)
          .build();

      expect(config.nGpuLayersDefault, 32);
      expect(config.nCtxDefault, 4096);
      expect(config.nBatchDefault, 2048);
      expect(config.nThreadsDefault, 4);
      expect(config.tempDefault, 0.8);
    });

    test('should build minimal config', () {
      final config = TestConfigBuilder()
          .withGpuLayers(0)
          .withContext(128)
          .build();

      expect(config.nGpuLayersDefault, 0);
      expect(config.nCtxDefault, 128);
    });

    test('should build empty config with defaults', () {
      final config = TestConfigBuilder().build();

      expect(config.nGpuLayersDefault, 64);
      expect(config.nCtxDefault, 8192);
    });
  });

  group('Model Extensions', () {
    test('should check initial state', () {
      final model = TestHelpers.createTestStandardModel();

      expect(model.isInInitialState, true);
      expect(model.isReady, false);
      expect(model.isDestroyed, false);

      TestHelpers.cleanupModel(model);
    });

    test('should check destroyed state', () async {
      final model = TestHelpers.createTestStandardModel();
      await model.dispose();

      expect(model.isInInitialState, false);
      expect(model.isReady, false);
      expect(model.isDestroyed, true);
    });
  });
}
