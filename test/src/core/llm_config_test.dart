import 'package:flutter_test/flutter_test.dart';
import 'package:llamadart/llamadart.dart' show ModelParams;
import 'package:mt_llmkit/mt_llmkit.dart';

void main() {
  group('LlmConfig', () {
    test('should create config with default values', () {
      const config = LlmConfig();

      // Unset sizing knobs resolve to llamadart's auto behaviour rather than
      // to fixed numbers: full GPU offload, and 0 = "let llama.cpp decide".
      expect(config.nGpuLayersDefault, ModelParams.maxGpuLayers);
      expect(config.nCtxDefault, 8192);
      expect(config.nBatchDefault, 0);
      expect(config.nPredictDefault, 8192);
      expect(config.nThreadsDefault, 0);
      expect(config.tempDefault, 0.72);
      expect(config.topKDefault, 64);
      expect(config.topPDefault, 0.95);
      expect(config.penaltyRepeatDefault, 1.1);
    });

    test('should create config with custom values', () {
      const config = LlmConfig(
        nGpuLayers: 32,
        nCtx: 4096,
        nBatch: 2048,
        nPredict: 4096,
        nThreads: 4,
        temp: 0.8,
        topK: 40,
        topP: 0.9,
        penaltyRepeat: 1.2,
      );

      expect(config.nGpuLayersDefault, 32);
      expect(config.nCtxDefault, 4096);
      expect(config.nBatchDefault, 2048);
      expect(config.nPredictDefault, 4096);
      expect(config.nThreadsDefault, 4);
      expect(config.tempDefault, 0.8);
      expect(config.topKDefault, 40);
      expect(config.topPDefault, 0.9);
      expect(config.penaltyRepeatDefault, 1.2);
    });

    test('should handle null values gracefully', () {
      const config = LlmConfig(nGpuLayers: null, nCtx: null);

      expect(config.nGpuLayersDefault, ModelParams.maxGpuLayers);
      expect(config.nCtxDefault, 8192);
    });

    test('should preserve original null values', () {
      const config = LlmConfig();

      expect(config.nGpuLayers, null);
      expect(config.nCtx, null);
      expect(config.nBatch, null);
    });
  });

  group('LlmConfig — 0.8.x knobs', () {
    test('exposes the KV cache and thinking knobs llamadart 0.8.x added', () {
      const config = LlmConfig(
        flashAttention: FlashAttention.enabled,
        cacheTypeK: KvCacheType.q8_0,
        cacheTypeV: KvCacheType.q8_0,
        presencePenalty: 0.5,
        enableThinking: false,
        thinkingBudget: ThinkingBudget(maxTokens: 256),
      );

      expect(config.cacheTypeKDefault, KvCacheType.q8_0);
      expect(config.presencePenaltyDefault, 0.5);
      expect(config.enableThinkingDefault, isFalse);
      expect(config.thinkingBudget?.maxTokens, 256);
      expect(config.validate, returnsNormally);
    });

    test('thinking is on by default, matching llamadart', () {
      expect(const LlmConfig().enableThinkingDefault, isTrue);
    });

    test('rejects a quantized KV cache without flash attention', () {
      const config = LlmConfig(
        flashAttention: FlashAttention.disabled,
        cacheTypeK: KvCacheType.q4_0,
      );

      expect(config.validate, throwsArgumentError);
    });

    test('allows a quantized KV cache with flash attention on auto', () {
      const config = LlmConfig(cacheTypeK: KvCacheType.q4_0);

      expect(config.validate, returnsNormally);
    });
  });
}
