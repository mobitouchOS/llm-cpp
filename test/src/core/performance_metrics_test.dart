// Tests for PerformanceMetrics class
import 'package:flutter_test/flutter_test.dart';
import 'package:mt_llmkit/mt_llmkit.dart';

void main() {
  group('PerformanceMetrics', () {
    test('should create from generation with correct calculations', () {
      final startTime = DateTime(2024, 1, 1, 12, 0, 0);
      final endTime = DateTime(2024, 1, 1, 12, 0, 2); // 2 seconds later
      const tokenCount = 100;

      final metrics = PerformanceMetrics.fromGeneration(
        tokenCount: tokenCount,
        startTime: startTime,
        endTime: endTime,
      );

      expect(metrics.tokensGenerated, 100);
      expect(metrics.durationMs, 2000);
      expect(metrics.tokensPerSecond, 50.0); // 100 tokens / 2 seconds
      expect(metrics.msPerToken, 20.0); // 2000ms / 100 tokens
    });

    test('should handle zero duration', () {
      final now = DateTime.now();
      const tokenCount = 10;

      final metrics = PerformanceMetrics.fromGeneration(
        tokenCount: tokenCount,
        startTime: now,
        endTime: now,
      );

      expect(metrics.tokensGenerated, 10);
      expect(metrics.durationMs, 0);
      expect(metrics.tokensPerSecond, 0.0);
    });

    test('should handle zero tokens', () {
      final startTime = DateTime.now();
      final endTime = startTime.add(const Duration(seconds: 1));

      final metrics = PerformanceMetrics.fromGeneration(
        tokenCount: 0,
        startTime: startTime,
        endTime: endTime,
      );

      expect(metrics.tokensGenerated, 0);
      expect(metrics.msPerToken, 0.0);
    });

    test('should calculate duration correctly', () {
      final startTime = DateTime(2024, 1, 1, 12, 0, 0);
      final endTime = DateTime(2024, 1, 1, 12, 0, 5, 500);

      final metrics = PerformanceMetrics.fromGeneration(
        tokenCount: 100,
        startTime: startTime,
        endTime: endTime,
      );

      expect(metrics.duration, const Duration(milliseconds: 5500));
      expect(metrics.durationMs, 5500);
    });

    test('should convert to string correctly', () {
      final metrics = PerformanceMetrics(
        tokensGenerated: 100,
        durationMs: 2000,
        tokensPerSecond: 50.0,
        msPerToken: 20.0,
        startTime: DateTime.now(),
        endTime: DateTime.now(),
      );

      final str = metrics.toString();
      expect(str, contains('tokens: 100'));
      expect(str, contains('duration: 2000ms'));
      expect(str, contains('t/s: 50.00'));
      expect(str, contains('ms/token: 20.00'));
    });

    test('should convert to and from JSON', () {
      final startTime = DateTime(2024, 1, 1, 12, 0, 0);
      final endTime = DateTime(2024, 1, 1, 12, 0, 2);

      final original = PerformanceMetrics.fromGeneration(
        tokenCount: 100,
        startTime: startTime,
        endTime: endTime,
      );

      final json = original.toJson();
      final restored = PerformanceMetrics.fromJson(json);

      expect(restored.tokensGenerated, original.tokensGenerated);
      expect(restored.durationMs, original.durationMs);
      expect(restored.tokensPerSecond, original.tokensPerSecond);
      expect(restored.msPerToken, original.msPerToken);
      expect(restored.startTime, original.startTime);
      expect(restored.endTime, original.endTime);
    });

    test('should calculate high throughput correctly', () {
      final startTime = DateTime(2024, 1, 1, 12, 0, 0);
      final endTime = DateTime(2024, 1, 1, 12, 0, 0, 500); // 500ms

      final metrics = PerformanceMetrics.fromGeneration(
        tokenCount: 100,
        startTime: startTime,
        endTime: endTime,
      );

      expect(metrics.tokensPerSecond, 200.0); // 100 tokens / 0.5 seconds
      expect(metrics.msPerToken, 5.0); // 500ms / 100 tokens
    });

    test('should calculate low throughput correctly', () {
      final startTime = DateTime(2024, 1, 1, 12, 0, 0);
      final endTime = DateTime(2024, 1, 1, 12, 0, 10); // 10 seconds

      final metrics = PerformanceMetrics.fromGeneration(
        tokenCount: 50,
        startTime: startTime,
        endTime: endTime,
      );

      expect(metrics.tokensPerSecond, 5.0); // 50 tokens / 10 seconds
      expect(metrics.msPerToken, 200.0); // 10000ms / 50 tokens
    });
  });

  group('PerformanceMetrics.fromBackendPerf', () {
    final startTime = DateTime(2024, 1, 1, 12, 0, 0);
    final endTime = DateTime(2024, 1, 1, 12, 0, 10);

    test('uses the backend decode time, not wall clock, for throughput', () {
      final metrics = PerformanceMetrics.fromBackendPerf(
        evalTokens: 100,
        startTime: startTime,
        endTime: endTime,
        promptTokens: 512,
        promptEvalMs: 8000,
        evalMs: 2000,
      );

      // 100 tokens in 2s of decoding, even though 10s of wall clock elapsed
      // (8s of it spent ingesting the prompt).
      expect(metrics.tokensPerSecond, 50.0);
      expect(metrics.msPerToken, 20.0);
      expect(metrics.durationMs, 10000);
      expect(metrics.promptTokens, 512);
      expect(metrics.isExact, isTrue);
    });

    test(
      'falls back to wall clock when the backend reports no decode time',
      () {
        final metrics = PerformanceMetrics.fromBackendPerf(
          evalTokens: 100,
          startTime: startTime,
          endTime: endTime,
        );

        expect(metrics.tokensPerSecond, 10.0);
        expect(metrics.isExact, isTrue);
      },
    );

    test('marks chunk-count estimates as inexact', () {
      final metrics = PerformanceMetrics.fromGeneration(
        tokenCount: 100,
        startTime: startTime,
        endTime: endTime,
      );

      expect(metrics.isExact, isFalse);
      expect(metrics.promptTokens, isNull);
    });

    test('round-trips the backend fields through JSON', () {
      final metrics = PerformanceMetrics.fromBackendPerf(
        evalTokens: 100,
        startTime: startTime,
        endTime: endTime,
        promptTokens: 512,
        promptEvalMs: 8000,
        evalMs: 2000,
      );

      final restored = PerformanceMetrics.fromJson(metrics.toJson());

      expect(restored.promptTokens, 512);
      expect(restored.promptEvalMs, 8000);
      expect(restored.evalMs, 2000);
      expect(restored.isExact, isTrue);
    });
  });
}
