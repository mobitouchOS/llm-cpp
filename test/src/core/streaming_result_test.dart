import 'package:flutter_test/flutter_test.dart';
import 'package:mt_llmkit/mt_llmkit.dart';

void main() {
  group('StreamingChunk.finishReason', () {
    test('reports truncation when the token budget ran out', () {
      final chunk = StreamingChunk(
        text: '',
        isFinal: true,
        finishReason: 'length',
      );

      expect(chunk.isTruncated, isTrue);
    });

    test('does not report truncation on a clean stop', () {
      final chunk = StreamingChunk(
        text: '',
        isFinal: true,
        finishReason: 'stop',
      );

      expect(chunk.isTruncated, isFalse);
    });

    test('does not guess when the backend reports no reason', () {
      final chunk = StreamingChunk(text: '', isFinal: true);

      expect(chunk.finishReason, isNull);
      expect(chunk.isTruncated, isFalse);
    });
  });
}
