import 'package:flutter_test/flutter_test.dart';
import 'package:mt_llmkit/mt_llmkit.dart';
import 'package:mt_llmkit/src/core/llm_errors.dart';

void main() {
  group('encodeError / decodeError', () {
    test('round-trips a llamadart exception keeping its type', () {
      final encoded = encodeError(
        LlamaBackendInitializationException('Vulkan unavailable', 'code 3'),
      );
      final decoded = decodeError(encoded);

      expect(decoded, isA<LlamaBackendInitializationException>());
      final exception = decoded as LlamaBackendInitializationException;
      expect(exception.message, 'Vulkan unavailable');
      expect(exception.details, 'code 3');
    });

    test('keeps model failures distinguishable from backend failures', () {
      final decoded = decodeError(
        encodeError(LlamaModelException('bad magic')),
      );

      expect(decoded, isA<LlamaModelException>());
      expect(decoded, isNot(isA<LlamaBackendInitializationException>()));
    });

    test('prefixes the message with the operation context', () {
      final decoded =
          decodeError(
                encodeError(LlamaModelException('bad magic')),
                context: 'LlmModelIsolated init failed',
              )
              as LlamaException;

      expect(decoded.message, 'LlmModelIsolated init failed: bad magic');
    });

    test(
      'carries only sendable values, so the map survives an isolate hop',
      () {
        final encoded = encodeError(
          LlamaModelException('boom', const Duration(seconds: 1)),
        );

        expect(encoded.values.every((v) => v is String), isTrue);
        expect(encoded['details'], '0:00:01.000000');
      },
    );

    test('degrades an unknown error type to a plain Exception', () {
      final decoded = decodeError(encodeError(StateError('nope')));

      expect(decoded, isA<Exception>());
      expect(decoded, isNot(isA<LlamaException>()));
      expect('$decoded', contains('nope'));
    });

    test('omits details when the exception has none', () {
      expect(
        encodeError(LlamaStateException('no context')),
        isNot(contains('details')),
      );
    });

    test('LlamaUnsupportedException survives despite taking no details', () {
      final decoded = decodeError(
        encodeError(LlamaUnsupportedException('aLoRA adapters')),
      );

      expect(decoded, isA<LlamaUnsupportedException>());
    });
  });
}
