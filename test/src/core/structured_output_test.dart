import 'package:flutter_test/flutter_test.dart';
import 'package:mt_llmkit/mt_llmkit.dart';

const _citySchema = {
  'type': 'object',
  'properties': {
    'city': {'type': 'string'},
    'population': {'type': 'integer'},
  },
  'required': ['city'],
};

Stream<StreamingChunk> _stream(
  List<String> texts, {
  String? finishReason = 'stop',
  String? thinking,
}) async* {
  for (final text in texts) {
    yield StreamingChunk(text: text);
  }
  if (thinking != null) {
    yield StreamingChunk(text: '', thinking: thinking);
  }
  yield StreamingChunk(text: '', isFinal: true, finishReason: finishReason);
}

void main() {
  group('LlmStructuredOutput', () {
    test('exposes a response format the worker can be sent', () {
      final output = LlmStructuredOutput.jsonSchema(
        schema: _citySchema,
        decoder: (json) => json['city'] as String,
      );

      expect(output.responseFormat['type'], 'json_schema');
      expect(output.schema, _citySchema);
    });

    test('jsonObject asks for a bare JSON object', () {
      final output = LlmStructuredOutput.jsonObject(decoder: (json) => json);

      expect(output.responseFormat, {'type': 'json_object'});
      expect(output.schema, isNull);
    });

    test('rejects a schema it cannot constrain — at construction', () {
      // Not "later, mid-generation": the failure is knowable up front.
      expect(
        () => LlmStructuredOutput.jsonSchema(
          schema: const {'type': 'object', 'unevaluatedProperties': false},
          decoder: (json) => json,
        ),
        throwsA(isA<LlamaUnsupportedException>()),
      );
    });

    test('rejects a non-object root for jsonSchema', () {
      expect(
        () => LlmStructuredOutput.jsonSchema(
          schema: const {'type': 'array'},
          decoder: (json) => json,
        ),
        throwsA(isA<LlamaUnsupportedException>()),
      );
    });

    test('decodes valid output', () {
      final output = LlmStructuredOutput.jsonSchema(
        schema: _citySchema,
        decoder: (json) => json['city'] as String,
      );

      expect(output.parse('{"city":"Kraków"}'), 'Kraków');
    });

    test('malformed JSON becomes an inference error', () {
      final output = LlmStructuredOutput.jsonObject(decoder: (json) => json);

      expect(
        () => output.parse('{"city":'),
        throwsA(isA<LlamaInferenceException>()),
      );
    });

    test('output that violates the schema becomes an inference error', () {
      final output = LlmStructuredOutput.jsonSchema(
        schema: _citySchema,
        decoder: (json) => json['city'] as String,
      );

      expect(
        () => output.parse('{"population": 800000}'), // 'city' is required
        throwsA(isA<LlamaInferenceException>()),
      );
    });

    test('a throwing decoder is wrapped, not leaked raw', () {
      final output = LlmStructuredOutput.jsonObject(
        decoder: (json) => throw const FormatException('nope'),
      );

      expect(() => output.parse('{}'), throwsA(isA<LlamaInferenceException>()));
    });
  });

  group('parseStructured', () {
    final output = LlmStructuredOutput.jsonSchema(
      schema: _citySchema,
      decoder: (json) => json['city'] as String,
    );

    test('reassembles JSON split across chunks', () async {
      final result = await _stream([
        '{"ci',
        'ty":"Kra',
        'ków"}',
      ]).parseStructured(output);

      expect(result.value, 'Kraków');
      expect(result.rawJson, '{"city":"Kraków"}');
      expect(result.finishReason, 'stop');
    });

    test('keeps reasoning out of the JSON buffer', () async {
      final result = await _stream([
        '{"city":"Kraków"}',
      ], thinking: 'The user asked about Poland.').parseStructured(output);

      expect(result.rawJson, '{"city":"Kraków"}');
      expect(result.thinking, 'The user asked about Poland.');
    });

    test('names truncation instead of reporting a JSON parse offset', () async {
      // Hitting the token budget mid-object is the most likely real failure,
      // and `jsonDecode` describes it uselessly.
      await expectLater(
        _stream([
          '{"city":"Kra',
        ], finishReason: 'length').parseStructured(output),
        throwsA(
          isA<LlamaInferenceException>().having(
            (e) => e.message,
            'message',
            contains('truncated at the token budget'),
          ),
        ),
      );
    });
  });
}
