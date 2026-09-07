// lib/src/core/llm_errors.dart

import 'package:llamadart/llamadart.dart';

/// Serialises [error] into a plain map that survives an isolate hop.
///
/// llamadart exceptions carry a `details` field of type `dynamic`, which may
/// hold objects that cannot be sent through a [SendPort]. Only the type name
/// and the human-readable parts are transferred; [decodeError] rebuilds an
/// exception of the same class on the receiving side, so callers can still
/// tell a failed backend init apart from a corrupt model file.
Map<String, dynamic> encodeError(Object error) => {
  'errorType': error.runtimeType.toString(),
  'message': error is LlamaException ? error.message : '$error',
  if (error is LlamaException && error.details != null)
    'details': '${error.details}',
};

/// Rebuilds the exception encoded by [encodeError].
///
/// Unknown types — anything that is not part of llamadart's [LlamaException]
/// hierarchy — degrade to a plain [Exception] carrying the original text.
/// [context] is prepended to the message to say which operation failed.
Object decodeError(Map<String, dynamic> payload, {String? context}) {
  final message = payload['message'] as String? ?? 'Unknown error';
  final details = payload['details'] as String?;
  final prefixed = context == null ? message : '$context: $message';

  return switch (payload['errorType'] as String?) {
    'LlamaModelException' => LlamaModelException(prefixed, details),
    'LlamaContextException' => LlamaContextException(prefixed, details),
    'LlamaInferenceException' => LlamaInferenceException(prefixed, details),
    'LlamaBackendInitializationException' =>
      LlamaBackendInitializationException(prefixed, details),
    'LlamaSpeechException' => LlamaSpeechException(prefixed, details),
    'LlamaTextToSpeechException' => LlamaTextToSpeechException(
      prefixed,
      details,
    ),
    'LlamaAudioFormatException' => LlamaAudioFormatException(prefixed, details),
    'LlamaUnsupportedException' => LlamaUnsupportedException(prefixed),
    'LlamaStateException' => LlamaStateException(prefixed, details),
    _ => Exception(details == null ? prefixed : '$prefixed ($details)'),
  };
}
