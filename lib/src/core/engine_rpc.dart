// lib/src/core/engine_rpc.dart

import 'package:llamadart/llamadart.dart';

import 'model_diagnostics.dart';

/// Non-streaming engine operations, addressed by name.
///
/// The isolate backend has to reach these through a port and the in-process
/// backend calls them directly; naming them once keeps the two from drifting.
/// Every return value is isolate-sendable.
Future<Object?> dispatchEngineCall(
  LlamaEngine engine,
  String method,
  Map<String, dynamic> args,
) async {
  switch (method) {
    case 'tokenize':
      return engine.tokenize(
        args['text'] as String,
        addSpecial: args['addSpecial'] as bool? ?? true,
      );

    case 'detokenize':
      return engine.detokenize(
        (args['tokens'] as List).cast<int>(),
        special: args['special'] as bool? ?? false,
      );

    case 'countTokens':
      return engine.getTokenCount(args['text'] as String);

    case 'contextSize':
      return engine.getContextSize();

    case 'metadata':
      return engine.getMetadata();

    case 'supportsStatePersistence':
      return engine.supportsStatePersistence;

    case 'saveState':
      return engine.stateSaveFile(
        args['path'] as String,
        tokens: (args['tokens'] as List).cast<int>(),
      );

    case 'loadState':
      final capacity =
          args['tokenCapacity'] as int? ?? await engine.getContextSize();
      final result = await engine.stateLoadFile(
        args['path'] as String,
        tokenCapacity: capacity,
      );
      return result.tokens;

    case 'setLora':
      await engine.setLora(
        args['path'] as String,
        scale: (args['scale'] as num?)?.toDouble() ?? 1.0,
      );
      return null;

    case 'removeLora':
      await engine.removeLora(args['path'] as String);
      return null;

    case 'clearLoras':
      await engine.clearLoras();
      return null;

    case 'diagnostics':
      return _diagnostics(engine);

    default:
      throw UnsupportedError('Unknown engine call: $method');
  }
}

Future<ModelDiagnostics> _diagnostics(LlamaEngine engine) async {
  // VRAM reporting is optional and throws on backends without it; diagnostics
  // must not fail just because one field is unavailable.
  ({int total, int free})? vram;
  try {
    vram = await engine.getVramInfo();
  } catch (_) {
    vram = null;
  }

  return ModelDiagnostics(
    backendName: await engine.getBackendName(),
    availableBackends: await engine.getAvailableBackends(),
    resolvedGpuLayers: await engine.getResolvedGpuLayers(),
    gpuSupported: await engine.isGpuSupported(),
    modelFileType: (await engine.getModelFileType())?.name,
    contextSize: await engine.getContextSize(),
    supportsVision: await engine.supportsVision,
    supportsAudio: await engine.supportsAudio,
    supportsStatePersistence: engine.supportsStatePersistence,
    vramTotal: vram?.total,
    vramFree: vram?.free,
  );
}
