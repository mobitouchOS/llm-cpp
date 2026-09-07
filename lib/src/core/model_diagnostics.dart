// lib/src/core/model_diagnostics.dart

/// What the backend actually resolved for a loaded model.
///
/// Configuration says what was asked for; this says what happened — which
/// backend won, how many layers really reached the GPU, whether the build
/// supports vision or audio at all. Without it the only way to find out that
/// GPU offload silently fell back to CPU is to notice the tokens/second.
class ModelDiagnostics {
  /// Backend that ended up running inference (e.g. `Metal`, `Vulkan`, `CPU`).
  final String backendName;

  /// Every backend this build could have used.
  final String availableBackends;

  /// Layers actually offloaded to the GPU. Null when the backend does not
  /// report it; `0` means everything ran on the CPU regardless of
  /// `LlmConfig.nGpuLayers`.
  final int? resolvedGpuLayers;

  /// Whether the current hardware and backend support GPU acceleration.
  final bool gpuSupported;

  /// Quantization of the loaded GGUF (e.g. `q4_K_M`), when reported.
  final String? modelFileType;

  /// Context window the model was actually loaded with, in tokens.
  final int contextSize;

  /// Whether image input is usable — requires a multimodal projector.
  final bool supportsVision;

  /// Whether audio input is usable.
  final bool supportsAudio;

  /// Whether the KV cache can be saved and restored — see
  /// `LocalModel.saveState`.
  final bool supportsStatePersistence;

  /// Total VRAM in bytes, when the backend reports it.
  final int? vramTotal;

  /// Free VRAM in bytes, when the backend reports it.
  final int? vramFree;

  const ModelDiagnostics({
    required this.backendName,
    required this.availableBackends,
    required this.resolvedGpuLayers,
    required this.gpuSupported,
    required this.modelFileType,
    required this.contextSize,
    required this.supportsVision,
    required this.supportsAudio,
    required this.supportsStatePersistence,
    required this.vramTotal,
    required this.vramFree,
  });

  @override
  String toString() =>
      'ModelDiagnostics(backend: $backendName, gpuLayers: $resolvedGpuLayers, '
      'ctx: $contextSize, fileType: $modelFileType, vision: $supportsVision, '
      'audio: $supportsAudio, statePersistence: $supportsStatePersistence)';
}
