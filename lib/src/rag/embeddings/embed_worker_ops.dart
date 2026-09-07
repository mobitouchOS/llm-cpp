// lib/src/rag/embeddings/embed_worker_ops.dart

import 'package:llamadart/llamadart.dart';

/// Trims [text] so it fits [maxTokens] of the encoder's context.
///
/// Chunks used to be cut at a fixed character count, which is not a token
/// count: 2000 characters is roughly 500–700 tokens, so long chunks quietly
/// overran a 512-token encoder. This asks the model's own tokenizer.
Future<String> fitToContext(
  LlamaEngine engine,
  String text,
  int maxTokens,
) async {
  if (text.isEmpty || maxTokens <= 0) return text;

  final tokens = await engine.tokenize(text);
  if (tokens.length <= maxTokens) return text;

  return engine.detokenize(tokens.sublist(0, maxTokens));
}

/// Reads the embedding width from GGUF metadata.
///
/// The key is architecture-prefixed (`nomic-bert.embedding_length`,
/// `bert.embedding_length`, …), so match on the suffix. Returns null when the
/// model does not declare it — callers then fall back to embedding a probe
/// string, which costs an inference pass.
Future<int?> embeddingDimensionsFromMetadata(LlamaEngine engine) async {
  try {
    final metadata = await engine.getMetadata();
    for (final entry in metadata.entries) {
      if (entry.key.endsWith('.embedding_length')) {
        return int.tryParse(entry.value.trim());
      }
    }
  } catch (_) {
    // Metadata is a convenience here; the probe path still works.
  }
  return null;
}
