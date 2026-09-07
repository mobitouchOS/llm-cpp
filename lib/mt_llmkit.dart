// lib/mt_llmkit.dart

// ── llamadart re-exports ──────────────────────────────────────────────────────
export 'package:llamadart/llamadart.dart'
    show
        LlamaImageContent,
        LlamaAudioContent,
        LlamaTextContent,
        LlamaContentPart,
        GpuBackend,
        LoraAdapterConfig,
        GenerationGrammarTrigger,
        // Memory and sampling knobs surfaced through LlmConfig.
        FlashAttention,
        KvCacheType,
        ThinkingBudget,
        SpeculativeDecodingConfig,
        // Tool declarations: ToolParam builds the schema, ToolChoice picks
        // the policy. Tool handlers stay on your side — see LlmTool.
        ToolParam,
        ToolChoice,
        // Exception hierarchy — model load, generation and teardown failures
        // keep their llamadart type across the worker isolate boundary, so
        // callers can tell "GPU backend failed, retry on CPU" apart from
        // "model file is corrupt".
        LlamaException,
        LlamaModelException,
        LlamaContextException,
        LlamaInferenceException,
        LlamaBackendInitializationException,
        LlamaUnsupportedException,
        LlamaStateException;

// ── AI Chat providers (conversation-based) ───────────────────────────────────
export 'src/api/ai_chat_provider.dart';
export 'src/api/ai_chat_provider_factory.dart';
export 'src/api/chat_exceptions.dart';
export 'src/api/chat_models.dart';
export 'src/api/claude_chat_provider.dart';
export 'src/api/gemini_chat_provider.dart';
export 'src/api/mistral_chat_provider.dart';
export 'src/api/openai_chat_provider.dart';

// ── Core ─────────────────────────────────────────────────────────────────────
export 'src/core/chat_message.dart';
export 'src/core/conversation.dart' show Conversation;
export 'src/core/conversation_types.dart';
export 'src/core/generation_overrides.dart';
export 'src/core/generation_result.dart';
export 'src/core/llm_config.dart';
export 'src/core/llm_interface.dart';
export 'src/core/model_diagnostics.dart';
export 'src/core/performance_metrics.dart';
export 'src/core/streaming_result.dart';
export 'src/core/structured_output.dart';
export 'src/core/tools.dart';

// ── Local Model (GGUF) ───────────────────────────────────────────────────────
export 'src/gguf/local_model.dart';

// ── RAG (Retrieval-Augmented Generation) ─────────────────────────────────────
export 'src/rag/chunking/text_chunker.dart';
export 'src/rag/document/document.dart';
export 'src/rag/document/document_chunk.dart';
export 'src/rag/embeddings/embedding_provider.dart';
export 'src/rag/embeddings/llama_embedding_provider.dart';
export 'src/rag/rag_engine.dart';
export 'src/rag/rag_pipeline.dart';
export 'src/rag/vector_store/in_memory_vector_store.dart';
export 'src/rag/vector_store/vector_store.dart';
