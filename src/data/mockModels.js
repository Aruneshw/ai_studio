// Comprehensive Mock Model Data & Architecture Metadata
export const MODEL_CATALOG = [
  {
    id: 'llama-3.2-8b-instruct',
    name: 'Llama 3.2 8B Instruct',
    family: 'Llama 3.2',
    parameterCount: '8.03B',
    format: 'Safetensors (FP16)',
    rawFormat: 'safetensors',
    quantization: 'None (FP16)',
    isQuantized: false,
    contextWindow: 8192,
    diskSize: '16.1 GB',
    layers: 32,
    hiddenSize: 4096,
    intermediateSize: 14336,
    attentionHeads: 32,
    kvHeads: 8,
    headDim: 128,
    vocabSize: 128256,
    activation: 'silu (SwiGLU)',
    source: 'Local Directory',
    path: '/media/models/llama-3.2-8b-instruct',
    status: 'Ready',
    capabilities: ['Inference', 'LoRA / QLoRA', 'Full Fine-Tuning', 'Architecture Introspection'],
    avatar: '🦙',
    config: {
      architectures: ["LlamaForCausalLM"],
      attention_bias: false,
      attention_dropout: 0.0,
      bos_token_id: 128000,
      eos_token_id: 128001,
      hidden_act: "silu",
      hidden_size: 4096,
      initializer_range: 0.02,
      intermediate_size: 14336,
      max_position_embeddings: 8192,
      num_attention_heads: 32,
      num_hidden_layers: 32,
      num_key_value_heads: 8,
      rms_norm_eps: 1e-05,
      rope_scaling: null,
      rope_theta: 500000.0,
      tie_word_embeddings: false,
      torch_dtype: "float16",
      transformers_version: "4.44.0",
      use_cache: true,
      vocab_size: 128256
    }
  },
  {
    id: 'mistral-7b-v0.3',
    name: 'Mistral 7B Instruct v0.3',
    family: 'Mistral',
    parameterCount: '7.24B',
    format: 'GGUF (Q4_K_M)',
    rawFormat: 'gguf',
    quantization: '4-bit (Q4_K_M)',
    isQuantized: true,
    contextWindow: 32768,
    diskSize: '4.37 GB',
    layers: 32,
    hiddenSize: 4096,
    intermediateSize: 14336,
    attentionHeads: 32,
    kvHeads: 8,
    headDim: 128,
    vocabSize: 32768,
    activation: 'silu (SwiGLU)',
    source: 'Ollama',
    path: '~/.ollama/models/blobs/sha256-mistral7b',
    status: 'Ready',
    capabilities: ['Inference', 'Architecture Introspection', 'Adapter Merge (Via Base)'],
    avatar: '🌪️',
    config: {
      architectures: ["MistralForCausalLM"],
      hidden_size: 4096,
      intermediate_size: 14336,
      num_attention_heads: 32,
      num_hidden_layers: 32,
      num_key_value_heads: 8,
      sliding_window: 4096,
      vocab_size: 32768
    }
  },
  {
    id: 'deepseek-r1-distill-qwen-8b',
    name: 'DeepSeek-R1 Distill Qwen 8B',
    family: 'DeepSeek / Qwen',
    parameterCount: '8.19B',
    format: 'GGUF (Q8_0)',
    rawFormat: 'gguf',
    quantization: '8-bit (Q8_0)',
    isQuantized: true,
    contextWindow: 131072,
    diskSize: '8.54 GB',
    layers: 36,
    hiddenSize: 4096,
    intermediateSize: 18944,
    attentionHeads: 32,
    kvHeads: 8,
    headDim: 128,
    vocabSize: 152064,
    activation: 'silu',
    source: 'Local Cache',
    path: '/media/models/deepseek-r1-distill-qwen-8b.gguf',
    status: 'Ready',
    capabilities: ['Inference', 'Architecture Introspection'],
    avatar: '🐋',
    config: {
      architectures: ["Qwen2ForCausalLM"],
      hidden_size: 4096,
      num_hidden_layers: 36,
      vocab_size: 152064
    }
  },
  {
    id: 'phi-3-mini-4k',
    name: 'Phi-3 Mini 4K Instruct',
    family: 'Phi-3',
    parameterCount: '3.82B',
    format: 'Safetensors (FP16)',
    rawFormat: 'safetensors',
    quantization: 'None (FP16)',
    isQuantized: false,
    contextWindow: 4096,
    diskSize: '7.64 GB',
    layers: 32,
    hiddenSize: 3072,
    intermediateSize: 8192,
    attentionHeads: 32,
    kvHeads: 32,
    headDim: 96,
    vocabSize: 32064,
    activation: 'gelu_new',
    source: 'Hugging Face Cache',
    path: '~/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct',
    status: 'Ready',
    capabilities: ['Inference', 'LoRA / QLoRA', 'Full Fine-Tuning', 'Architecture Introspection'],
    avatar: '⚡',
    config: {
      architectures: ["Phi3ForCausalLM"],
      hidden_size: 3072,
      num_hidden_layers: 32,
      vocab_size: 32064
    }
  }
];

// Sample Tensors for Inspection Screen
export const TENSORS_DATA = {
  'model.layers.12.self_attn.q_proj.weight': {
    name: 'model.layers.12.self_attn.q_proj.weight',
    module: 'Multi-Head Attention (Query Projection)',
    shape: [4096, 4096],
    dtype: 'float16',
    parameters: 16777216,
    mean: -0.0002,
    stdDev: 0.0214,
    min: -0.4321,
    max: 0.4873,
    sparsity: 0.0,
    gridSize: 32
  },
  'model.layers.12.self_attn.k_proj.weight': {
    name: 'model.layers.12.self_attn.k_proj.weight',
    module: 'Multi-Head Attention (Key Projection)',
    shape: [1024, 4096],
    dtype: 'float16',
    parameters: 4194304,
    mean: 0.0001,
    stdDev: 0.0221,
    min: -0.3892,
    max: 0.4105,
    sparsity: 0.0,
    gridSize: 32
  },
  'model.layers.12.self_attn.v_proj.weight': {
    name: 'model.layers.12.self_attn.v_proj.weight',
    module: 'Multi-Head Attention (Value Projection)',
    shape: [1024, 4096],
    dtype: 'float16',
    parameters: 4194304,
    mean: -0.0004,
    stdDev: 0.0219,
    min: -0.4011,
    max: 0.3955,
    sparsity: 0.0,
    gridSize: 32
  },
  'model.layers.12.mlp.gate_proj.weight': {
    name: 'model.layers.12.mlp.gate_proj.weight',
    module: 'Feed Forward Network (SwiGLU Gate)',
    shape: [14336, 4096],
    dtype: 'float16',
    parameters: 58720256,
    mean: -0.0001,
    stdDev: 0.0188,
    min: -0.3654,
    max: 0.4219,
    sparsity: 0.0,
    gridSize: 32
  },
  'model.layers.12.mlp.down_proj.weight': {
    name: 'model.layers.12.mlp.down_proj.weight',
    module: 'Feed Forward Network (MLP Down Projection)',
    shape: [4096, 14336],
    dtype: 'float16',
    parameters: 58720256,
    mean: 0.0003,
    stdDev: 0.0175,
    min: -0.3421,
    max: 0.3899,
    sparsity: 0.0,
    gridSize: 32
  }
};

// Vocabulary Samples for Tokenizer View
export const SAMPLE_TOKENS_DICT = [
  { text: 'Hello', id: 15496, type: 'Word', color: '#38bdf8' },
  { text: ',', id: 11, type: 'Punctuation', color: '#94a3b8' },
  { text: ' how', id: 1267, type: 'Word', color: '#818cf8' },
  { text: ' are', id: 527, type: 'Word', color: '#a78bfa' },
  { text: ' you', id: 366, type: 'Word', color: '#c084fc' },
  { text: ' today', id: 1634, type: 'Word', color: '#f472b6' },
  { text: '?', id: 29973, type: 'Punctuation', color: '#94a3b8' },
  { text: '<|eot_id|>', id: 128009, type: 'Special Token', color: '#fb7185' }
];

// Sample Datasets for Dataset Manager
export const SAMPLE_DATASETS = [
  {
    id: 'alpaca_instructions',
    filename: 'alpaca_instructions.jsonl',
    type: 'Supervised Fine-Tuning (SFT)',
    format: 'instruction / input / output',
    records: 2000,
    size: '5.3 MB',
    status: 'Valid',
    avgTokenLength: 284,
    preview: [
      {
        instruction: 'Explain quantum computing in simple terms for a high school student.',
        input: '',
        output: 'Quantum computing is a type of computation that uses quantum mechanics principles, such as superposition and entanglement, allowing qubits to perform certain calculations exponentially faster than classical bits.'
      },
      {
        instruction: 'Write a Python function to compute the Fibonacci sequence up to N.',
        input: 'N = 10',
        output: 'def fibonacci(n):\n    fib = [0, 1]\n    while len(fib) < n:\n        fib.append(fib[-1] + fib[-2])\n    return fib[:n]'
      },
      {
        instruction: 'List the primary benefits of renewable energy technologies.',
        input: '',
        output: '1. Clean energy with zero or minimal carbon emissions.\n2. Inexhaustible supply from solar, wind, and geothermal sources.\n3. Long-term cost stabilization and energy independence.'
      }
    ]
  },
  {
    id: 'dpo_pairwise_preferences',
    filename: 'dpo_code_preferences.json',
    type: 'Direct Preference Optimization (DPO)',
    format: 'prompt / chosen / rejected',
    records: 1250,
    size: '8.1 MB',
    status: 'Valid',
    avgTokenLength: 412,
    preview: [
      {
        prompt: 'How do I safely parse untrusted JSON in Python with size limits?',
        chosen: 'To parse untrusted JSON safely, use ijson or validate payload length before json.loads to prevent memory DOS vulnerabilities.',
        rejected: 'Just do json.loads(payload) without any checks, Python handles it.'
      }
    ]
  }
];

// Historical Experiments Data
export const EXPERIMENT_HISTORY = [
  {
    id: 'exp-01',
    name: 'llama3.2-alpaca-lora-r16',
    model: 'Llama 3.2 8B',
    method: 'LoRA (Unsloth)',
    dataset: 'alpaca_instructions.jsonl',
    date: '2026-10-19 14:22',
    epochs: 3,
    lossStart: 2.14,
    lossFinal: 0.853,
    status: 'Completed',
    vramPeak: '5.1 GB',
    evalScore: '89.4%'
  },
  {
    id: 'exp-02',
    name: 'mistral-sft-code-corpus',
    model: 'Mistral 7B',
    method: 'SFT (Full Attention)',
    dataset: 'code_corpus_v2.jsonl',
    date: '2026-10-18 20:10',
    epochs: 2,
    lossStart: 2.89,
    lossFinal: 1.120,
    status: 'Completed',
    vramPeak: '5.8 GB',
    evalScore: '86.1%'
  },
  {
    id: 'exp-03',
    name: 'phi3-dpo-safety-alignment',
    model: 'Phi-3 Mini',
    method: 'DPO',
    dataset: 'dpo_code_preferences.json',
    date: '2026-10-17 09:45',
    epochs: 1,
    lossStart: 0.693,
    lossFinal: 0.312,
    status: 'Completed',
    vramPeak: '4.2 GB',
    evalScore: '92.0%'
  },
  {
    id: 'exp-04',
    name: 'qwen-qlora-medical-instruct',
    model: 'DeepSeek-Qwen 8B',
    method: 'QLoRA (4-bit)',
    dataset: 'medical_instruct.jsonl',
    date: '2026-10-16 11:30',
    epochs: 4,
    lossStart: 3.40,
    lossFinal: 1.450,
    status: 'Failed (OOM at step 450)',
    vramPeak: '6.1 GB',
    evalScore: 'N/A'
  }
];
