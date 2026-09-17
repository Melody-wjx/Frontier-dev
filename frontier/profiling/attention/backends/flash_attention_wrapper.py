"""Profile the installed vLLM FlashAttention paged-cache kernel.

Mixed prefill/decode is ONE varlen call, as in vLLM V1. Its entire time is
recorded in attn_decode, with attn_prefill exactly zero. The predictor recognizes
FLASH_ATTN profiles and consumes this unified mixed cost once, not twice.
This is operator profiling, not a claim to measure the engine's async overlap.
"""

import torch

from frontier.profiling.attention.backends.base_attention_wrapper import BaseAttentionWrapper
from frontier.profiling.common.constants import OperationMetrics


class FlashAttentionWrapper(BaseAttentionWrapper):
    _inst = None
    unified_mixed_attention = True
    requires_disjoint_blocks = True

    def init(self, model_config, parallel_config, block_size, device):
        from vllm._custom_ops import reshape_and_cache_flash
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        from vllm.attention.utils.fa_utils import get_flash_attn_version

        super().init(model_config, parallel_config, block_size, device)
        self.cache_write = reshape_and_cache_flash
        self.attention = flash_attn_varlen_func
        self.fa_version = get_flash_attn_version()
        self.softmax_scale = 1.0 / (self.head_dim ** 0.5)
        self.scale = torch.ones((), dtype=torch.float32, device=device)
        self.is_metadata_initialized = False

    def get_cache_block(self, num_blocks, **kwargs):
        # Native FlashAttentionBackend layout: [2, blocks, block_size, heads, dim].
        return torch.randn(2, num_blocks, self.block_size, self.num_kv_heads,
                           self.head_dim, **kwargs)

    def begin_forward(self, seq_metadata_list):
        # AttentionWrapper supplies prefill first, then decode, as do its Q/K/V.
        ordered = sorted(seq_metadata_list, key=lambda seq: not seq.is_prompt)
        query_starts, seq_lens, slots, tables, query_lens = [0], [], [], [], []
        self.contains_prefill = any(seq.is_prompt for seq in ordered)
        self.contains_decode = any(not seq.is_prompt for seq in ordered)
        for seq in ordered:
            if seq.block_table is None:
                raise ValueError("FlashAttention profiling requires a physical block table")
            qlen = seq.prompt_chunk_len if seq.is_prompt else 1
            length = (seq.seq.get_num_prompt_tokens_processed() + qlen
                      if seq.is_prompt else seq.seq.get_len())
            query_lens.append(qlen)
            query_starts.append(query_starts[-1] + qlen)
            seq_lens.append(length)
            tables.append(seq.block_table[:(length + self.block_size - 1) // self.block_size])
            for token in range(length - qlen, length):
                slots.append(seq.block_table[token // self.block_size] * self.block_size
                             + token % self.block_size)
        width = max(map(len, tables))
        self.block_table = torch.tensor([table + [0] * (width - len(table)) for table in tables],
                                        dtype=torch.int32, device=self.device)
        self.query_start_loc = torch.tensor(query_starts, dtype=torch.int32, device=self.device)
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
        self.slot_mapping = torch.tensor(slots, dtype=torch.int64, device=self.device)
        self.max_query_len = max(query_lens)
        self.max_seq_len = max(seq_lens)
        self.is_metadata_initialized = True

    def end_forward(self):
        self.is_metadata_initialized = False
        self.slot_mapping = None

    def forward(self, query, key, value, kv_cache, softmax_scale=1.0, layer_id=None):
        if not self.is_metadata_initialized:
            raise RuntimeError("FlashAttention metadata is not initialized")
        if softmax_scale != self.softmax_scale:
            raise ValueError("softmax_scale differs from model configuration")
        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
            query = query.reshape(-1, self.num_q_heads, self.head_dim)
            key = key.reshape(-1, self.num_kv_heads, self.head_dim)
            value = value.reshape(-1, self.num_kv_heads, self.head_dim)
        output = torch.empty_like(query)
        key_cache, value_cache = kv_cache.unbind(0)
        with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
            self.cache_write(key, value, key_cache, value_cache, self.slot_mapping,
                             "auto", self.scale, self.scale)
        op = OperationMetrics.ATTN_DECODE if self.contains_decode else OperationMetrics.ATTN_PREFILL
        with self.get_timer(op, layer_id):
            self.attention(q=query, k=key_cache, v=value_cache, out=output,
                           cu_seqlens_q=self.query_start_loc, max_seqlen_q=self.max_query_len,
                           seqused_k=self.seq_lens, max_seqlen_k=self.max_seq_len,
                           softmax_scale=self.softmax_scale, causal=True,
                           block_table=self.block_table, fa_version=self.fa_version,
                           num_splits=0)
        # Emit the inactive physical scope for the existing CSV schema. In
        # CUDA_EVENT mode AttentionWrapper normalizes this empty scope to zero.
        inactive = OperationMetrics.ATTN_PREFILL if self.contains_decode else OperationMetrics.ATTN_DECODE
        with self.get_timer(inactive, layer_id):
            pass
        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            output = output.reshape(-1, self.num_q_heads * self.head_dim)
        return output
