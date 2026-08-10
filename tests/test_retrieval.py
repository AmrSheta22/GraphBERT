import math
import unittest

import torch
from transformers import LongformerConfig, LongformerForMaskedLM, LongformerModel

from graphbert.config import GraphAttentionConfig
from graphbert.mldr import merge_topk, ndcg_at_k, recall_at_k
from graphbert.modeling import add_longformer_appnp_adapters
from graphbert.retrieval import (
    LongContextRetriever,
    RetrievalConfig,
    configure_trainable_parameters,
    maxsim_scores,
)
from graphbert.synthetic_retrieval import build_synthetic_benchmark, make_synthetic_triplet


class RetrievalEvaluationTests(unittest.TestCase):
    def test_ndcg_and_recall(self):
        rankings = {"q1": ["d2", "d1"], "q2": ["d3"]}
        relevance = {"q1": {"d1"}, "q2": {"d3"}}
        expected = (1.0 / math.log2(3) + 1.0) / 2.0
        self.assertAlmostEqual(ndcg_at_k(rankings, relevance, 10), expected)
        self.assertEqual(recall_at_k(rankings, relevance, 100), 1.0)

    def test_merge_topk(self):
        current_scores = torch.tensor([[0.9, 0.4]])
        current_indices = torch.tensor([[4, 1]])
        new_scores = torch.tensor([[0.8, 0.95]])
        new_indices = torch.tensor([[7, 9]])
        scores, indices = merge_topk(current_scores, current_indices, new_scores, new_indices, 3)
        self.assertTrue(torch.equal(indices, torch.tensor([[9, 4, 7]])))
        self.assertTrue(torch.allclose(scores, torch.tensor([[0.95, 0.9, 0.8]])))

    def test_retriever_single_and_token_embeddings(self):
        config = LongformerConfig(
            vocab_size=50,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=66,
            attention_window=[8, 8],
        )
        retriever = LongContextRetriever(
            LongformerModel(config),
            RetrievalConfig(source_model="tiny", graph_config={}, projection_dim=8),
        )
        batch = {
            "input_ids": torch.randint(0, 50, (2, 16)),
            "attention_mask": torch.ones((2, 16), dtype=torch.long),
        }
        single = retriever.encode_single(batch)
        tokens, mask = retriever.encode_tokens(batch)
        self.assertEqual(single.shape, (2, 8))
        self.assertEqual(tokens.shape, (2, 16, 8))
        self.assertTrue(torch.allclose(single.norm(dim=-1), torch.ones(2), atol=1e-5))
        self.assertTrue(mask.all())

    def test_maxsim_masks_padding(self):
        queries = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        query_mask = torch.tensor([[True, True]])
        documents = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [100.0, 100.0]],
            ]
        )
        document_mask = torch.tensor([[True, True], [True, False]])
        scores = maxsim_scores(queries, query_mask, documents, document_mask)
        self.assertTrue(torch.allclose(scores, torch.tensor([[2.0, 1.0]])))

    def test_head_only_training_freezes_encoder(self):
        config = LongformerConfig(
            vocab_size=50,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=66,
            attention_window=[8],
        )
        retriever = LongContextRetriever(
            LongformerModel(config),
            RetrievalConfig(source_model="tiny", graph_config={}, projection_dim=8),
        )
        summary = configure_trainable_parameters(retriever, "head")
        self.assertGreater(summary["trainable_parameters"], 0)
        self.assertTrue(all(not parameter.requires_grad for parameter in retriever.encoder.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in retriever.projection.parameters()))

    def test_adapter_only_training_selects_appnp_parameters(self):
        config = LongformerConfig(
            vocab_size=50,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=66,
            attention_window=[8],
        )
        mlm_model = LongformerForMaskedLM(config)
        add_longformer_appnp_adapters(
            mlm_model,
            GraphAttentionConfig(num_replaced_layers=1, appnp_steps=2),
        )
        retriever = LongContextRetriever(
            mlm_model.longformer,
            RetrievalConfig(source_model="tiny", graph_config={}),
        )
        summary = configure_trainable_parameters(retriever, "adapters")
        trainable_names = [name for name, parameter in retriever.named_parameters() if parameter.requires_grad]
        self.assertGreater(summary["trainable_parameters"], 0)
        self.assertTrue(
            all("appnp_projection" in name or name.endswith("appnp_gate") for name in trainable_names)
        )

    def test_synthetic_data_controls_needle_position_and_relevance(self):
        early = make_synthetic_triplet(0, document_words=30, seed=7, position_fractions=(0.0,))
        late = make_synthetic_triplet(0, document_words=30, seed=7, position_fractions=(1.0,))
        self.assertTrue(early["positive"].startswith("personalized passkey is key-"))
        self.assertIn("personalized passkey is key-", late["positive"])
        self.assertGreater(late["positive"].index("personalized passkey"), 100)

        benchmark = build_synthetic_benchmark(
            num_queries=6,
            document_words=30,
            seed=11,
            position_fractions=(0.1, 0.5, 0.9),
        )
        self.assertEqual(len(benchmark.queries), 6)
        self.assertEqual(len(benchmark.documents), 12)
        self.assertEqual({len(relevant) for relevant in benchmark.relevance.values()}, {1})


if __name__ == "__main__":
    unittest.main()
