import math
import unittest

import torch
from transformers import LongformerConfig, LongformerModel

from graphbert.mldr import merge_topk, ndcg_at_k, recall_at_k
from graphbert.retrieval import LongContextRetriever, RetrievalConfig, maxsim_scores


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
        documents = torch.tensor([
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [100.0, 100.0]],
        ])
        document_mask = torch.tensor([[True, True], [True, False]])
        scores = maxsim_scores(queries, query_mask, documents, document_mask)
        self.assertTrue(torch.allclose(scores, torch.tensor([[2.0, 1.0]])))


if __name__ == "__main__":
    unittest.main()