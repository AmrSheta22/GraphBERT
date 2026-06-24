import unittest

import torch
from transformers import LongformerConfig, LongformerForMaskedLM

from graphbert.config import GraphAttentionConfig
from graphbert.graph_attention import GraphLongformerLayer
from graphbert.modeling import replacement_layer_indices, replace_longformer_layers


class LongformerGCNTests(unittest.TestCase):
    def tiny_model(self):
        config = LongformerConfig(
            vocab_size=101,
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=130,
            attention_window=[8, 8, 8, 8],
        )
        return LongformerForMaskedLM(config)

    def test_layer_placement(self):
        self.assertEqual(
            replacement_layer_indices(12, GraphAttentionConfig(num_replaced_layers=2)),
            [10, 11],
        )
        self.assertEqual(
            replacement_layer_indices(
                12,
                GraphAttentionConfig(num_replaced_layers=1, replacement_strategy="uniform"),
            ),
            [6],
        )
        self.assertEqual(
            replacement_layer_indices(
                12,
                GraphAttentionConfig(num_replaced_layers=2, replacement_strategy="intermediate"),
            ),
            [5, 6],
        )
        self.assertEqual(
            replacement_layer_indices(
                12,
                GraphAttentionConfig(
                    num_replaced_layers=2,
                    replacement_strategy="explicit",
                    layer_indices=[8, 5],
                ),
            ),
            [5, 8],
        )

    def test_forward_and_backward_with_global_token(self):
        model = self.tiny_model()
        graph_config = GraphAttentionConfig(
            num_replaced_layers=2,
            replacement_strategy="uniform",
            gcn_dropout=0.0,
        )
        self.assertEqual(replace_longformer_layers(model, graph_config), [0, 3])
        self.assertIsInstance(model.longformer.encoder.layer[0], GraphLongformerLayer)

        input_ids = torch.randint(0, model.config.vocab_size, (2, 32))
        attention_mask = torch.ones_like(input_ids)
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            labels=input_ids,
        )
        self.assertEqual(output.logits.shape, (2, 32, model.config.vocab_size))
        output.loss.backward()
        self.assertIsNotNone(model.longformer.encoder.layer[0].gcn.weight.grad)


if __name__ == "__main__":
    unittest.main()
