import unittest
from copy import deepcopy
from tempfile import TemporaryDirectory

import torch
from transformers import LongformerConfig, LongformerForMaskedLM

from graphbert.config import GraphAttentionConfig
from graphbert.graph_attention import GraphLongformerLayer
from graphbert.modeling import add_longformer_gcn_adapters, load_graph_bert_checkpoint, selected_layer_indices


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
            selected_layer_indices(12, GraphAttentionConfig(num_replaced_layers=2)),
            [10, 11],
        )
        self.assertEqual(
            selected_layer_indices(
                12,
                GraphAttentionConfig(num_replaced_layers=1, replacement_strategy="uniform"),
            ),
            [6],
        )
        self.assertEqual(
            selected_layer_indices(
                12,
                GraphAttentionConfig(num_replaced_layers=2, replacement_strategy="intermediate"),
            ),
            [5, 6],
        )
        self.assertEqual(
            selected_layer_indices(
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
        self.assertEqual(add_longformer_gcn_adapters(model, graph_config), [0, 3])
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
        adapter = model.longformer.encoder.layer[0]
        self.assertIsNotNone(adapter.gcn.weight.grad)
        self.assertEqual(adapter.gcn.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(adapter.gcn_gate.grad)
        self.assertGreater(adapter.gcn_gate.grad.abs().item(), 0.0)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        second_loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            labels=input_ids,
        ).loss
        second_loss.backward()
        self.assertGreater(adapter.gcn.weight.grad.abs().sum().item(), 0.0)

    def test_zero_gate_is_exactly_baseline_equivalent(self):
        baseline = self.tiny_model().eval()
        adapted = deepcopy(baseline)
        graph_config = GraphAttentionConfig(
            num_replaced_layers=2,
            replacement_strategy="uniform",
            gcn_dropout=0.0,
            gcn_initial_scale=0.0,
        )
        add_longformer_gcn_adapters(adapted, graph_config)
        adapted.eval()

        input_ids = torch.randint(0, baseline.config.vocab_size, (2, 32))
        attention_mask = torch.ones_like(input_ids)
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1
        with torch.no_grad():
            baseline_logits = baseline(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            ).logits
            adapted_logits = adapted(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            ).logits

        self.assertTrue(torch.equal(baseline_logits, adapted_logits))

    def test_nonzero_gate_changes_output(self):
        baseline = self.tiny_model().eval()
        adapted = deepcopy(baseline)
        add_longformer_gcn_adapters(
            adapted,
            GraphAttentionConfig(
                num_replaced_layers=1,
                gcn_dropout=0.0,
                gcn_activation="none",
                gcn_initial_scale=0.1,
            ),
        )
        adapted.eval()
        input_ids = torch.randint(0, baseline.config.vocab_size, (1, 32))
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            baseline_logits = baseline(input_ids=input_ids, attention_mask=attention_mask).logits
            adapted_logits = adapted(input_ids=input_ids, attention_mask=attention_mask).logits
        self.assertFalse(torch.equal(baseline_logits, adapted_logits))

    def test_residual_adapter_checkpoint_round_trip(self):
        model = self.tiny_model().eval()
        graph_config = GraphAttentionConfig(
            num_replaced_layers=1,
            gcn_dropout=0.0,
            gcn_initial_scale=0.05,
        )
        add_longformer_gcn_adapters(model, graph_config)
        input_ids = torch.randint(0, model.config.vocab_size, (1, 32))
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            expected = model(input_ids=input_ids, attention_mask=attention_mask).logits

        with TemporaryDirectory(ignore_cleanup_errors=True) as checkpoint:
            model.save_pretrained(checkpoint)
            restored = load_graph_bert_checkpoint(checkpoint, graph_config).eval()
            with torch.no_grad():
                actual = restored(input_ids=input_ids, attention_mask=attention_mask).logits

        self.assertTrue(torch.equal(expected, actual))

    def test_vanilla_checkpoint_can_receive_adapters_when_loaded(self):
        baseline = self.tiny_model().eval()
        graph_config = GraphAttentionConfig(num_replaced_layers=1, gcn_dropout=0.0)
        input_ids = torch.randint(0, baseline.config.vocab_size, (1, 32))
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            expected = baseline(input_ids=input_ids, attention_mask=attention_mask).logits

        with TemporaryDirectory(ignore_cleanup_errors=True) as checkpoint:
            baseline.save_pretrained(checkpoint)
            adapted = load_graph_bert_checkpoint(checkpoint, graph_config).eval()
            with torch.no_grad():
                actual = adapted(input_ids=input_ids, attention_mask=attention_mask).logits

        self.assertIsInstance(adapted.longformer.encoder.layer[-1], GraphLongformerLayer)
        self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
