import unittest
from copy import deepcopy
from tempfile import TemporaryDirectory

import torch
from transformers import LongformerConfig, LongformerForMaskedLM

from graphbert.config import GraphAttentionConfig
from graphbert.graph_attention import APPNPLongformerLayer
from graphbert.modeling import add_longformer_appnp_adapters, load_graph_bert_checkpoint, selected_layer_indices


class LongformerAPPNPTests(unittest.TestCase):
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
            appnp_dropout=0.0,
            appnp_steps=2,
        )
        self.assertEqual(add_longformer_appnp_adapters(model, graph_config), [0, 3])
        self.assertIsInstance(model.longformer.encoder.layer[0], APPNPLongformerLayer)

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
        self.assertIsNotNone(adapter.appnp_projection.weight.grad)
        self.assertEqual(adapter.appnp_projection.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(adapter.appnp_gate.grad)
        self.assertGreater(adapter.appnp_gate.grad.abs().item(), 0.0)

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
        self.assertGreater(adapter.appnp_projection.weight.grad.abs().sum().item(), 0.0)

    def test_zero_gate_is_exactly_baseline_equivalent(self):
        baseline = self.tiny_model().eval()
        adapted = deepcopy(baseline)
        graph_config = GraphAttentionConfig(
            num_replaced_layers=2,
            replacement_strategy="uniform",
            appnp_dropout=0.0,
            appnp_initial_scale=0.0,
            appnp_steps=2,
        )
        add_longformer_appnp_adapters(adapted, graph_config)
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
        add_longformer_appnp_adapters(
            adapted,
            GraphAttentionConfig(
                num_replaced_layers=1,
                appnp_dropout=0.0,
                appnp_activation="none",
                appnp_initial_scale=0.1,
                appnp_steps=2,
            ),
        )
        adapted.eval()
        input_ids = torch.randint(0, baseline.config.vocab_size, (1, 32))
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            baseline_logits = baseline(input_ids=input_ids, attention_mask=attention_mask).logits
            adapted_logits = adapted(input_ids=input_ids, attention_mask=attention_mask).logits
        self.assertFalse(torch.equal(baseline_logits, adapted_logits))

    def test_more_appnp_steps_change_propagation(self):
        model_k1 = self.tiny_model().eval()
        model_k2 = deepcopy(model_k1)
        common = {
            "num_replaced_layers": 1,
            "appnp_dropout": 0.0,
            "appnp_activation": "none",
            "appnp_initial_scale": 1.0,
            "appnp_teleport_probability": 0.1,
        }
        add_longformer_appnp_adapters(
            model_k1,
            GraphAttentionConfig(appnp_steps=1, **common),
        )
        add_longformer_appnp_adapters(
            model_k2,
            GraphAttentionConfig(appnp_steps=2, **common),
        )
        input_ids = torch.randint(0, model_k1.config.vocab_size, (1, 32))
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits_k1 = model_k1(input_ids=input_ids, attention_mask=attention_mask).logits
            logits_k2 = model_k2(input_ids=input_ids, attention_mask=attention_mask).logits
        self.assertFalse(torch.equal(logits_k1, logits_k2))

    def test_residual_adapter_checkpoint_round_trip(self):
        model = self.tiny_model().eval()
        graph_config = GraphAttentionConfig(
            num_replaced_layers=1,
            appnp_dropout=0.0,
            appnp_initial_scale=0.05,
            appnp_steps=2,
        )
        add_longformer_appnp_adapters(model, graph_config)
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
        graph_config = GraphAttentionConfig(
            num_replaced_layers=1,
            appnp_dropout=0.0,
            appnp_steps=2,
        )
        input_ids = torch.randint(0, baseline.config.vocab_size, (1, 32))
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            expected = baseline(input_ids=input_ids, attention_mask=attention_mask).logits

        with TemporaryDirectory(ignore_cleanup_errors=True) as checkpoint:
            baseline.save_pretrained(checkpoint)
            adapted = load_graph_bert_checkpoint(checkpoint, graph_config).eval()
            with torch.no_grad():
                actual = adapted(input_ids=input_ids, attention_mask=attention_mask).logits

        self.assertIsInstance(adapted.longformer.encoder.layer[-1], APPNPLongformerLayer)
        self.assertTrue(torch.equal(expected, actual))

    def test_checkpoint_with_legacy_layernorm_gamma_beta_keys_loads(self):
        baseline = self.tiny_model().eval()
        graph_config = GraphAttentionConfig(
            num_replaced_layers=1,
            appnp_dropout=0.0,
            appnp_steps=2,
        )
        input_ids = torch.randint(0, baseline.config.vocab_size, (1, 32))
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            expected = baseline(input_ids=input_ids, attention_mask=attention_mask).logits

        with TemporaryDirectory(ignore_cleanup_errors=True) as checkpoint:
            baseline.config.save_pretrained(checkpoint)
            checkpoint_path = f"{checkpoint}/pytorch_model.bin"
            state_dict = baseline.state_dict()
            legacy_state_dict = {}
            for key, value in state_dict.items():
                if key.endswith(".LayerNorm.weight"):
                    key = f"{key[:-len('.weight')]}.gamma"
                elif key.endswith(".LayerNorm.bias"):
                    key = f"{key[:-len('.bias')]}.beta"
                legacy_state_dict[key] = value
            torch.save(legacy_state_dict, checkpoint_path)

            adapted = load_graph_bert_checkpoint(checkpoint, graph_config).eval()
            with torch.no_grad():
                actual = adapted(input_ids=input_ids, attention_mask=attention_mask).logits

        self.assertIsInstance(adapted.longformer.encoder.layer[-1], APPNPLongformerLayer)
        self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
