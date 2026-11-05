"""One entry module per thesis model: a map for reading the code, not new code.

WARNING: `g1_model.policies.act` (thesis model "ACT") is NOT `g1_model.act` (its implementation).
Guides: docs/MODELS_GUIDE.md, docs/ACT_FAITHFULNESS.md. why: see DESIGN_NOTES.md#policies-package
"""
from . import act, act_12, act_lstm, act_lstm_12, bc

#: The three D8 policies, by thesis name.
HEADLINE = {"BC": bc, "ACT": act, "ACT-LSTM": act_lstm}

#: The short-memory ablation pair (W_o = 12), by thesis name.
ABLATION = {"ACT-12": act_12, "ACT-LSTM-12": act_lstm_12}

#: Every entry by its code name: the value of `tools/train_bc.py --model` and of
#: a run's `metadata.json` -> `config.notes.model`.
BY_CODE_NAME = {"bc": bc, "act_orig": act, "act_lstm_full": act_lstm,
                "act": act_12, "act_lstm": act_lstm_12}

__all__ = ["bc", "act", "act_lstm", "act_12", "act_lstm_12",
           "HEADLINE", "ABLATION", "BY_CODE_NAME"]
