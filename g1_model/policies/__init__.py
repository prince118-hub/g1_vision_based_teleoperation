"""One entry module per thesis model: a map for reading the code, not new code.

    thesis name    entry module                      code name (--model)   real class
    -----------    ------------------------------    -------------------   ------------------------------------------
    BC             g1_model/policies/bc.py           bc                    models.BCPolicy           (g1_model/models.py)
    ACT            g1_model/policies/act.py          act_orig              act.ACTPolicy             (g1_model/act.py)
    ACT-LSTM       g1_model/policies/act_lstm.py     act_lstm_full         act_memory.ACTLSTMFullPolicy (g1_model/act_memory.py)
    ACT-12         g1_model/policies/act_12.py       act                   act.ACTPolicy             (ablation, W_o = 12)
    ACT-LSTM-12    g1_model/policies/act_lstm_12.py  act_lstm              act.ACTPolicy, use_lstm   (ablation, W_o = 12)

The thesis names follow CLAUDE.md §7 D8 (2026-10-25b, pending adviser approval):
the headline comparison is BC (batch 256), ACT = act_orig and ACT-LSTM =
act_lstm_full; ACT-12 / ACT-LSTM-12 are the short-memory ablation.

Every entry module holds ONLY a docstring, imports/aliases of the real objects
(`Config`, `Policy`, `build`) and the settings the thesis runs use, written as
data (`CONFIG_KWARGS`, `TRAIN_ARGS`). Nothing here defines maths, a default or a
code path: the trainer (`tools/train_bc.py`, also reachable as
`tools/train_policy.py`), the deployment harness (`tools/deploy.py`) and every
checkpoint keep using the real modules directly. `g1_model/test_policies.py`
checks that each entry builds bit-for-bit the model the trainer builds.

The full study guide is `docs/MODELS_GUIDE.md`; the comparison with the original
ACT is `docs/ACT_FAITHFULNESS.md`.

NAME CLASH TO KEEP IN MIND: `g1_model.policies.act` (this package's entry for the
thesis model "ACT") is NOT `g1_model.act` (the file that implements ACTPolicy).
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
