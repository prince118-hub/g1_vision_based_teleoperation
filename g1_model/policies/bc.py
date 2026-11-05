"""BC (code name bc): behavioural cloning, the thesis baseline. Entry module only (aliases).

Real code: BCConfig, BCPolicy, build_bc in g1_model/models.py.
why: see DESIGN_NOTES.md#policies-bc
"""
from g1_model.models import BCConfig, BCPolicy, build_bc

#: Aliases of the real objects, under the same names in every entry module.
Config = BCConfig
Policy = BCPolicy
build = build_bc

THESIS_NAME = "BC"
CODE_NAME = "bc"
ROLE = "headline"

#: `Config(**CONFIG_KWARGS)` is the config `tools/train_bc.py` builds for the
#: thesis runs (outside the overfit-10 gate). Checked by g1_model/test_policies.py.
CONFIG_KWARGS = dict(obs_window=12, chunk_size=1, batch_size=256)

#: The model arguments of the thesis runs' `tools/train_bc.py train` command
#: (budget, data and output arguments are in the launcher). Checked by the test.
TRAIN_ARGS = "--model bc --chunk 1 --obs-window 12 --batch 256"
