"""IAM Condition expressions only get a small CEL subset: no regex matches().

terraform validate can't see this; the API rejects it only at apply time, in Cloud Shell.
Allowed string functions: startsWith, endsWith, extract (plus ==, !=, in, &&, ||, !).
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNSUPPORTED = ("matches(", "contains(", "size(", "lowerAscii(", "exists(")


def test_iam_conditions_use_only_supported_functions():
    for tf in sorted(ROOT.glob("infra/**/*.tf")):
        text = tf.read_text()
        for fn in UNSUPPORTED:
            assert not re.search(rf"resource\.\w+\.{re.escape(fn)}", text), f"{tf}: {fn}"
