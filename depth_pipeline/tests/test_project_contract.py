from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


def test_synthetic_project_matches_machine_readable_contract() -> None:
    repository = Path(__file__).parents[2]
    schema = json.loads((repository / "docs" / "annotation-project.schema.json").read_text())
    example = json.loads((repository / "examples" / "synthetic-project.json").read_text())

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(example)
    assert example["schema_version"] == "palona.annotation-project/v1"
