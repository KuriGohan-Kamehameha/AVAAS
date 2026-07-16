from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from webui import prompts


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PRESENTATIONS = {
    "satraj.en-ca.neutral": {
        "identity": "satraj",
        "language_tag": "en-CA",
        "accent": "general-north-american",
        "delivery": "neutral",
    },
    "piranesi.en-gb.neutral": {
        "identity": "piranesi",
        "language_tag": "en-GB",
        "accent": "received-pronunciation",
        "delivery": "neutral",
    },
}
EXPECTED_DELIVERIES = {
    "neutral",
    "warm",
    "authoritative",
    "urgent",
    "whisper",
    "projected",
}


def _copy_prompt_sources(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "prompts", root / "prompts")
    return root


def test_one_human_source_has_two_bounded_default_presentations() -> None:
    profile = prompts.load_identities(ROOT)
    assert profile["speaker_id"] == "satraj"
    assert profile["voice_model_id"] == "satraj-piranesi"
    assert profile["recording_profile"] == {
        "id": "satraj.en-ca.natural",
        "language_tag": "en-CA",
        "accent": "natural",
        "default_delivery": "neutral",
        "accent_imitation_required": False,
    }
    assert profile["additional_languages"] == []

    identities = {item["id"]: item for item in profile["identities"]}
    assert identities["satraj"]["default_presentation"] == "satraj.en-ca.neutral"
    assert identities["piranesi"]["default_presentation"] == "piranesi.en-gb.neutral"

    presentations = {item["id"]: item for item in profile["presentations"]}
    assert set(presentations) == set(EXPECTED_PRESENTATIONS)
    for presentation_id, expected in EXPECTED_PRESENTATIONS.items():
        assert presentations[presentation_id] == {"id": presentation_id, **expected}
        assert len(presentation_id) <= 64


def test_compiled_prompt_metadata_separates_performance_from_synthesis() -> None:
    records = prompts.compile_prompts(ROOT)
    paired = [record for record in records if record["identity"] != "shared"]
    shared = [record for record in records if record["identity"] == "shared"]
    assert paired and shared

    for record in records:
        assert record["recording_profile_id"] == "satraj.en-ca.natural"
        assert record["recording_language_tag"] == "en-CA"
        assert record["recording_accent"] == "natural"
        assert record["recording_delivery"] in EXPECTED_DELIVERIES
        assert 1 <= len(record["recording_safety_cue"]) <= 256
        assert isinstance(record["calibration_optional"], bool)
        # A desired synthesis accent is never asserted as an observed human take.
        assert record["recording_accent"] not in {
            "general-north-american",
            "received-pronunciation",
        }

    for record in paired:
        expected = f"{record['identity']}.{'en-ca' if record['identity'] == 'satraj' else 'en-gb'}.neutral"
        assert record["synthesis_presentation"] == expected
        assert record["eligible_presentations"] == [expected]

    for record in shared:
        assert record["synthesis_presentation"] is None
        assert record["eligible_presentations"] == [
            "satraj.en-ca.neutral",
            "piranesi.en-gb.neutral",
        ]


def test_calibration_sets_are_complete_optional_and_physically_safe() -> None:
    profile = prompts.load_identities(ROOT)
    sets = profile["calibration_sets"]
    assert {item["delivery"] for item in sets} == EXPECTED_DELIVERIES
    assert len({prompt_id for item in sets for prompt_id in item["prompt_ids"]}) == sum(
        len(item["prompt_ids"]) for item in sets
    )
    assert all(1 <= len(item["prompt_ids"]) <= 16 for item in sets)
    assert all(item["optional"] is (item["delivery"] != "neutral") for item in sets)
    assert "shout" not in {item["delivery"] for item in sets}

    by_base: dict[str, list[dict]] = {}
    for record in prompts.compile_prompts(ROOT):
        by_base.setdefault(record["variant_of"], []).append(record)
    for calibration in sets:
        for base_id in calibration["prompt_ids"]:
            assert base_id in by_base
            for record in by_base[base_id]:
                assert record["recording_delivery"] == calibration["delivery"]
                assert record["recording_safety_cue"] == calibration["safety_cue"]
                assert record["calibration_optional"] is calibration["optional"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["identities"][1].update(
                {"default_presentation": "unknown.presentation"}
            ),
            "default presentation",
        ),
        (
            lambda value: value["presentations"].append(value["presentations"][0]),
            "presentation",
        ),
        (
            lambda value: value["recording_profile"].update(
                {"accent_imitation_required": True}
            ),
            "accent imitation",
        ),
        (
            lambda value: value["additional_languages"].append("guessed"),
            "additional language",
        ),
        (
            lambda value: value["calibration_sets"][0]["prompt_ids"].append(
                "sec99_999"
            ),
            "calibration prompt",
        ),
        (
            lambda value: value["calibration_sets"][1].update(
                {"delivery": "shout"}
            ),
            "calibration delivery",
        ),
    ],
)
def test_invalid_or_guessed_presentation_metadata_fails_closed(
    tmp_path: Path, mutate, message: str
) -> None:
    root = _copy_prompt_sources(tmp_path)
    path = root / "prompts" / "identities.json"
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value))
    with pytest.raises(prompts.PromptContractError, match=message):
        prompts.compile_prompts(root)
