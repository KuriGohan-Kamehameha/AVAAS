from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from webui import prompts
from webui.script_parser import parse_script


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_HASHES = {
    "harvard_100": "eb79c0bfadc4b3988ee7caa343b84691c2ae1ded100f61fdb6966ef5b5bb9c72",
    "cmu_arctic_400": "58f6d9d3e2d7d8d836eb3a2d7eb36ebff824657e6ba5191099891aaa8f87de25",
    "avaas_local": "939172b760f9ba57f6e0b72afad2f40d6d285bb7863f29257f264b33d4898c71",
}
REQUIRED_FIELDS = {
    "id",
    "variant_of",
    "identity",
    "speaker_id",
    "voice_model_id",
    "text",
    "kind",
    "section",
    "source",
    "corpus_version",
}


def _copy_prompt_sources(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "prompts").mkdir(parents=True)
    shutil.copytree(ROOT / "prompts", root / "prompts", dirs_exist_ok=True)
    return root


def test_identity_contract_is_explicit_and_unambiguous() -> None:
    identities = prompts.load_identities(ROOT)
    assert identities["schema"] == "avaas/identities@v1"
    assert identities["speaker_id"] == "satraj"
    assert identities["voice_model_id"] == "satraj-piranesi"
    assert [(item["id"], item["display_name"]) for item in identities["identities"]] == [
        ("satraj", "Satraj"),
        ("piranesi", "Piranesi"),
    ]
    assert {entry["name"] for entry in identities["pronunciation_lexicon"]} == {
        "Satraj",
        "Piranesi",
    }
    assert all(entry["status"] == "speaker-validation-required" for entry in identities["pronunciation_lexicon"])


def test_required_corpora_have_exact_counts_and_hashes() -> None:
    provenance = prompts.load_provenance(ROOT)
    assert provenance["schema"] == "avaas/prompt-provenance@v1"
    by_id = {item["id"]: item for item in provenance["sources"]}
    assert set(by_id) == {"harvard_100", "cmu_arctic_400", "avaas_local"}
    assert prompts.PINNED_SOURCE_HASHES["2026.07.16"] == EXPECTED_SOURCE_HASHES

    for source_id, expected_count in (("harvard_100", 100), ("cmu_arctic_400", 400)):
        source = by_id[source_id]
        path = ROOT / "prompts" / source["path"]
        lines = prompts.read_corpus_lines(path)
        assert len(lines) == expected_count
        assert source["expected_lines"] == expected_count
        assert prompts.sha256_file(path) == source["sha256"] == EXPECTED_SOURCE_HASHES[source_id]
        assert source["source_url"].startswith(("https://", "http://"))
        assert source["source_revision"]
        assert source["license_name"]
        assert source["license_url"].startswith(("https://", "http://"))

    harvard = by_id["harvard_100"]
    assert harvard["materialization"] == "build-only"
    assert harvard["redistribution_allowed"] is False


def test_compilation_is_stable_complete_and_bounded() -> None:
    first = prompts.compile_prompts(ROOT)
    second = prompts.compile_prompts(ROOT)
    assert first == second
    assert 800 <= len(first) <= prompts.MAX_COMPILED_PROMPTS
    assert len({record["id"] for record in first}) == len(first)
    assert all(REQUIRED_FIELDS <= record.keys() for record in first)
    assert all(record["speaker_id"] == "satraj" for record in first)
    assert all(record["voice_model_id"] == "satraj-piranesi" for record in first)
    assert all(0 < len(record["text"]) <= prompts.MAX_PROMPT_CHARS for record in first)
    assert sum(record["source"] == "harvard_100" for record in first) == 100
    assert sum(record["source"] == "cmu_arctic_400" for record in first) == 400


def test_every_identity_template_compiles_to_exactly_one_pair() -> None:
    records = prompts.compile_prompts(ROOT)
    paired: dict[str, list[dict]] = {}
    for record in records:
        if record["identity"] in {"satraj", "piranesi"}:
            paired.setdefault(record["variant_of"], []).append(record)

    assert len(paired) >= 35
    for base_id, variants in paired.items():
        assert {item["identity"] for item in variants} == {"satraj", "piranesi"}, base_id
        assert {item["id"] for item in variants} == {
            f"{base_id}__satraj",
            f"{base_id}__piranesi",
        }
        by_identity = {item["identity"]: item["text"] for item in variants}
        assert by_identity["satraj"] != by_identity["piranesi"], base_id
        normalized_satraj = re.sub(r"[^A-Za-z]", "", by_identity["satraj"]).lower()
        normalized_piranesi = re.sub(r"[^A-Za-z]", "", by_identity["piranesi"]).lower()
        assert "satraj" in normalized_satraj, base_id
        assert "piranesi" in normalized_piranesi, base_id


def test_identity_material_covers_bequest_phone_and_wakeword_use() -> None:
    records = prompts.compile_prompts(ROOT)
    paired_text = "\n".join(
        record["text"] for record in records if record["identity"] in {"satraj", "piranesi"}
    ).lower()
    required_fragments = (
        "this is satraj",
        "this is piranesi",
        "s a t r a j",
        "p i r a n e s i",
        "voicemail",
        "call you back",
        "please hold",
        "transfer",
        "wrong person",
        "hey piranesi",
        "speaking as piranesi",
        "bequeathed this voice",
        "noisy room",
    )
    for fragment in required_fragments:
        assert fragment in paired_text


def test_user_facing_prompt_text_never_uses_sat_as_a_name() -> None:
    # Lowercase "sat" remains a normal English past-tense verb in CMU ARCTIC.
    forbidden = re.compile(r"(?<![A-Za-z])Sat(?![A-Za-z])")
    records = prompts.compile_prompts(ROOT)
    assert not [(record["id"], record["text"]) for record in records if forbidden.search(record["text"])]
    assert not forbidden.search(prompts.render_markdown(ROOT))


def test_script_parser_preserves_server_section_shape() -> None:
    sections = parse_script(ROOT)
    assert len(sections) >= 13
    assert sum(len(section["prompts"]) for section in sections) == len(prompts.compile_prompts(ROOT))
    for section in sections:
        assert {"section", "num", "title", "target_min", "prompts"} <= section.keys()
        for prompt in section["prompts"]:
            assert {"id", "section", "idx", "text", "kind", "slug"} <= prompt.keys()


def test_generated_markdown_is_checked_for_drift() -> None:
    generated = subprocess.run(
        [sys.executable, "-m", "webui.prompts", "--write"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert generated.returncode == 0, generated.stderr
    result = subprocess.run(
        [sys.executable, "-m", "webui.prompts", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_unknown_schema_and_source_drift_fail_closed(tmp_path: Path) -> None:
    root = _copy_prompt_sources(tmp_path)
    identities_path = root / "prompts" / "identities.json"
    identities = json.loads(identities_path.read_text())
    identities["schema"] = "avaas/identities@v999"
    identities_path.write_text(json.dumps(identities))
    with pytest.raises(prompts.PromptContractError, match="identity schema"):
        prompts.compile_prompts(root)

    shutil.copy(ROOT / "prompts" / "identities.json", identities_path)
    corpus_path = root / "prompts" / "corpora" / "harvard_100.txt"
    corpus_path.write_text(corpus_path.read_text() + "Undeclared drift.\n")
    with pytest.raises(prompts.PromptContractError, match="line count|sha256"):
        prompts.compile_prompts(root)


def test_duplicate_base_ids_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _copy_prompt_sources(tmp_path)
    catalog_path = root / "prompts" / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["sections"][2]["prompts"].append(catalog["sections"][2]["prompts"][0])
    catalog_path.write_text(json.dumps(catalog))

    # Exercise declaration validation after simulating an authenticated new
    # catalog snapshot; unauthenticated changes are covered separately below.
    changed_hash = prompts.sha256_file(catalog_path)
    provenance_path = root / "prompts" / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    next(item for item in provenance["sources"] if item["id"] == "avaas_local")[
        "sha256"
    ] = changed_hash
    provenance_path.write_text(json.dumps(provenance))
    monkeypatch.setitem(
        prompts.PINNED_SOURCE_HASHES["2026.07.16"], "avaas_local", changed_hash
    )
    with pytest.raises(prompts.PromptContractError, match="duplicate"):
        prompts.compile_prompts(root)


def test_source_path_escape_and_missing_section_fail_closed(tmp_path: Path) -> None:
    root = _copy_prompt_sources(tmp_path)
    provenance_path = root / "prompts" / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["sources"][0]["path"] = "../outside.txt"
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(prompts.PromptContractError, match="escapes root"):
        prompts.compile_prompts(root)

    shutil.copy(ROOT / "prompts" / "provenance.json", provenance_path)
    catalog_path = root / "prompts" / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["sections"] = catalog["sections"][:-1]
    catalog_path.write_text(json.dumps(catalog))
    with pytest.raises(prompts.PromptContractError, match="mandatory section|sha256"):
        prompts.compile_prompts(root)


def test_source_and_editable_hash_cannot_drift_without_version_bump(tmp_path: Path) -> None:
    root = _copy_prompt_sources(tmp_path)
    corpus_path = root / "prompts" / "corpora" / "harvard_100.txt"
    lines = corpus_path.read_text().splitlines()
    lines[0] = "A changed sentence with the same line count."
    corpus_path.write_text("\n".join(lines) + "\n")

    provenance_path = root / "prompts" / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["sources"][0]["sha256"] = prompts.sha256_file(corpus_path)
    provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(prompts.PromptContractError, match="without a corpus version bump"):
        prompts.compile_prompts(root)


def test_catalog_hash_is_authenticated_before_declarations_are_used(tmp_path: Path) -> None:
    root = _copy_prompt_sources(tmp_path)
    catalog_path = root / "prompts" / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["sections"][2]["prompts"].append(catalog["sections"][2]["prompts"][0])
    catalog_path.write_text(json.dumps(catalog))
    with pytest.raises(prompts.PromptContractError, match="sha256|without a corpus version bump"):
        prompts.compile_prompts(root)


@pytest.mark.parametrize(
    ("filename", "mutation", "message"),
    [
        ("identities.json", lambda value: value.update({"unexpected": True}), "identity keys"),
        (
            "identities.json",
            lambda value: value["pronunciation_lexicon"][0].update(
                {"validation_prompts": ["sec99_999__satraj"]}
            ),
            "validation prompt",
        ),
        (
            "provenance.json",
            lambda value: value["sources"][0].update({"source_sha256": "not-a-digest"}),
            "source_sha256",
        ),
        ("provenance.json", lambda value: value["sources"][0].update({"extra": 1}), "source keys"),
    ],
)
def test_identity_and_provenance_schema_mutations_fail_closed(
    tmp_path: Path, filename: str, mutation, message: str
) -> None:
    root = _copy_prompt_sources(tmp_path)
    path = root / "prompts" / filename
    value = json.loads(path.read_text())
    mutation(value)
    path.write_text(json.dumps(value))
    with pytest.raises(prompts.PromptContractError, match=message):
        prompts.compile_prompts(root)


def test_harvard_build_asset_parser_is_bounded_and_deterministic(tmp_path: Path) -> None:
    from scripts.materialize_prompt_corpora import MaterializationError, extract_harvard_100

    html = "<html>" + "".join(
        f"<h2>List {list_number}</h2><ol>"
        + "".join(
            f"<li>Synthetic list {list_number} sentence {sentence_number}.</li>"
            for sentence_number in range(1, 11)
        )
        + "</ol>"
        for list_number in range(1, 11)
    ) + "</html>"
    lines = extract_harvard_100(html.encode())
    assert len(lines) == 100
    assert lines[0] == "Synthetic list 1 sentence 1."
    assert lines[-1] == "Synthetic list 10 sentence 10."
    with pytest.raises(MaterializationError, match="size"):
        extract_harvard_100(b"x" * (256 * 1024 + 1))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["sections"][2].update({"unexpected": True}), "section keys"),
        (lambda value: value["sections"][0].update({"unexpected": True}), "section keys"),
        (
            lambda value: value["sections"][2]["prompts"][0].update({"unexpected": True}),
            "declaration keys",
        ),
        (
            lambda value: value["sections"][2]["prompts"][0].update(
                {"text": "conflicts with template"}
            ),
            "declaration keys",
        ),
        (
            lambda value: value["sections"][12]["prompts"][0]["variants"].update(
                {"satraj": 7}
            ),
            "variant text",
        ),
        (
            lambda value: value["sections"][3]["prompts"][0].update({"kind": "typo"}),
            "kind",
        ),
    ],
)
def test_catalog_section_and_declaration_shapes_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    root = _copy_prompt_sources(tmp_path)
    catalog_path = root / "prompts" / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    mutation(catalog)
    catalog_path.write_text(json.dumps(catalog))
    changed_hash = prompts.sha256_file(catalog_path)
    provenance_path = root / "prompts" / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    next(item for item in provenance["sources"] if item["id"] == "avaas_local")[
        "sha256"
    ] = changed_hash
    provenance_path.write_text(json.dumps(provenance))
    monkeypatch.setitem(
        prompts.PINNED_SOURCE_HASHES["2026.07.16"], "avaas_local", changed_hash
    )
    with pytest.raises(prompts.PromptContractError, match=message):
        prompts.compile_prompts(root)
