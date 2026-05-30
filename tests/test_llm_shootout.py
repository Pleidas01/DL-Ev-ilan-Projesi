import base64
from io import BytesIO

from PIL import Image

from llm.clients import CANDIDATES, _image_data_url, _openai_chat_temperature, candidate_by_id, estimate_cost_usd
from llm.shootout import (
    FEW_SHOT_SLOT_EXAMPLES,
    build_slot_prompt,
    choose_winners,
    run_text_slot_benchmark,
    score_expected_slots,
)


def test_build_slot_prompt_includes_few_shot_examples():
    prompt = build_slot_prompt("test sorgusu")

    assert len(FEW_SHOT_SLOT_EXAMPLES) >= 2
    for index, example in enumerate(FEW_SHOT_SLOT_EXAMPLES, start=1):
        assert f"Ornek {index}:" in prompt
        assert example["query"] in prompt
    assert "test sorgusu" in prompt
    assert '"hard_filters"' in prompt
    assert '"filters"' in prompt
    assert '"price_currency"' in prompt


def test_build_slot_prompt_can_disable_few_shot():
    prompt = build_slot_prompt("yalnizca sema", include_few_shot=False)

    assert "Ornek 1:" not in prompt
    assert "yalnizca sema" in prompt


def test_feasibility_first_candidate_set_excludes_prohibitive_models():
    candidate_ids = {candidate.id for candidate in CANDIDATES}

    assert candidate_ids == {
        "deepseek_v4_flash",
        "deepseek_v4_pro",
        "kimi_k2_6",
        "gemini_3_5_flash",
        "glm_4_6",
        "gemma_4_local",
        "qwen3_vl_local",
    }
    assert not any("gpt" in candidate.id or "claude" in candidate.id for candidate in CANDIDATES)


def test_100k_listing_projection_stays_under_feasibility_threshold():
    projected_costs = {
        candidate.id: estimate_cost_usd(candidate, input_tokens=700 * 100_000, output_tokens=200 * 100_000)
        for candidate in CANDIDATES
    }

    assert projected_costs["deepseek_v4_flash"] < 50
    assert projected_costs["kimi_k2_6"] < 500
    assert projected_costs["gemini_3_5_flash"] < 500
    assert projected_costs["gemma_4_local"] == 0


def test_choose_winners_requires_feasible_quality_and_modality_fit():
    rows = [
        {"model_id": "deepseek_v4_flash", "supports_vision": False, "quality_score": 0.86, "cost_100k_usd": 13},
        {"model_id": "glm_4_6", "supports_vision": False, "quality_score": 0.84, "cost_100k_usd": 35},
        {"model_id": "kimi_k2_6", "supports_vision": True, "quality_score": 0.85, "cost_100k_usd": 220},
        {"model_id": "gemini_3_5_flash", "supports_vision": True, "quality_score": 0.88, "cost_100k_usd": 430},
        {"model_id": "gemma_4_local", "supports_vision": True, "quality_score": 0.72, "cost_100k_usd": 0},
    ]

    winners = choose_winners(rows)

    assert winners["text_model"] == "deepseek_v4_flash"
    assert winners["vision_model"] == "gemini_3_5_flash"


def test_openai_chat_temperature_for_moonshot_is_one():
    assert _openai_chat_temperature(candidate_by_id("kimi_k2_6")) == 1.0
    assert _openai_chat_temperature(candidate_by_id("deepseek_v4_flash")) == 0.0


def test_image_data_url_downsizes_large_images(tmp_path):
    from llm.clients import VISION_MAX_IMAGE_EDGE

    image_path = tmp_path / "large.png"
    Image.new("RGB", (2400, 1600), "white").save(image_path)

    data_url = _image_data_url(str(image_path))

    prefix = "data:image/jpeg;base64,"
    assert data_url.startswith(prefix)
    decoded = base64.b64decode(data_url[len(prefix):])
    resized = Image.open(BytesIO(decoded))
    assert max(resized.size) == VISION_MAX_IMAGE_EDGE


def test_score_expected_slots_accepts_scalar_actual_for_list_fields():
    parsed = {
        "hard_filters": {"filters": {"room_count": "1+1", "district": "Kadikoy"}, "max_price_amount": 30000},
        "free_form_tr": "test",
    }
    expected = {"room_count": ["1+1"], "district": ["Kadikoy"], "max_price_amount": 30000}

    score = score_expected_slots(parsed, expected)

    assert score == 1.0


def test_score_expected_slots_does_not_crash_on_int_list_mismatch():
    parsed = {
        "hard_filters": {"filters": {"room_count": 1}},
        "free_form_tr": "test",
    }

    score = score_expected_slots(parsed, {"room_count": ["1+1"]})

    assert 0.0 <= score <= 1.0


def test_run_text_slot_benchmark_records_provider_errors(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")

    def fail_call(*_args, **_kwargs):
        raise RuntimeError("ollama unavailable")

    monkeypatch.setattr("llm.shootout.complete_json", fail_call)

    rows = run_text_slot_benchmark(["gemma_4_local"])

    assert rows[0]["model_id"] == "gemma_4_local"
    assert rows[0]["quality_score"] == 0.0
    assert rows[0]["status"].startswith("error:")
