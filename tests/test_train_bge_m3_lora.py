import pytest

from finetune.text_embed.train_bge_m3_lora import require_cuda, validate_target_modules


def test_require_cuda_fails_before_training_when_cuda_is_unavailable():
    class Cuda:
        @staticmethod
        def is_available():
            return False

    class Torch:
        cuda = Cuda()

    with pytest.raises(RuntimeError, match="CUDA is required"):
        require_cuda(Torch())


def test_validate_target_modules_requires_explicit_matches():
    class Model:
        def named_modules(self):
            return [
                ("encoder.layer.0.attention.self.query", object()),
                ("encoder.layer.0.attention.self.value", object()),
            ]

    assert validate_target_modules(Model(), ["query", "value"]) == {
        "query": ["encoder.layer.0.attention.self.query"],
        "value": ["encoder.layer.0.attention.self.value"],
    }

    with pytest.raises(ValueError, match="key"):
        validate_target_modules(Model(), ["query", "key", "value"])
