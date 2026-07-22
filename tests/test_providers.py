import json

import pytest

from maxgaffer.core import providers


def test_openai_compatible_translates_text_and_base64_images():
    seen = {}

    def post(url, headers, body, timeout):
        seen.update(url=url, headers=headers, body=json.loads(body), timeout=timeout)
        return 200, json.dumps({"choices": [{"message": {"content": "{\"ok\":true}"}}]})

    out = providers.call(
        "openai_compatible", "", "system",
        [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                           "data": "YWJj"}},
            {"type": "text", "text": "inspect"},
        ]}], model="vision-local", base_url="http://127.0.0.1:1234/v1/chat/completions",
        post=post)
    assert out == '{"ok":true}'
    assert seen["body"]["messages"][0] == {"role": "system", "content": "system"}
    content = seen["body"]["messages"][1]["content"]
    assert content[0]["image_url"]["url"] == "data:image/png;base64,YWJj"
    assert "authorization" not in seen["headers"]


def test_anthropic_direct_keeps_native_blocks_and_headers():
    seen = {}

    def post(url, headers, body, timeout):
        seen.update(url=url, headers=headers, body=json.loads(body))
        return 200, json.dumps({"content": [{"type": "text", "text": "OK"}]})

    out = providers.call("anthropic", "sk-test", "s",
                         [{"role": "user", "content": "hello"}], model="model-a",
                         post=post)
    assert out == "OK"
    assert seen["headers"]["x-api-key"] == "sk-test"
    assert seen["body"]["system"] == "s"


def test_offline_and_unknown_providers_fail_fast_with_typed_error():
    with pytest.raises(providers.ProviderError) as err:
        providers.call("offline", "", "s", [], model="none")
    assert err.value.kind == "offline"
    with pytest.raises(providers.ProviderError, match="unknown semantic provider"):
        providers.call("mystery", "", "s", [], model="none")
