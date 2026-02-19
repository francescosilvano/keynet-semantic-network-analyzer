import importlib


def test_import_keynet():
    keynet = importlib.import_module("keynet")
    assert getattr(keynet, "__version__", None)


def test_keywords_include_required():
    from keynet import config

    expected = [
        "green transition", "greenhouse effect", "loss of biodiversity", "extreme weather events",
        "CO2", "emissions", "global warming", "melting glaciers", "renewable energy", "misinformation",
        "ecosystem", "fossil fuels", "energy consumption", "normatives", "deforestation",
        "flooding", "tesla", "green policies", "rain", "electric vehicles",
        "natural disaster", "clean energy", "net zero", "AI", "heatwaves"
    ]

    actual_lower = {k.lower() for k in config.KEYWORDS}
    for kw in expected:
        assert kw.lower() in actual_lower, f"Missing keyword: {kw}"
