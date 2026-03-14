from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app


class _FakeTab:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeCacheData:
    def __init__(self) -> None:
        self.clear_calls = 0

    def clear(self) -> None:
        self.clear_calls += 1


class _FakeStreamlit:
    def __init__(self, session_state=None) -> None:
        self.session_state = {} if session_state is None else dict(session_state)
        self.cache_data = _FakeCacheData()

    def set_page_config(self, **kwargs) -> None:
        del kwargs

    def title(self, *_args, **_kwargs) -> None:
        return None

    def caption(self, *_args, **_kwargs) -> None:
        return None

    def tabs(self, labels):
        return [_FakeTab() for _ in labels]


def _patch_main_renderers(monkeypatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(app, "_render_macro_tab", lambda window: calls.append(f"macro:{window}"))
    monkeypatch.setattr(app, "_render_stock_tab", lambda: calls.append("stock"))
    monkeypatch.setattr(app, "_render_signal_history_tab", lambda: calls.append("history"))
    return calls


def test_main_clears_cache_on_first_load(monkeypatch) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)
    calls = _patch_main_renderers(monkeypatch)
    monkeypatch.setattr(app, "_compute_dashboard_data_signature", lambda: (("data.csv", 1, 128),))

    app.main()

    assert fake_st.cache_data.clear_calls == 1
    assert fake_st.session_state[app.APP_DATA_SIGNATURE_SESSION_KEY] == (("data.csv", 1, 128),)
    assert calls == ["macro:15Y", "stock", "history"]


def test_main_clears_cache_only_once_per_session(monkeypatch) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)
    calls = _patch_main_renderers(monkeypatch)
    monkeypatch.setattr(app, "_compute_dashboard_data_signature", lambda: (("data.csv", 1, 128),))

    app.main()
    app.main()

    assert fake_st.cache_data.clear_calls == 1
    assert calls == ["macro:15Y", "stock", "history"] * 2


def test_main_clears_cache_when_data_signature_changes(monkeypatch) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(app, "st", fake_st)
    calls = _patch_main_renderers(monkeypatch)

    signatures = iter(
        [
            (("data.csv", 1, 128),),
            (("data.csv", 1, 128),),
            (("data.csv", 2, 256),),
        ]
    )
    monkeypatch.setattr(app, "_compute_dashboard_data_signature", lambda: next(signatures))

    app.main()
    app.main()
    app.main()

    assert fake_st.cache_data.clear_calls == 2
    assert calls == ["macro:15Y", "stock", "history"] * 3
