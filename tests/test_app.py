from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_renders_every_program():
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    assert not at.exception, at.exception
    sel = [s for s in at.selectbox if s.label == "Program"][0]
    for pid in sel.options:
        sel.set_value(pid.split(" · ")[0] if " · " in pid else pid).run()
        assert not at.exception, at.exception


def test_margin_slider_moves_the_headline():
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    at.sidebar.slider[0].set_value(0.5).run()
    assert not at.exception, at.exception
