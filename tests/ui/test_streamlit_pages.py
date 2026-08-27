from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINTS = [ROOT / "predictions.py", *sorted((ROOT / "pages").glob("*.py"))]


@pytest.mark.ui
@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.stem)
def test_streamlit_page_renders_without_exception(entrypoint: Path) -> None:
    timeout = 90 if entrypoint.stem == "2_Sports_Picks" else 30
    app = AppTest.from_file(str(entrypoint), default_timeout=timeout).run()
    assert not app.exception, [exception.message for exception in app.exception]
