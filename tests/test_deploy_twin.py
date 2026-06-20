import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_demo_writes_html_with_crew_and_roster(tmp_path):
    out = tmp_path / "fire.html"
    subprocess.run([sys.executable, "apps/deploy_twin.py", "--out", str(out)], cwd=REPO, check=True)
    html = out.read_text()
    assert "FF-01" in html and "FF-06" in html  # crew plan embedded
    assert "ScatterplotLayer" in html  # crew markers wired into the deck overlay
    assert 'id="roster"' in html  # roster panel present
