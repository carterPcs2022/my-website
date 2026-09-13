import pytest

from zane.interfaces.api import web_console


@pytest.mark.asyncio
async def test_web_console_exposes_dual_companion_ui():
    response = await web_console()
    assert response.status_code == 200
    assert "Zane &amp; P.I.X.A.L." in response.body.decode("utf-8")
    assert "data.pixal_reply" in response.body.decode("utf-8")
    assert "Zane" in response.body.decode("utf-8")
    assert "P.I.X.A.L." in response.body.decode("utf-8")
