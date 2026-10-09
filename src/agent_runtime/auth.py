"""Authenticate calls from the Java backend without a custom HTTP wrapper."""

import hmac
import os

from langgraph_sdk import Auth

auth = Auth()


@auth.authenticate
async def authenticate(headers: dict) -> Auth.types.MinimalUserDict:
    expected = os.getenv("DCP_AGENT_API_KEY", "")
    supplied = headers.get(b"x-agent-key", b"")
    if isinstance(supplied, bytes):
        supplied = supplied.decode("utf-8", errors="replace")
    if len(expected) < 32 or not hmac.compare_digest(
        expected.encode("utf-8"), supplied.encode("utf-8")
    ):
        raise Auth.exceptions.HTTPException(status_code=403, detail="Forbidden")
    return {"identity": "dailycuppaper-backend"}
