"""High-precision analytical math/physics/materials tool via the
Wolfram|Alpha LLM API — objective, computed answers rather than predicted
text, for exactly the kind of query an LLM's own arithmetic is unreliable
for.

Endpoint choice, correcting the original spec's `http://wolframalpha.com`:
Wolfram serves this over HTTPS only (sending an API key over plaintext
HTTP would be poor practice regardless), and the real host is
`www.wolframalpha.com`. Of Wolfram's several API products, this uses the
**LLM API** (`/api/v1/llm-api`) specifically — it's the product Wolfram
built for exactly this tool-calling use case, and returns plain text
ready to hand back to the model. The alternative "Full Results API"
(`api.wolframalpha.com/v2/query`) returns structured XML/JSON "pods" that
would need real parsing logic to turn into clean text, and isn't worth
that complexity here without a live key to test the actual response
shape against.

Fallback, on any failure (missing WOLFRAM_APP_ID, network error, non-200/
empty response): a restricted, `ast`-based arithmetic evaluator — real
Python `eval()` is never used on LLM-supplied text. This handles a
genuine, meaningful subset of "verifying mathematical realities" (literal
numeric/algebraic expressions with a small whitelist of math functions),
but it is explicitly **not** a natural-language physics/materials solver;
an NL query Wolfram can't be reached for gets an honest "cannot verify
this without Wolfram" response rather than a fabricated plausible-sounding
number — this tool exists specifically so Zane doesn't guess at exact
figures, so the fallback path must not reintroduce that failure mode.
"""
from __future__ import annotations

import ast
import logging
import math
import operator
from typing import Any, Callable, Dict, Optional

import httpx

logger = logging.getLogger("zane.tools.wolfram_tool")

WOLFRAM_LLM_API_URL = "https://www.wolframalpha.com/api/v1/llm-api"

SYSTEM_LOG_FALLBACK_NOTICE = (
    "[SYSTEM LOG]: Wolfram engine unavailable. Commencing local fallback calculation."
)

WOLFRAM_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "query_wolfram_alpha",
        "description": (
            "Use this tool for exact structural calculations, material stresses, "
            "torque requirements, or verifying mathematical realities. Queries "
            "Wolfram|Alpha for an objective, computed answer rather than relying "
            "on predicted text. If Wolfram is unavailable, falls back to a local "
            "arithmetic evaluator that handles literal numeric expressions only — "
            "not natural-language physics questions — in that degraded mode."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A natural-language math/physics/engineering query (e.g. "
                        "'torque required to shear a 10mm steel bolt') or a literal "
                        "arithmetic expression (e.g. '(200*9.81)/0.0005')."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


class WolframQueryError(RuntimeError):
    """Raised internally by WolframAlphaClient; always caught by
    query_wolfram_alpha, which falls back rather than letting this
    propagate into the tool-calling loop."""


class WolframAlphaClient:
    def __init__(self, app_id: Optional[str], timeout_s: float = 10.0) -> None:
        self.app_id = app_id
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def query(self, query_text: str) -> str:
        if not self.app_id:
            raise WolframQueryError("WOLFRAM_APP_ID is not configured.")

        try:
            response = await self._client.get(
                WOLFRAM_LLM_API_URL, params={"input": query_text, "appid": self.app_id}
            )
        except httpx.HTTPError as exc:
            raise WolframQueryError(f"Wolfram|Alpha request failed: {exc}") from exc

        if response.status_code != 200:
            raise WolframQueryError(
                f"Wolfram|Alpha returned HTTP {response.status_code}: {response.text[:300]}"
            )

        text = response.text.strip()
        if not text:
            raise WolframQueryError("Wolfram|Alpha returned an empty response.")
        return text

    async def close(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------
# Local fallback: a restricted arithmetic evaluator (ast-based, NOT eval()).
# --------------------------------------------------------------------------

_SAFE_CONSTANTS: Dict[str, float] = {"pi": math.pi, "e": math.e}
_SAFE_FUNCTIONS: Dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "pow": pow,
    "round": round,
}
_SAFE_BINARY_OPERATORS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_SAFE_UNARY_OPERATORS: Dict[type, Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINARY_OPERATORS:
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return _SAFE_BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPERATORS:
        return _SAFE_UNARY_OPERATORS[type(node.op)](_safe_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCTIONS:
        args = [_safe_eval_node(arg) for arg in node.args]
        if node.keywords:
            raise ValueError("Keyword arguments are not supported in local fallback expressions.")
        return _SAFE_FUNCTIONS[node.func.id](*args)
    if isinstance(node, ast.Name) and node.id in _SAFE_CONSTANTS:
        return _SAFE_CONSTANTS[node.id]
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def safe_local_eval(expression: str) -> Optional[float]:
    """Evaluates a restricted arithmetic expression — numeric literals,
    + - * / ** % //, unary +/-, parentheses, and a small whitelist of
    `math` functions/constants — without ever calling Python's real
    `eval()`/`exec()` on the (LLM-supplied, effectively untrusted) input.
    Returns None if `expression` isn't parseable as such an expression at
    all, rather than raising — this is a safety-net fallback, not a
    general math/physics solver, and callers are expected to treat None
    as "cannot resolve locally," not as an error."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    try:
        return _safe_eval_node(tree.body)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


async def query_wolfram_alpha(query: str, client: Optional[WolframAlphaClient]) -> str:
    """LLM tool entry point. Tries the real Wolfram|Alpha LLM API first
    (if `client` has an app_id configured); on any failure, logs and
    returns the mandated system notice, then attempts the local safe
    arithmetic fallback. Never raises — a failed calculation is always a
    clear text result, never an exception into the tool-calling loop."""
    if client is not None:
        try:
            result = await client.query(query)
            logger.info("[WOLFRAM] Query resolved via the Wolfram|Alpha LLM API.")
            return result
        except WolframQueryError as exc:
            logger.warning("Wolfram|Alpha query failed, falling back locally: %s", exc)
    else:
        logger.debug("No Wolfram|Alpha client configured; using local fallback directly.")

    logger.info(SYSTEM_LOG_FALLBACK_NOTICE)

    local_result = safe_local_eval(query)
    if local_result is not None:
        return f"{SYSTEM_LOG_FALLBACK_NOTICE} Local analytical core computed: {query} = {local_result}"

    return (
        f"{SYSTEM_LOG_FALLBACK_NOTICE} {query!r} is not a literal arithmetic expression "
        f"my local fallback can resolve without Wolfram|Alpha; I am unable to verify an "
        f"exact numeric answer for this query right now."
    )
