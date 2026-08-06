"""Tool registry: wires the LLM function-calling schema to real Python/C++
implementations and dispatches Groq tool_call requests to them.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from zane.analytics_bridge import AnalyticsEngine, ProbabilityFactors
from zane.groq_client import AsyncGroqClient
from zane.hardware.peripheral_io import CRYO_DISCHARGE_TOOL_SCHEMA, PeripheralManager
from zane.tools.translate import TRANSLATE_TOOL_SCHEMA, translate_text
from zane.tools.web_search import WEB_SEARCH_TOOL_SCHEMA, WebSearchTool

logger = logging.getLogger("zane.tools.registry")

CALCULATE_PROBABILITY_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calculate_success_probability",
        "description": (
            "Runs Zane's internal analytical core (a C++ probability matrix "
            "engine) to compute an exact, data-driven percentage chance of "
            "success for the current situation, along with a risk index. Use "
            "this instead of inventing a percentage yourself whenever you are "
            "about to quote a specific analytical figure."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "danger_level": {
                    "type": "number",
                    "description": "Assessed threat level, 0-100.",
                },
                "team_synergy": {
                    "type": "number",
                    "description": "How well the team is coordinating, 0-100.",
                },
                "resource_availability": {
                    "type": "number",
                    "description": "Equipment/elemental power on hand, 0-100.",
                },
                "historical_success_rate": {
                    "type": "number",
                    "description": "Success rate in similar past scenarios, 0-100.",
                },
                "complexity_index": {
                    "type": "number",
                    "description": "How many moving parts the plan has, 0-100.",
                },
            },
            "required": [
                "danger_level",
                "team_synergy",
                "resource_availability",
                "historical_success_rate",
                "complexity_index",
            ],
        },
    },
}


class ToolRegistry:
    """Owns tool schemas (for the Groq API request) and tool dispatch (for
    executing a tool_call the model returned)."""

    def __init__(
        self,
        analytics: AnalyticsEngine,
        web_search: WebSearchTool,
        groq_client: AsyncGroqClient,
        peripheral_manager: Optional[PeripheralManager] = None,
    ) -> None:
        self._analytics = analytics
        self._web_search = web_search
        self._groq_client = groq_client
        # None unless ZANE_HARDWARE_ENABLED — see zane/core.py's
        # SharedBackend.build(). A session with no PeripheralManager never
        # advertises engage_cryo_discharge as a tool at all.
        self._peripheral_manager = peripheral_manager

    def schemas(self) -> List[Dict[str, Any]]:
        schemas = [WEB_SEARCH_TOOL_SCHEMA, CALCULATE_PROBABILITY_TOOL_SCHEMA, TRANSLATE_TOOL_SCHEMA]
        if self._peripheral_manager is not None:
            schemas.append(CRYO_DISCHARGE_TOOL_SCHEMA)
        return schemas

    async def dispatch(self, name: str, arguments_json: str) -> str:
        """Executes a single tool call and returns its string result, ready
        to be placed in a `{"role": "tool", ...}` message."""
        try:
            args = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError as exc:
            logger.error("Malformed tool arguments for %s: %s", name, arguments_json)
            return f"TOOL_ERROR: could not parse arguments for {name}: {exc}"

        try:
            if name == "web_search":
                query = args["query"]
                max_results = args.get("max_results")
                return await self._web_search.search_as_tool_result(query, max_results)

            if name == "calculate_success_probability":
                factors = ProbabilityFactors(
                    danger_level=float(args.get("danger_level", 0.0)),
                    team_synergy=float(args.get("team_synergy", 0.0)),
                    resource_availability=float(args.get("resource_availability", 0.0)),
                    historical_success_rate=float(args.get("historical_success_rate", 0.0)),
                    complexity_index=float(args.get("complexity_index", 0.0)),
                )
                result = self._analytics.calculate_success_probability(factors)
                return (
                    f"success_probability={result.success_probability:.2f}%, "
                    f"risk_index={result.risk_index:.2f}%, "
                    f"confidence_interval=[{result.confidence_interval_low:.2f}%, "
                    f"{result.confidence_interval_high:.2f}%], "
                    f"methodology={result.analytical_summary}"
                )

            if name == "translate_text":
                text = args["text"]
                target_language = args["target_language"]
                return await translate_text(text, target_language, self._groq_client)

            if name == "engage_cryo_discharge":
                if self._peripheral_manager is None:
                    return "TOOL_ERROR: engage_cryo_discharge is not available in this session."
                duration_seconds = float(args.get("duration_seconds", 0.0))
                return await self._peripheral_manager.engage_cryo_discharge(duration_seconds)

            return f"TOOL_ERROR: unknown tool '{name}'"
        except Exception as exc:  # noqa: BLE001 - tool failures must not crash the chat loop
            logger.exception("Tool '%s' raised during dispatch", name)
            return f"TOOL_ERROR: {name} failed: {exc}"
