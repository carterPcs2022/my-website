from zane.personality import HumorSwitch, PersonaContext, build_system_prompt


def test_base_prompt_contains_core_identity():
    prompt = build_system_prompt(humor_enabled=False)
    assert "Zane Julien" in prompt
    assert "Nindroid" in prompt
    assert "HUMOR SWITCH" not in prompt


def test_humor_addendum_only_present_when_enabled():
    prompt_off = build_system_prompt(humor_enabled=False)
    prompt_on = build_system_prompt(humor_enabled=True)
    assert "HUMOR SWITCH: ACTIVE" not in prompt_off
    assert "HUMOR SWITCH: ACTIVE" in prompt_on


def test_tools_guidance_toggle():
    with_tools = build_system_prompt(tools_enabled=True)
    without_tools = build_system_prompt(tools_enabled=False)
    assert "web_search" in with_tools
    assert "TOOLS AVAILABLE" not in without_tools


def test_context_injection():
    ctx = PersonaContext(addressed_by="Kai", mission_context="Infiltrating the Fire Temple")
    prompt = build_system_prompt(context=ctx)
    assert "Kai" in prompt
    assert "Fire Temple" in prompt


def test_relevant_memories_rendered_as_distinct_block():
    ctx = PersonaContext(
        relevant_memories=["[user @ 2026-01-01 00:00 UTC] I prefer ice-related metaphors."]
    )
    prompt = build_system_prompt(context=ctx)
    assert "RELEVANT PAST CONTEXT" in prompt
    assert "ice-related metaphors" in prompt


def test_conversation_summary_rendered_separately_from_relevant_memories():
    ctx = PersonaContext(
        conversation_summary="The user and Zane discussed a prior mission to the Fire Temple.",
        relevant_memories=["[user @ 2026-01-01 00:00 UTC] The vault code is 4471."],
    )
    prompt = build_system_prompt(context=ctx)
    assert "EARLIER CONVERSATION SUMMARY" in prompt
    assert "Fire Temple" in prompt
    assert "RELEVANT PAST CONTEXT" in prompt
    assert "vault code" in prompt
    # The summary block must appear before the retrieved-memory block.
    assert prompt.index("EARLIER CONVERSATION SUMMARY") < prompt.index("RELEVANT PAST CONTEXT")


def test_no_memory_blocks_when_context_has_none():
    ctx = PersonaContext(addressed_by="Nya")
    prompt = build_system_prompt(context=ctx)
    assert "EARLIER CONVERSATION SUMMARY" not in prompt
    assert "RELEVANT PAST CONTEXT" not in prompt


def test_humor_switch_toggle():
    switch = HumorSwitch(enabled=False)
    assert switch.enabled is False
    assert switch.toggle() is True
    assert switch.enabled is True
    switch.off()
    assert switch.enabled is False
    switch.on()
    assert switch.enabled is True
