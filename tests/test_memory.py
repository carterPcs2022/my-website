from zane.memory import ConversationMemory


def test_add_and_retrieve_messages():
    memory = ConversationMemory(max_messages=10, max_chars=10000)
    memory.add_user_message("Hello, Zane.")
    memory.add_assistant_message("Greetings. How may I assist you?")

    messages = memory.get_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_trims_by_message_count():
    memory = ConversationMemory(max_messages=4, max_chars=100000)
    for i in range(10):
        memory.add_user_message(f"message {i}")

    messages = memory.get_messages()
    assert len(messages) == 4
    assert messages[-1]["content"] == "message 9"


def test_trims_by_char_budget():
    memory = ConversationMemory(max_messages=1000, max_chars=50)
    memory.add_user_message("a" * 30)
    memory.add_user_message("b" * 30)

    messages = memory.get_messages()
    total_chars = sum(len(m["content"]) for m in messages)
    assert total_chars <= 50
    assert messages[-1]["content"] == "b" * 30


def test_tool_exchange_roundtrip():
    memory = ConversationMemory()
    tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}]
    tool_results = [{"role": "tool", "tool_call_id": "call_1", "name": "web_search", "content": "result"}]
    memory.add_tool_exchange(tool_calls, tool_results)

    messages = memory.get_messages()
    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"] == tool_calls
    assert messages[1]["content"] == "result"


def test_clear():
    memory = ConversationMemory()
    memory.add_user_message("hi")
    assert len(memory) == 1
    memory.clear()
    assert len(memory) == 0
