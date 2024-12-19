import re


def clean_triple_trailing_backticks(text: str) -> str:
    """
    Removes commonly created backticks that don't enclose code
    at the end of an agents text output.

    Args:
        text: The text to be cleaned. Usually output of agent.invoke().

    Returns:
        The cleaned text (no trailing backticks).
    """
    if text.endswith("```"):
        # Count occurrences of triple backticks
        backtick_groups = re.findall(r"```", text)
        if len(backtick_groups) % 2 == 0:
            # Even count means all are matched,keep
            return text
        else:
            # Odd count means there is an unpaired ending, remove
            return text[:-3]
    return text
