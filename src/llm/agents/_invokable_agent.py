import abc

from src.llm.agents._utils import clean_triple_trailing_backticks


class InvokableAgent(abc.ABC):
    """Abstract base class for LLM agents that can be invoked with a single query and return a single answer."""

    def invoke(self, query: str) -> str:
        raw_response = self._invoke(query)
        cleaned_response = self._clean(raw_response)
        return cleaned_response

    def _clean(self, raw_response: str) -> str:
        return clean_triple_trailing_backticks(raw_response)

    @abc.abstractmethod
    def _invoke(self, query: str) -> str:
        """This is implemented by the concrete subclasses."""
        pass
