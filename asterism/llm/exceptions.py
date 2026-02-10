"""LLM provider exceptions."""


class AllProvidersFailedError(Exception):
    """Raised when all LLM providers in the fallback chain fail.

    Attributes:
        message: Error message describing the failure
        last_error: The last exception that occurred
        provider_chain: List of provider names that were tried
    """

    def __init__(
        self,
        message: str,
        last_error: Exception | None = None,
        provider_chain: list[str] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.last_error = last_error
        self.provider_chain = provider_chain or []

    def __str__(self) -> str:
        if self.last_error:
            return f"{self.message} Last error: {self.last_error}"
        return self.message
