"""OpenAI LLM provider implementation."""

import os
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider using LangChain."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: OpenAI model name/version
            base_url: OpenAI base URL (if None, uses default OpenAI URL)
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            **kwargs: Additional LangChain ChatOpenAI parameters
        """
        self._model = model
        self._base_url = base_url
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self._api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")

        # Initialize LangChain OpenAI client
        self.client = ChatOpenAI(
            model=model,
            base_url=self._base_url,
            api_key=self._api_key,
            **kwargs
        )

    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke OpenAI LLM with a text prompt."""
        try:
            response = self.client.invoke(prompt, **kwargs)
            return response.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")

    def invoke_structured(self, prompt: str, schema: type, **kwargs) -> Any:
        """Invoke OpenAI LLM with structured output."""
        try:
            # Create output parser
            parser = PydanticOutputParser(pydantic_object=schema)

            # Create prompt template with format instructions
            template = PromptTemplate(
                template="{prompt}\n\n{format_instructions}",
                input_variables=["prompt"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )

            # Create chain
            chain = template | self.client | parser

            # Invoke and parse
            result = chain.invoke({"prompt": prompt}, **kwargs)
            return result

        except Exception as e:
            raise RuntimeError(f"OpenAI structured output error: {str(e)}")

    @property
    def name(self) -> str:
        """Name of the LLM provider."""
        return "OpenAI"

    @property
    def model(self) -> str:
        """Model name/version being used."""
        return self._model