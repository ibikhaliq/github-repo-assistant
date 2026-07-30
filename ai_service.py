import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from models import (
    AIRepositoryAnswer,
    FileSummary,
    RepositorySummary,
)
from prompts import (
    build_file_summary_prompt,
    build_repository_question_prompt,
    build_repository_summary_prompt,
)


load_dotenv()

SPIDERAI_API_KEY = os.getenv("SPIDERAI_API_KEY")
SPIDERAI_BASE_URL = os.getenv(
    "SPIDERAI_BASE_URL",
    "https://spideraiapi.richmond.edu/v1",
)
SPIDERAI_MODEL = os.getenv("SPIDERAI_MODEL", "gpt-4.1")
SPIDERAI_EMBEDDING_MODEL = os.getenv(
    "SPIDERAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)


if not SPIDERAI_API_KEY:
    raise RuntimeError(
        "SPIDERAI_API_KEY was not found. "
        "Add it to the backend/.env file."
    )


client = OpenAI(
    api_key=SPIDERAI_API_KEY,
    base_url=SPIDERAI_BASE_URL,
)

def get_response_text(response) -> str:
    text_parts = []

    for output_item in response.output or []:
        if output_item.type != "message":
            continue

        for content_item in output_item.content or []:
            if content_item.type != "output_text":
                continue

            if content_item.text:
                text_parts.append(
                    content_item.text
                )

    response_text = "".join(
        text_parts
    ).strip()

    if not response_text:
        raise ValueError(
            "The AI returned no usable text."
        )

    return response_text

def summarize_file_with_ai(
    file_path: str,
    file_content: str,
) -> FileSummary:
    prompt = build_file_summary_prompt(
        file_path=file_path,
        file_content=file_content,
    )

    response = client.responses.create(
        model=SPIDERAI_MODEL,
        input=prompt,
    )

    try:
        response_text = get_response_text(response)
        summary_data = json.loads(response_text)
        validated_summary = FileSummary.model_validate(summary_data)
    except json.JSONDecodeError as error:
        raise ValueError(
            "The AI returned text that was not valid JSON."
        ) from error
    except ValidationError as error:
        raise ValueError(
            "The AI response did not match the expected summary structure."
        ) from error

    return validated_summary

def summarize_repository_with_ai(
    repository_name: str,
    file_summaries: list[dict],
) -> RepositorySummary:
    prompt = build_repository_summary_prompt(
        repository_name=repository_name,
        file_summaries=file_summaries,
    )

    response = client.responses.create(
        model=SPIDERAI_MODEL,
        input=prompt,
    )

    try:
        response_text = get_response_text(response)
        summary_data = json.loads(response_text)

        validated_summary = RepositorySummary.model_validate(
            summary_data
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "The AI returned an invalid repository summary."
        ) from error
    except ValidationError as error:
        raise ValueError(
            "The repository summary did not match the expected structure."
        ) from error

    return validated_summary

def answer_repository_question_with_ai(
    repository_summary: dict,
    relevant_file_summaries: list[dict],
    question: str,
    retrieval_mode: str,
) -> AIRepositoryAnswer:
    prompt = build_repository_question_prompt(
    repository_summary=repository_summary,
    relevant_file_summaries=relevant_file_summaries,
    question=question,
    retrieval_mode=retrieval_mode,
    )

    response = client.responses.create(
        model=SPIDERAI_MODEL,
        input=prompt,
    )

    try:
        response_text = get_response_text(response)
        parsed_response = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "The AI response was not valid JSON."
        ) from error

    try:
        return AIRepositoryAnswer.model_validate(
            parsed_response
        )
    except Exception as error:
        raise ValueError(
            "The AI response did not match the expected answer structure."
        ) from error