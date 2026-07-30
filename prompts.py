def build_file_summary_prompt(file_path: str, file_content: str) -> str:
    return f"""
You are analyzing one file from a software repository.

Use only the provided file content. Do not invent unsupported behavior.

Return only valid JSON. Do not include Markdown, code fences, introductory
text, or text after the JSON.

Use exactly this structure:

{{
  "purpose": "A concise explanation of what the file does.",
  "important_components": [
    "Important function, class, route, variable, or configuration"
  ],
  "dependencies": [
    "Important library, module, framework, or related file"
  ],
  "role_in_project": "How this file contributes to the larger application.",
  "potential_issues": [
    "Clear bug, maintainability concern, or missing error handling"
  ]
}}

If there are no obvious dependencies, return an empty list.

If there are no obvious issues, return an empty list.

File path:
{file_path}

File content:
{file_content}
"""

import json


def build_repository_summary_prompt(
    repository_name: str,
    file_summaries: list[dict],
) -> str:
    summaries_json = json.dumps(
        file_summaries,
        indent=2
    )

    return f"""
You are analyzing a software repository using structured summaries of its
individual files.

Repository:
{repository_name}

Use only the provided file summaries. Do not invent unsupported features.

Return only valid JSON. Do not include Markdown, code fences, introductory
text, or text after the JSON.

Use exactly this structure:

{{
  "purpose": "A concise explanation of what the repository does.",
  "architecture": "A clear explanation of how the project is organized.",
  "main_components": [
    "Important application component"
  ],
  "technologies": [
    "Important language, framework, library, or tool"
  ],
  "entry_points": [
    "Files or functions where execution begins"
  ],
  "data_flow": "How information moves through the application.",
  "strengths": [
    "A technical or architectural strength"
  ],
  "potential_issues": [
    "A bug, risk, missing feature, or maintainability concern"
  ]
}}

If a section has no supported items, return an empty list.

File summaries:
{summaries_json}
"""

def build_repository_question_prompt(
    repository_summary: dict,
    relevant_file_summaries: list[dict],
    question: str,
    retrieval_mode: str
) -> str:
    return f"""
You are analyzing a software repository using previously generated summaries.

Answer the user's question using only the repository information provided below.

Rules:
- Do not invent files, functions, classes, or behavior.
- If the repository summaries do not contain enough information, say so clearly.
- Include only files that are directly relevant to the answer.
- Return valid JSON only.
- Do not include markdown formatting.
- Retrieval mode indicates the strength of the available file evidence.
- If retrieval mode is "none", no relevant file summaries were found.
- When retrieval mode is "none", do not claim that a specific file implements the requested behavior unless the repository summary clearly supports it.
- If the available information is insufficient, explicitly say that the feature could not be found in the repository summaries.
- The relevant_files field may only contain file paths that appear in the provided relevant file summaries.
- Never cite a file solely because it is mentioned in the overall repository summary.
- Do not invent, infer, or add file paths that were not included in the retrieved file context.
- If the retrieved files do not directly support the answer, return an empty relevant_files list.


Return this exact structure:

{{
  "answer": "A clear answer to the question.",
  "relevant_files": ["path/to/file.py"]
  
}}

Repository Summary:

{repository_summary}

Relevant File Summaries:

{relevant_file_summaries}



User question:

{question}

Retrieval mode:

{retrieval_mode}
"""