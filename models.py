from pydantic import BaseModel


class RepoValidateRequest(BaseModel):
    repo_url: str


class FileSummaryRequest(BaseModel):
    file_path: str


class FileSummary(BaseModel):
    purpose: str
    important_components: list[str]
    dependencies: list[str]
    role_in_project: str
    potential_issues: list[str]


class RepositorySummary(BaseModel):
    purpose: str
    architecture: str
    main_components: list[str]
    technologies: list[str]
    entry_points: list[str]
    data_flow: str
    strengths: list[str]
    potential_issues: list[str]


class RepositoryQuestionRequest(BaseModel):
    question: str


class AIRepositoryAnswer(BaseModel):
    answer: str
    relevant_files: list[str]


class RepositoryAnswer(BaseModel):
    answer: str
    relevant_files: list[str]
    evidence_strength: str
    evidence_reason: str