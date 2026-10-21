from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from git.exc import GitCommandError
from openai import APIError

from indexing_service import index_repository_service


from ai_service import (
    answer_repository_question_with_ai,
    summarize_file_with_ai,
    
)
from github_service import (
    REPOSITORIES_DIR,
    clone_repository,
    is_valid_github_repo,
)
from models import (
    FileSummaryRequest,
    RepositoryAnswer,
    RepositoryQuestionRequest,
    RepoValidateRequest,
)
from repository_service import (
    MAX_FILE_SIZE_BYTES,
    get_repository_files,
    read_repository_files,
)
from retrieval_service import (
    retrieve_relevant_file_summaries,
)
from semantic_retrieval_service import (
    retrieve_semantic_matches,
)
from summary_service import (
    load_repository_summaries,
    summarize_repository_service,
)
from vector_store import (
    load_repository_embeddings,
    
)

from hybrid_retrieval_service import (
    rank_hybrid_matches,
)


app = FastAPI(
    title="GitHub Repository Assistant"
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)

templates = Jinja2Templates(
    directory="templates",
)

def calculate_evidence_strength(
    retrieval_mode: str,
    validated_relevant_files: list[str],
    repository_summary_available: bool,
) -> tuple[str, str]:
    file_count = len(
        validated_relevant_files
    )

    if (
        retrieval_mode == "standard"
        and file_count >= 3
    ):
        return (
            "strong",
            (
                f"{file_count} directly relevant "
                "repository files supported the answer."
            ),
        )

    if file_count >= 1:
        file_word = (
            "file"
            if file_count == 1
            else "files"
        )

        return (
            "moderate",
            (
                f"{file_count} relevant repository "
                f"{file_word} supported the answer."
            ),
        )

    if repository_summary_available:
        return (
            "limited",
            (
                "The answer relies only on the overall "
                "repository summary."
            ),
        )

    return (
        "limited",
        (
            "No directly supporting repository "
            "evidence was found."
        ),
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get(
    "/",
    response_class=HTMLResponse,
)
def frontend_home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )

@app.post("/repos/validate")
def validate_repo(request: RepoValidateRequest):
    if not is_valid_github_repo(request.repo_url):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid GitHub repository URL. "
                "Expected format: https://github.com/owner/repo"
            ),
        )

    return {
        "repo_url": request.repo_url,
        "valid": True,
    }


@app.post("/repos/clone")
def clone_repo(request: RepoValidateRequest):
    if not is_valid_github_repo(request.repo_url):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid GitHub repository URL. "
                "Expected format: https://github.com/owner/repository"
            ),
        )

    try:
        cloned_repo = clone_repository(request.repo_url)
    except GitCommandError as error:
        raise HTTPException(
            status_code=400,
            detail=(
                "The repository could not be cloned. "
                "Check that the repository exists and is public."
            ),
        ) from error

    return {
        "status": "success",
        "repository": cloned_repo,
    }


@app.get("/repos/{owner}/{repo_name}/files")
def list_repository_files(owner: str, repo_name: str):
    repository_path = REPOSITORIES_DIR / f"{owner}__{repo_name}"

    if not repository_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Repository has not been cloned.",
        )

    repository_files = get_repository_files(repository_path)

    return {
        "repository": f"{owner}/{repo_name}",
        "total_files": len(repository_files),
        "files": repository_files,
    }


@app.get("/repos/{owner}/{repo_name}/contents")
def get_repository_contents(owner: str, repo_name: str):
    repository_path = REPOSITORIES_DIR / f"{owner}__{repo_name}"

    if not repository_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Repository has not been cloned.",
        )

    files = read_repository_files(repository_path)

    return {
        "repository": f"{owner}/{repo_name}",
        "files_read": len(files),
        "files": files,
    }

@app.post("/repos/{owner}/{repo_name}/summarize-file")
def summarize_repository_file(
    owner: str,
    repo_name: str,
    request: FileSummaryRequest,
):
    repository_path = REPOSITORIES_DIR / f"{owner}__{repo_name}"

    if not repository_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Repository has not been cloned.",
        )

    requested_file_path = repository_path / request.file_path

    if not requested_file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="The requested file does not exist.",
        )

    if not requested_file_path.is_file():
        raise HTTPException(
            status_code=400,
            detail="The requested path is not a file.",
        )

    try:
        relative_path = requested_file_path.relative_to(repository_path)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail="The requested file must be inside the repository.",
        ) from error

    file_size = requested_file_path.stat().st_size

    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="The requested file is too large to analyze.",
        )

    try:
        file_content = requested_file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=400,
            detail="The requested file is not valid UTF-8 text.",
        ) from error
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail="The requested file could not be read.",
        ) from error

    try:
        summary = summarize_file_with_ai(
            file_path=str(relative_path),
            file_content=file_content,
        )
    except APIError as error:
        print("OpenAI API error:", repr(error))

        raise HTTPException(
            status_code=502,
            detail=f"The AI request failed: {str(error)}",
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=502,
            detail=str(error),
        ) from error

    return {
        "repository": f"{owner}/{repo_name}",
        "file": str(relative_path),
        "summary": summary.model_dump(),
    }


@app.post("/repos/{owner}/{repo_name}/summarize")
def summarize_repository(
    owner: str,
    repo_name: str,
):
    try:
        return summarize_repository_service(
            owner=owner,
            repo_name=repo_name,
            repositories_dir=REPOSITORIES_DIR,
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except APIError as error:
        raise HTTPException(
            status_code=502,
            detail=(
                "The AI request failed while "
                "summarizing the repository."
            ),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error




@app.post("/repos/{owner}/{repo_name}/index")
def index_repository(
    owner: str,
    repo_name: str,
    force: bool = False,
):
    try:
        return index_repository_service(
            owner=owner,
            repo_name=repo_name,
            repositories_dir=REPOSITORIES_DIR,
            force=force,
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except APIError as error:
        raise HTTPException(
            status_code=502,
            detail=(
                "An AI request failed while "
                "indexing the repository."
            ),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error
    
@app.get("/repos/{owner}/{repo_name}/summary")
def get_saved_repository_summary(
    owner: str,
    repo_name: str,
):
    repository_path = (
        REPOSITORIES_DIR / f"{owner}__{repo_name}"
    )

    if not repository_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Repository not found. Clone it first.",
        )

    try:
        return load_repository_summaries(repository_path)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

@app.post("/repos/{owner}/{repo_name}/ask")
def ask_repository_question(
    owner: str,
    repo_name: str,
    request: RepositoryQuestionRequest,
):
    repository_path = (
        REPOSITORIES_DIR / f"{owner}__{repo_name}"
    )

    if not repository_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Repository not found. Clone it first.",
        )

    # Load the saved repository summaries.
    try:
        repository_data = load_repository_summaries(
            repository_path
        )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=(
                "No saved summary was found. "
                "Summarize the repository first."
            ),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    repository_summary = repository_data.get(
        "repository_summary",
        {},
    )

    file_summaries = repository_data.get(
        "file_summaries",
        [],
    )

    # -----------------------------
    # 1. Keyword retrieval
    # -----------------------------
    keyword_results, keyword_retrieval_mode = (
        retrieve_relevant_file_summaries(
            repository_data=repository_data,
            question=request.question,
            top_k=5,
            min_score=2,
        )
    )


    # -----------------------------
    # 2. Semantic retrieval
    # -----------------------------
    try:
        embedding_data = load_repository_embeddings(
            repository_path
        )

        embedding_records = embedding_data.get(
            "embeddings",
            [],
        )

        semantic_matches = retrieve_semantic_matches(
            question=request.question,
            embedding_records=embedding_records,
            top_k=10,
        )

        MIN_SEMANTIC_SIMILARITY = 0.35

        semantic_matches = [
            match
            for match in semantic_matches
            if match.get(
                "similarity_score",
                0.0,
            ) >= MIN_SEMANTIC_SIMILARITY
        ]

    except FileNotFoundError:
        # Keyword retrieval can still work if embeddings
        # have not been generated yet.
        semantic_matches = []
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error


    # -----------------------------
    # 3. Hybrid ranking
    # -----------------------------
    try:
        hybrid_results = rank_hybrid_matches(
            keyword_results=keyword_results,
            semantic_matches=semantic_matches,
            file_summaries=file_summaries,
            top_k=5,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    combined_file_paths = [
        result["file_path"]
        for result in hybrid_results
    ]

    combined_file_summaries = [
        result["summary"]
        for result in hybrid_results
        if result.get("summary")
    ]

    # Determine how strong the available retrieval
    # evidence is.
    if keyword_retrieval_mode == "standard":
        retrieval_mode = "standard"
    elif hybrid_results:
        retrieval_mode = "fallback"
    else:
        retrieval_mode = "none"

    # -----------------------------
    # 4. Generate the AI answer
    # -----------------------------
    try:
        answer = answer_repository_question_with_ai(
            repository_summary=repository_summary,
            relevant_file_summaries=(
                combined_file_summaries
            ),
            question=request.question,
            retrieval_mode=retrieval_mode,
        )
        # Only allow the model to cite files that were
        # actually retrieved and supplied as evidence.
        allowed_file_paths = set(combined_file_paths)

        validated_relevant_files = [
            file_path
            for file_path in answer.relevant_files
            if file_path in allowed_file_paths
        ]

        evidence_strength, evidence_reason = (
            calculate_evidence_strength(
                retrieval_mode=retrieval_mode,
                validated_relevant_files=(
                    validated_relevant_files
                ),
            repository_summary_available=bool(
                repository_summary
            ),
        )
    )

        answer = RepositoryAnswer(
            answer=answer.answer,
            relevant_files=validated_relevant_files,
            evidence_strength=evidence_strength,
            evidence_reason=evidence_reason,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    # -----------------------------
    # 5. Return response
    # -----------------------------
    return {
        "repository": f"{owner}/{repo_name}",
        "question": request.question,
        "retrieval_mode": retrieval_mode,
        "files_retrieved": len(
            combined_file_paths
        ),
        "keyword_matches": [
            {
                "file_path": item[
                    "summary"
                ].get(
                    "file_path",
                    "",
                ),
                "score": item["score"],
            }
            for item in keyword_results
        ],
        "semantic_matches": [
            {
                "file_path": match.get(
                    "file_path",
                    "",
                ),
                "similarity_score": round(
                    match.get(
                        "similarity_score",
                        0.0,
                    ),
                    4,
                ),
            }
            for match in semantic_matches
        ],

        "hybrid_matches": [
            {
                "file_path": result["file_path"],
                "keyword_score": result["keyword_score"],
                "normalized_keyword_score": round(
                    result["normalized_keyword_score"],
                    4,
                ),
                "semantic_score": round(
                    result["semantic_score"],
                    4,
                ),
                "matched_by_keyword": result[
                    "matched_by_keyword"
                ],
                "matched_by_semantic": result[
                    "matched_by_semantic"
                ],
                "hybrid_score": round(
                    result["hybrid_score"],
                    4,
                ),
            }
            for result in hybrid_results
        ],
        "retrieved_files": (
            combined_file_paths
        ),
        "answer": answer,
    }