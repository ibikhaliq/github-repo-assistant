const repositoryForm = document.getElementById(
    "repositoryForm"
);

const repositoryUrlInput = document.getElementById(
    "repositoryUrl"
);

const repositoryStatus = document.getElementById(
    "repositoryStatus"
);

const repositoryWorkspace = document.getElementById(
    "repositoryWorkspace"
);

const repositoryName = document.getElementById(
    "repositoryName"
);

const repositorySummary = document.getElementById(
    "repositorySummary"
);

const questionForm = document.getElementById(
    "questionForm"
);

const questionInput = document.getElementById(
    "questionInput"
);

const answerContent = document.getElementById(
    "answerContent"
);

const confidenceBadge = document.getElementById(
    "confidenceBadge"
);

const evidenceStrength = document.getElementById(
    "evidenceStrength"
);

const evidenceReason = document.getElementById(
    "evidenceReason"
);

const relevantFilesSection = document.getElementById(
    "relevantFilesSection"
);

const relevantFilesList = document.getElementById(
    "relevantFilesList"
);

let currentOwner = "";
let currentRepoName = "";

repositoryForm.addEventListener(
    "submit",
    async function (event) {
        event.preventDefault();

        const repositoryUrl =
            repositoryUrlInput.value.trim();

        if (!repositoryUrl) {
            repositoryStatus.textContent =
                "Please enter a GitHub repository URL.";

            return;
        }

        repositoryWorkspace.classList.add(
            "hidden"
        );

        try {
            //----------------------------------
            // Validate
            //----------------------------------

            repositoryStatus.textContent =
                "Validating repository...";

            let response = await fetch(
                "/repos/validate",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json",
                    },
                    body: JSON.stringify({
                        repo_url: repositoryUrl,
                    }),
                }
            );

            let data = await getResponseData(
                response
            );

            //----------------------------------
            // Clone
            //----------------------------------

            repositoryStatus.textContent =
                "Cloning repository...";

            response = await fetch(
                "/repos/clone",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json",
                    },
                    body: JSON.stringify({
                        repo_url: repositoryUrl,
                    }),
                }
            );

            data = await getResponseData(
                response
            );

            //----------------------------------
            // Extract repository details
            //----------------------------------

            const repositoryDetails =
                getRepositoryDetails(
                    repositoryUrl
                );

            currentOwner =
                repositoryDetails.owner;

            currentRepoName =
                repositoryDetails.repoName;

            repositoryName.textContent =
                `${currentOwner}/${currentRepoName}`;

            //----------------------------------
            // Index
            //----------------------------------

            repositoryStatus.textContent =
                "Analyzing repository...";

            response = await fetch(
                `/repos/${currentOwner}/${currentRepoName}/index`,
                {
                    method: "POST",
                }
            );

            data = await getResponseData(
                response
            );

            //----------------------------------
            // Load repository summary
            //----------------------------------

            repositoryStatus.textContent =
                "Loading repository summary...";

            response = await fetch(
                `/repos/${currentOwner}/${currentRepoName}/summary`
            );

            const summaryData =
                await getResponseData(
                    response
                );

            displayRepositorySummary(
                summaryData
            );

            //----------------------------------
            // Finished
            //----------------------------------

            repositoryStatus.textContent =
                "✓ Repository ready.";

            repositoryWorkspace.classList.remove(
                "hidden"
            );
        }

        catch (error) {
            repositoryStatus.textContent =
                error.message ||
                "Something went wrong.";
        }
    }
);


questionForm.addEventListener(
    "submit",
    async function (event) {
        event.preventDefault();

        const question =
            questionInput.value.trim();

        if (!question) {
            answerContent.textContent =
                "Please enter a question.";

            return;
        }

        if (!currentOwner || !currentRepoName) {
            answerContent.textContent =
                "Analyze a repository first.";

            return;
        }

        answerContent.textContent =
            "Searching the repository...";

        answerContent.classList.remove(
            "empty-state"
        );

        evidenceStrength.classList.add(
            "hidden"
        );

        confidenceBadge.classList.remove(
            "high",
            "medium",
            "low"
        );

        confidenceBadge.textContent = "";

        evidenceReason.textContent = "";

        relevantFilesSection.classList.add(
            "hidden"
        );

        relevantFilesList.innerHTML = "";

        try {
            const response = await fetch(
                `/repos/${currentOwner}/${currentRepoName}/ask`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json",
                    },
                    body: JSON.stringify({
                        question: question,
                    }),
                }
            );

            const data = await getResponseData(
                response
            );

            displayAnswer(data);
        }

        catch (error) {
            answerContent.textContent =
                error.message ||
                "The question could not be answered.";
        }
    }
);


async function getResponseData(response) {
    let data;

    try {
        data = await response.json();
    }

    catch {
        throw new Error(
            "The server returned an invalid response."
        );
    }

    if (!response.ok) {
        throw new Error(
            data.detail ||
            "The request failed."
        );
    }

    return data;
}


function getRepositoryDetails(repositoryUrl) {
    const cleanUrl = repositoryUrl.replace(
        /\/+$/,
        ""
    );

    const parts = cleanUrl.split("/");

    return {
        owner: parts[parts.length - 2],
        repoName: parts[parts.length - 1],
    };
}


function displayRepositorySummary(summaryData) {
    const summary =
        summaryData.repository_summary;

    if (!summary) {
        repositorySummary.textContent =
            "No repository summary was returned.";

        return;
    }

    repositorySummary.innerHTML = "";

    addSummarySection(
        "Purpose",
        summary.purpose
    );

    addSummarySection(
        "Architecture",
        summary.architecture
    );

    addSummaryList(
        "Main Components",
        summary.main_components
    );

    addSummaryList(
        "Technologies",
        summary.technologies
    );

    addSummaryList(
        "Entry Points",
        summary.entry_points
    );

    addSummarySection(
        "Data Flow",
        summary.data_flow
    );

    addSummaryList(
        "Strengths",
        summary.strengths
    );

    addSummaryList(
        "Potential Issues",
        summary.potential_issues
    );
}

function addSummarySection(title, content) {
    if (!content) {
        return;
    }

    const section =
        document.createElement("section");

    const heading =
        document.createElement("h3");

    const paragraph =
        document.createElement("p");

    heading.textContent = title;
    paragraph.textContent = content;

    section.appendChild(heading);
    section.appendChild(paragraph);

    repositorySummary.appendChild(section);
}


function addSummaryList(title, items) {
    if (!items || items.length === 0) {
        return;
    }

    const section =
        document.createElement("section");

    const heading =
        document.createElement("h3");

    const list =
        document.createElement("ul");

    heading.textContent = title;

    for (const item of items) {
        const listItem =
            document.createElement("li");

        listItem.textContent = item;

        list.appendChild(listItem);
    }

    section.appendChild(heading);
    section.appendChild(list);

    repositorySummary.appendChild(section);
}


function displayAnswer(data) {
    const answer = data.answer;

    if (!answer) {
        answerContent.textContent =
            "No answer was returned.";

        evidenceStrength.classList.add(
            "hidden"
        );

        relevantFilesSection.classList.add(
            "hidden"
        );

        return;
    }

    answerContent.textContent =
        answer.answer ||
        "No answer text was returned.";

    answerContent.classList.remove(
        "empty-state"
    );

    // -----------------------------
    // Evidence Strength
    // -----------------------------

    const evidenceLevel =
        answer.evidence_strength ||
        "limited";

    const evidenceLabels = {
        strong: "Strong",
        moderate: "Moderate",
        limited: "Limited"
    };

    const evidenceClasses = {
        strong: "high",
        moderate: "medium",
        limited: "low"
    };

    confidenceBadge.textContent =
        evidenceLabels[evidenceLevel] ||
        "Limited";

    confidenceBadge.classList.remove(
        "high",
        "medium",
        "low"
    );

    confidenceBadge.classList.add(
        evidenceClasses[evidenceLevel] ||
        "low"
    );

    evidenceReason.textContent =
        answer.evidence_reason ||
        "No evidence explanation was returned.";

    evidenceStrength.classList.remove(
        "hidden"
    );

    // -----------------------------
    // Relevant Files
    // -----------------------------

    const relevantFiles =
        Array.isArray(answer.relevant_files)
            ? answer.relevant_files
            : [];

    relevantFilesList.innerHTML = "";

    if (relevantFiles.length > 0) {
        for (const filePath of relevantFiles) {
            const listItem =
                document.createElement("li");

            listItem.textContent =
                filePath;

            relevantFilesList.appendChild(
                listItem
            );
        }

        relevantFilesSection.classList.remove(
            "hidden"
        );
    }

    else {
        relevantFilesSection.classList.add(
            "hidden"
        );
    }
}