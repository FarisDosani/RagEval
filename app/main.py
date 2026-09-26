from fastapi import FastAPI

app = FastAPI(
    title="RAGEval",
    description="Evaluation framework for Retrieval-Augmented Generation systems",
    version="0.1.0",
)


@app.get("/")
def root():
    return {"name": "RAGEval", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy"}
