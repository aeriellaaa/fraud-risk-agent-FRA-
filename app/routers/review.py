from fastapi import APIRouter, HTTPException
from app.storage import store
from app.agents.reviewer_agent import review_score
from app.audit import log_stage

router = APIRouter()


@router.post("/transactions/{transaction_id}/review")
def review_endpoint(transaction_id: str):
    score = store.get_score(transaction_id)
    if score is None:
        raise HTTPException(status_code=404, detail="No score found -- run /score first")

    review = review_score(score)
    store.save_review(review)
    log_stage(transaction_id, "reviewer_agent", "agent_3", {"review": review.model_dump()})

    return review
