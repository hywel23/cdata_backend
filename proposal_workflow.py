"""CDATA Phase 8: Firebase-authenticated, Firestore-backed proposal workflow.

Register after the existing /api/simulate endpoint:
    from proposal_workflow import register_proposal_routes
    register_proposal_routes(app)

Local research prototype: records decisions but NEVER executes a retail price.
"""

import hashlib
import json
from datetime import datetime, timezone

import os
import firebase_admin
from firebase_admin import auth, firestore, credentials
from google.cloud import firestore as google_firestore
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
import httpx

def _firebase_app():
    try:
        return firebase_admin.get_app()

    except ValueError:
        credential_path = os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )

        # LOCAL COMPUTER
        if credential_path and os.path.exists(credential_path):
            return firebase_admin.initialize_app(
                credentials.Certificate(credential_path)
            )

        # CLOUD RUN
        return firebase_admin.initialize_app()

class ProposalInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    proposed_price_rm: float = Field(gt=0, le=100000)
    brent_change_pct: float = Field(default=0, ge=-20, le=20)
    wti_change_pct: float = Field(default=0, ge=-20, le=20)
    unit_cost_change_pct: float = Field(default=0, ge=-20, le=20)
    justification: str = Field(min_length=5, max_length=500)


class DecisionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    decision: str = Field(pattern="^(APPROVED|REJECTED)$")
    note: str = Field(min_length=5, max_length=500)


def _db():
    try:
        if not firebase_admin._apps:
            _firebase_app()
        return firestore.client()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Firebase Admin credentials/Firestore are not configured for this server.",
        ) from exc


def _verified_user(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Firebase ID token required.")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        if not firebase_admin._apps:
            _firebase_app()
        claims = auth.verify_id_token(token, check_revoked=True)
    except (ValueError, auth.InvalidIdTokenError, auth.ExpiredIdTokenError,
            auth.RevokedIdTokenError, auth.UserDisabledError) as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired Firebase ID token.") from exc
    except Exception as exc:
        import traceback

        print(
            "\n========== FIREBASE ERROR ==========",
            flush=True
        )

        traceback.print_exc()

        print(
            "====================================\n",
            flush=True
        )

        raise HTTPException(
            status_code=503,
            detail="Firebase token verification unavailable."
        ) from exc

    role = claims.get("role")
    if role not in ("analyst", "approver"):
        raise HTTPException(status_code=403, detail="Account has no CDATA role.")
    return {"uid": claims["uid"], "email": claims.get("email", ""), "role": role}


def _role(user: dict, expected: str):
    if user["role"] != expected:
        raise HTTPException(status_code=403, detail=f"{expected} role required.")


def _fingerprint(data: dict) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _public_document(snapshot):
    data = snapshot.to_dict() or {}
    for key in ("created_at", "decided_at"):
        if isinstance(data.get(key), datetime):
            data[key] = data[key].isoformat()
    return {"id": snapshot.id, **data}


def register_proposal_routes(app):
    """Attach protected proposal endpoints to the existing CDATA FastAPI app."""
    router = APIRouter(prefix="/api", tags=["Pricing approval workflow"])

    async def calculate_scenario(params: dict):
        # Use the same in-process FastAPI app: one source of truth for calculations.
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://cdata.internal") as client:
            response = await client.post("/api/simulate", json=params)
        if response.status_code != 200:
            raise HTTPException(
                status_code=422,
                detail=f"Cannot validate pricing scenario: {response.status_code} {response.text[:250]}",
            )
        payload = response.json()
        if not isinstance(payload, dict) or not all(
            key in payload for key in ("baseline", "scenario", "comparison", "governance", "data_date")
        ):
            raise HTTPException(status_code=503, detail="Simulation endpoint returned an incompatible schema.")
        return payload

    @router.get("/me")
    def me(user: dict = Depends(_verified_user)):
        return user

    @router.post("/proposals", status_code=201)
    async def submit_proposal(body: ProposalInput, user: dict = Depends(_verified_user)):
        _role(user, "analyst")
        params = body.model_dump(exclude={"justification"})
        simulation = await calculate_scenario(params)
        governance = simulation["governance"]
        status = governance.get("status", "")
        # A blocked scenario may be explored in Scenario Lab, but not submitted.
        if status == "BLOCKED_BY_GOVERNANCE":
            raise HTTPException(status_code=422, detail={
                "message": "Cannot submit a scenario blocked by governance.",
                "violations": governance.get("hard_rule_violations", []),
            })
        if status not in ("HOLD_CURRENT_PRICE", "PENDING_APPROVAL", "ELIGIBLE_FOR_REVIEW"):
            raise HTTPException(status_code=422, detail="Unknown governance status; proposal refused.")

        db = await run_in_threadpool(_db)
        ref = db.collection("pricing_proposals").document()
        record = {
            "created_by_uid": user["uid"],
            "created_by_email": user["email"],
            "status": "PENDING_REVIEW",
            "justification": body.justification,
            "inputs": params,
            "simulation": simulation,  # Server-computed, never taken from Flutter
            "simulation_fingerprint": _fingerprint(simulation),
            "created_at": google_firestore.SERVER_TIMESTAMP,
            "decided_at": None,
            "decision_note": None,
            "decided_by_uid": None,
        }
        event = {
            "event": "SUBMITTED",
            "actor_uid": user["uid"],
            "actor_role": user["role"],
            "created_at": google_firestore.SERVER_TIMESTAMP,
        }

        def commit_submission():
            batch = db.batch()
            batch.create(ref, record)
            batch.create(ref.collection("events").document(), event)
            batch.commit()

        await run_in_threadpool(commit_submission)
        return {
            "id": ref.id,
            "status": "PENDING_REVIEW",
            "governance_status": status,
            "message": "Submitted for human review. No price was executed.",
        }

    @router.get("/proposals")
    def list_proposals(user: dict = Depends(_verified_user)):
        db = _db()
        query = db.collection("pricing_proposals")
        if user["role"] == "analyst":
            query = query.where("created_by_uid", "==", user["uid"])
        results = [_public_document(snap) for snap in query.limit(50).stream()]
        results.sort(key=lambda d: d.get("created_at") or "", reverse=True)
        return {"proposals": results}

    @router.post("/proposals/{proposal_id}/decision")
    async def review_proposal(
        proposal_id: str, body: DecisionInput, user: dict = Depends(_verified_user)
    ):
        _role(user, "approver")
        db = await run_in_threadpool(_db)
        ref = db.collection("pricing_proposals").document(proposal_id)
        snapshot = await run_in_threadpool(ref.get)
        if not snapshot.exists:
            raise HTTPException(status_code=404, detail="Proposal not found.")
        original = snapshot.to_dict()
        if original["status"] != "PENDING_REVIEW":
            raise HTTPException(status_code=409, detail="Proposal has already been decided.")
        if original["created_by_uid"] == user["uid"]:
            raise HTTPException(status_code=403, detail="Users cannot review their own proposals.")

        if body.decision == "APPROVED":
            # Check the same model and governance rules again before approving.
            fresh = await calculate_scenario(original["inputs"])
            if _fingerprint(fresh) != original["simulation_fingerprint"]:
                raise HTTPException(status_code=409, detail="Model/data changed; submit a new scenario.")
            if fresh["governance"].get("status") not in (
                "PENDING_APPROVAL", "ELIGIBLE_FOR_REVIEW"
            ):
                raise HTTPException(
                    status_code=409,
                    detail="This scenario is HOLD/BLOCKED by governance and cannot be approved.",
                )

        @google_firestore.transactional
        def decide_transaction(tx):
            current = ref.get(transaction=tx)
            if not current.exists:
                raise HTTPException(status_code=404, detail="Proposal not found.")
            current_data = current.to_dict()
            if current_data.get("status") != "PENDING_REVIEW":
                raise HTTPException(status_code=409, detail="Proposal already decided.")
            if current_data.get("created_by_uid") == user["uid"]:
                raise HTTPException(status_code=403, detail="Self-review is prohibited.")
            if (body.decision == "APPROVED" and
                    current_data.get("simulation_fingerprint") != original.get("simulation_fingerprint")):
                raise HTTPException(status_code=409, detail="Scenario was modified after validation.")
            tx.update(ref, {
                "status": body.decision,
                "decision_note": body.note,
                "decided_by_uid": user["uid"],
                "decided_at": google_firestore.SERVER_TIMESTAMP,
            })
            tx.create(ref.collection("events").document(), {
                "event": body.decision,
                "actor_uid": user["uid"],
                "actor_role": user["role"],
                "note": body.note,
                "created_at": google_firestore.SERVER_TIMESTAMP,
            })

        await run_in_threadpool(lambda: decide_transaction(db.transaction()))
        return {
            "id": proposal_id,
            "status": body.decision,
            "message": "Decision recorded. No retail price was changed.",
        }

    app.include_router(router)
