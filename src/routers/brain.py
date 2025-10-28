"""
Brain Router
/brain endpoints for status, summary, and reset
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from datetime import datetime

from ..core.trae_orchestrator import get_trae_orchestrator
from ..state.memory import compute_metrics, write_daily_summary
from ..security import verify_api_key

router = APIRouter()


@router.get('/brain/status')
async def brain_status(_: bool = Depends(verify_api_key)) -> Dict[str, Any]:
    orchestrator = await get_trae_orchestrator()
    status = orchestrator.get_status()
    # Enrich with computed metrics
    status['computed'] = compute_metrics()
    return status


@router.get('/brain/summary')
async def brain_summary(_: bool = Depends(verify_api_key)) -> Dict[str, Any]:
    orchestrator = await get_trae_orchestrator()
    summary = orchestrator.get_summary()
    summary['metrics'] = compute_metrics()
    # Persist to reports/summary.json
    write_daily_summary(summary)
    return summary


@router.post('/brain/reset')
async def brain_reset(_: bool = Depends(verify_api_key)) -> Dict[str, Any]:
    try:
        orchestrator = await get_trae_orchestrator()
        await orchestrator.reset()
        return {
            'message': 'Brain reset complete',
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")