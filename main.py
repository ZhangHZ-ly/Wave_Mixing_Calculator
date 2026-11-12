from pathlib import Path
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from calculator import (
    eval_single_mode,
    eval_pf_form,
    eval_wm_solve,
    eval_operator_combination,
    eval_combination_assertions,
    save_latest_pf_variable,
    save_latest_wm_variable,
    TWM_SOLVERS,
    normalize_process,
    process_is_dc,
)


app = FastAPI(title="Quantum Operator Calculator")

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class SingleModeRequest(BaseModel):
    type: str  # 'boson' or 'fermion'
    name: str = ""
    first_op: str  # 'annihilation' or 'creation'
    exponents: str = ""
    action: str  # 'original', 'patform', 'normal_order'


class SymbolDef(BaseModel):
    name: str
    display_name: Optional[str] = None
    is_real: bool = False
    is_integer: bool = False
    is_nonnegative: bool = False
    is_positive: bool = False
    builtin_key: Optional[str] = None
    order: Optional[Union[int, float]] = 0


class OperatorDef(BaseModel):
    type: Optional[str] = 'evolved_op'
    kind: str = "boson"  # 'boson' or 'fermion'
    name: str
    display_name: Optional[str] = None
    mode_index: Optional[int] = 0
    order: Optional[Union[int, float]] = 0


class MMOperatorDef(BaseModel):
    name: str
    mm_kind: str = "COP"  # 'COP' (MultimodeCOP) or 'Num2' (MultimodeNum2)


class NamedExprDef(BaseModel):
    name: str
    expr_str: Optional[str] = ""
    expr_str_t: Optional[str] = None
    expr_str_g: Optional[str] = None
    source_input: Optional[str] = None
    latex: Optional[str] = None
    latex_t: Optional[str] = None
    latex_g: Optional[str] = None
    order: Optional[int] = None
    leading_order: Optional[Union[int, float]] = None
    dc: Optional[bool] = None
    process: Optional[str] = None
    single_mode: Optional[bool] = None


class SaveVarRequest(BaseModel):
    workspace: str
    name: str
    notes: Optional[str] = ""


class PatternFormRequest(BaseModel):
    text: str
    symbols: List[SymbolDef] = []
    operators: List[OperatorDef] = []
    named_expressions: List[NamedExprDef] = []
    use_rem: bool = False


class SolveRequest(BaseModel):
    dc: bool = True
    process: Optional[str] = None
    mismatch: bool = False
    var_index: int
    order: int
    use_gain: bool = False


class CombinationRequest(BaseModel):
    text: str
    dc: bool = True
    process: Optional[str] = None
    mismatch: bool = False
    order: int = 1
    use_gain: bool = False
    single_mode: bool = False
    symbols: List[SymbolDef] = []
    operators: List[OperatorDef] = []
    mm_operators: List[MMOperatorDef] = []
    named_expressions: List[NamedExprDef] = []


class AssertionRequest(CombinationRequest):
    coherent_drive: bool = False
    vacuum_signal: bool = False
    vacuum_other_weak: bool = False


def _format_lead_ord(val: Union[int, float]) -> Union[int, float]:
    f = float(val)
    return int(f) if f.is_integer() else f


@app.get("/")
def index():
    html_path = Path(__file__).parent / "index.html"
    has_local_katex = (Path(__file__).parent / "static" / "katex" / "katex.min.js").exists()
    if has_local_katex:
        return FileResponse(html_path)

    # Auto-fallback: if local static/katex is not installed, seamlessly serve with jsDelivr CDN
    content = html_path.read_text(encoding="utf-8")
    cdn_base = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"
    content = content.replace("/static/katex", cdn_base)
    return HTMLResponse(content)


@app.post("/api/single_mode")
def api_single_mode(req: SingleModeRequest):
    try:
        latex_str, is_herm, expr_str, note = eval_single_mode(
            req.type, req.name, req.first_op, req.exponents, req.action
        )
        return {
            "latex": latex_str,
            "is_hermitian": is_herm,
            "expr_str": expr_str,
            "note": note
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/pf_form")
@app.post("/api/pattern_form")
def api_pattern_form(req: PatternFormRequest):
    try:
        use_rem = req.use_rem
        latex_str, expr_str, rem_latex, rem_str = eval_pf_form(
            req.text,
            [s.model_dump() for s in req.symbols],
            [op.model_dump() for op in req.operators],
            [ne.model_dump() for ne in req.named_expressions],
            use_rem=use_rem
        )
        return {
            "latex": latex_str,
            "expr_str": expr_str,
            "remainder_latex": rem_latex,
            "remainder_str": rem_str,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/solve")
def solve(req: SolveRequest):
    try:
        proc = normalize_process(req.process, req.dc)
        latex_t, latex_g = eval_wm_solve(
            dc=process_is_dc(proc),
            mismatch=req.mismatch,
            var_index=req.var_index,
            order=req.order,
            process=proc,
        )
        return {
            "latex_t": latex_t,
            "latex_g": latex_g
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/combination")
def api_combination(req: CombinationRequest):
    try:
        proc = normalize_process(req.process, req.dc)
        is_dc = process_is_dc(proc)
        latex_curr, latex_t, latex_g, leading_order, sol_len = eval_operator_combination(
            req.text,
            dc=is_dc,
            mismatch=req.mismatch,
            order=req.order,
            use_gain=req.use_gain,
            single_mode=req.single_mode,
            symbols=[s.model_dump() for s in req.symbols],
            operators=[op.model_dump() for op in req.operators],
            mm_operators=[mm.model_dump() for mm in req.mm_operators],
            named_expressions=[ne.model_dump() for ne in req.named_expressions],
            process=proc,
        )
        lead_ord = _format_lead_ord(leading_order)
        return {
            "latex": str(latex_curr),
            "latex_t": str(latex_t),
            "latex_g": str(latex_g),
            "leading_order": lead_ord,
            "order": int(req.order),
            "sol_len": int(sol_len),
            "process": proc,
            "single_mode": bool(req.single_mode),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/combination/assert")
def api_combination_assert(req: AssertionRequest):
    try:
        proc = normalize_process(req.process, req.dc)
        is_dc = process_is_dc(proc)
        latex_curr, latex_t, latex_g, leading_order, sol_len = eval_combination_assertions(
            req.text,
            dc=is_dc,
            mismatch=req.mismatch,
            order=req.order,
            use_gain=req.use_gain,
            single_mode=req.single_mode,
            coherent_drive=req.coherent_drive,
            vacuum_signal=req.vacuum_signal,
            vacuum_other_weak=req.vacuum_other_weak,
            symbols=[s.model_dump() for s in req.symbols],
            operators=[op.model_dump() for op in req.operators],
            mm_operators=[mm.model_dump() for mm in req.mm_operators],
            named_expressions=[ne.model_dump() for ne in req.named_expressions],
            process=proc,
        )
        lead_ord = _format_lead_ord(leading_order)
        return {
            "latex": str(latex_curr),
            "latex_t": str(latex_t),
            "latex_g": str(latex_g),
            "leading_order": lead_ord,
            "order": int(req.order),
            "sol_len": int(sol_len),
            "process": proc,
            "single_mode": bool(req.single_mode),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/variable/save")
def api_save_variable(req: SaveVarRequest):
    try:
        if req.workspace == 'tab3':
            save_latest_wm_variable(req.name, notes=req.notes or "")
        else:
            save_latest_pf_variable(req.name)
        return {"status": "ok", "name": req.name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/combination/reset")
def api_reset_combination():
    try:
        for s in (TWM_SOLVERS[0][0], TWM_SOLVERS[1][0]):
            s.processor._init_pe()
        return {"status": "success", "message": "WM processor PE reset to defaults"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
