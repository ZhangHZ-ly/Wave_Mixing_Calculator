from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from calculator import (
    eval_single_mode,
    eval_pf_form,
    eval_operator_combination,
    eval_combination_assertions,
    eval_cascaded_combination,
    save_latest_pf_variable,
    save_latest_wm_variable,
    TWM_SOLVERS,
    FWM_SOLVER,
    normalize_process,
    process_is_dc,
    is_degenerate,
    order_text,
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
    # exact text such as '1/2' (parsed as a Rational by the backend)
    order: str = '0'


class OperatorDef(BaseModel):
    type: Optional[str] = 'evolved_op'
    kind: str = "boson"  # 'boson' or 'fermion'
    name: str
    display_name: Optional[str] = None
    mode_index: Optional[int] = 0
    # exact text such as '1/2' (parsed as a Rational by the backend)
    order: str = '0'


class MMOperatorDef(BaseModel):
    name: str
    mm_kind: str = "COP"  # 'COP' (MultimodeCOP) or 'Num2' (MultimodeNum2)


class NamedExprDef(BaseModel):
    # definition of a saved expression; `name` is None for a definition no longer in the table
    # but still referenced by others
    id: str
    name: Optional[str] = None
    source_input: Optional[str] = None
    # name in the source input -> id of the definition it referred to when saved
    refs: Dict[str, str] = {}
    # orders as exact text (e.g. '1/2')
    order: Optional[str] = None
    leading_order: Optional[str] = None


class SaveVarRequest(BaseModel):
    workspace: str
    id: str


class PatternFormRequest(BaseModel):
    text: str
    symbols: List[SymbolDef] = []
    operators: List[OperatorDef] = []
    named_expressions: List[NamedExprDef] = []
    use_rem: bool = False
    expand: bool = False


class CombinationRequest(BaseModel):
    text: str
    process: Optional[str] = None
    mismatch: bool = False
    order: int = 1
    use_gain: bool = False
    single_mode: bool = False
    # degenerate signal / idler (effective only with single_mode in down-conversion)
    degenerate: bool = False
    symbols: List[SymbolDef] = []
    operators: List[OperatorDef] = []
    mm_operators: List[MMOperatorDef] = []
    named_expressions: List[NamedExprDef] = []


class AssertionRequest(CombinationRequest):
    coherent_drive: bool = False
    vacuum_signal: bool = False
    vacuum_other_weak: bool = False
    # coherent seeding of a separate weak mode (None: no seeding); see WMWrapper.seeded_separate:
    # down-conversion 1 (signal) / 0 (degenerate), state transfer 1 (signal) / 0 (pump)
    seed_mode: Optional[int] = None
    # whether the coupled mode is in vacuum (else in its coherent state)
    seed_vac_c: bool = True
    # whether the single-mode weak-field operators are the seeded modes (multimode processing)
    seed_observed: bool = False


class CascadeRequest(AssertionRequest):
    cascade_symbol: str
    # phases (phi_0, phi_1, phi_2) of the input operators: real numbers or symbols, '' for 0
    cascade_phases: List[str] = ['', '', '']


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
        latex_str, is_herm, note = eval_single_mode(
            req.type, req.name, req.first_op, req.exponents, req.action
        )
        return {
            "latex": latex_str,
            "is_hermitian": is_herm,
            "note": note
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/pf_form")
def api_pattern_form(req: PatternFormRequest):
    try:
        use_rem = req.use_rem
        latex_str, rem_latex = eval_pf_form(
            req.text,
            [s.model_dump() for s in req.symbols],
            [op.model_dump() for op in req.operators],
            [ne.model_dump() for ne in req.named_expressions],
            use_rem=use_rem,
            expand=req.expand,
        )
        return {
            "latex": latex_str,
            "remainder_latex": rem_latex,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/combination")
def api_combination(req: CombinationRequest):
    try:
        proc = normalize_process(req.process)
        is_dc = process_is_dc(proc)
        latex_curr, latex_t, latex_g, leading_order, sol_len, res_order = eval_operator_combination(
            req.text,
            dc=is_dc,
            mismatch=req.mismatch,
            order=req.order,
            use_gain=req.use_gain,
            single_mode=req.single_mode,
            degenerate=req.degenerate,
            symbols=[s.model_dump() for s in req.symbols],
            operators=[op.model_dump() for op in req.operators],
            mm_operators=[mm.model_dump() for mm in req.mm_operators],
            named_expressions=[ne.model_dump() for ne in req.named_expressions],
            process=proc,
        )
        lead_ord = order_text(leading_order)
        return {
            "latex": str(latex_curr),
            "latex_t": str(latex_t),
            "latex_g": str(latex_g),
            "leading_order": lead_ord,
            # accuracy order of the result (the product rule may lower it below the preset)
            "order": order_text(res_order),
            "sol_len": int(sol_len),
            "process": proc,
            "single_mode": bool(req.single_mode),
            "degenerate": is_degenerate(proc, req.single_mode, req.degenerate),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/combination/assert")
def api_combination_assert(req: AssertionRequest):
    try:
        proc = normalize_process(req.process)
        is_dc = process_is_dc(proc)
        latex_curr, latex_t, latex_g, leading_order, sol_len, res_order = eval_combination_assertions(
            req.text,
            dc=is_dc,
            mismatch=req.mismatch,
            order=req.order,
            use_gain=req.use_gain,
            single_mode=req.single_mode,
            degenerate=req.degenerate,
            coherent_drive=req.coherent_drive,
            vacuum_signal=req.vacuum_signal,
            vacuum_other_weak=req.vacuum_other_weak,
            seed_mode=req.seed_mode,
            seed_vac_c=req.seed_vac_c,
            seed_observed=req.seed_observed,
            symbols=[s.model_dump() for s in req.symbols],
            operators=[op.model_dump() for op in req.operators],
            mm_operators=[mm.model_dump() for mm in req.mm_operators],
            named_expressions=[ne.model_dump() for ne in req.named_expressions],
            process=proc,
        )
        lead_ord = order_text(leading_order)
        return {
            "latex": str(latex_curr),
            "latex_t": str(latex_t),
            "latex_g": str(latex_g),
            "leading_order": lead_ord,
            # accuracy order of the result (the product rule may lower it below the preset)
            "order": order_text(res_order),
            "sol_len": int(sol_len),
            "process": proc,
            "single_mode": bool(req.single_mode),
            "degenerate": is_degenerate(proc, req.single_mode, req.degenerate),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/combination/cascade")
def api_combination_cascade(req: CascadeRequest):
    try:
        proc = normalize_process(req.process)
        is_dc = process_is_dc(proc)
        latex_curr, latex_t, latex_g, leading_order, sol_len, res_order = eval_cascaded_combination(
            req.text,
            dc=is_dc,
            mismatch=req.mismatch,
            order=req.order,
            use_gain=req.use_gain,
            single_mode=req.single_mode,
            degenerate=req.degenerate,
            cascade_symbol=req.cascade_symbol,
            cascade_phases=req.cascade_phases,
            coherent_drive=req.coherent_drive,
            vacuum_signal=req.vacuum_signal,
            vacuum_other_weak=req.vacuum_other_weak,
            seed_mode=req.seed_mode,
            seed_vac_c=req.seed_vac_c,
            seed_observed=req.seed_observed,
            symbols=[s.model_dump() for s in req.symbols],
            operators=[op.model_dump() for op in req.operators],
            mm_operators=[mm.model_dump() for mm in req.mm_operators],
            named_expressions=[ne.model_dump() for ne in req.named_expressions],
            process=proc,
        )
        lead_ord = order_text(leading_order)
        return {
            "latex": str(latex_curr),
            "latex_t": str(latex_t),
            "latex_g": str(latex_g),
            "leading_order": lead_ord,
            # accuracy order of the result (the product rule may lower it below the preset)
            "order": order_text(res_order),
            "sol_len": int(sol_len),
            "process": proc,
            "single_mode": bool(req.single_mode),
            "degenerate": is_degenerate(proc, req.single_mode, req.degenerate),
            "cascade_symbol": req.cascade_symbol,
            "cascade_phases": req.cascade_phases,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/variable/save")
def api_save_variable(req: SaveVarRequest):
    try:
        if req.workspace == 'tab3':
            save_latest_wm_variable(req.id)
        else:
            save_latest_pf_variable(req.id)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class ResetPERequest(BaseModel):
    clear: bool = False  # True: drop all custom orders; False: only restore the base mode orders


@app.post("/api/combination/reset")
def api_reset_combination(req: Optional[ResetPERequest] = None):
    clear = bool(req and req.clear)
    try:
        for p in {s.processor for pair in TWM_SOLVERS for s in pair} | {FWM_SOLVER.processor}:
            p._init_pe(clear=clear)
        msg = "WM processor PE reset to defaults" if clear else "WM processor base mode orders restored"
        return {"status": "success", "message": msg, "clear": clear}
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
