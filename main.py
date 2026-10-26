from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from calculator import (
    eval_single_mode,
    eval_pattern_form,
    eval_twm_solve,
    eval_operator_combination,
)


def _patch_reloader_ignore_draft():
    """
    Patch uvicorn and watchfiles reloaders to ignore changes in draft.py and temporary files.
    This prevents the development server from reloading whenever draft.py is modified.
    """
    # 1. In-memory monkey patch for uvicorn WatchFiles FileFilter
    try:
        from uvicorn.supervisors import watchfilesreload
        _orig_file_filter = watchfilesreload.FileFilter.__call__

        def _patched_file_filter(self, path: Path):
            name = path.name.lower()
            path_str = str(path).lower()
            if "draft" in name or name.endswith(".ipynb") or "scratch" in path_str:
                return False
            return _orig_file_filter(self, path)

        watchfilesreload.FileFilter.__call__ = _patched_file_filter
    except Exception:
        pass

    # 2. In-memory monkey patch for watchfiles filters
    try:
        import watchfiles.filters
        _orig_wf_filter = watchfiles.filters.BaseFilter.__call__

        def _patched_wf_filter(self, change, path: str):
            p = str(path).lower()
            if "draft" in p or p.endswith(".ipynb") or "scratch" in p:
                return False
            return _orig_wf_filter(self, change, path)

        watchfiles.filters.BaseFilter.__call__ = _patched_wf_filter
    except Exception:
        pass

    # 3. Ensure persistent hook in virtualenv site-packages so uvicorn CLI also ignores draft
    try:
        import site
        sp_dirs = site.getsitepackages() if hasattr(site, "getsitepackages") else []
        for sp in sp_dirs:
            sp_path = Path(sp)
            if sp_path.exists() and (sp_path / "uvicorn").exists():
                patch_py = sp_path / "_draft_reload_patch.py"
                pth_file = sp_path / "_draft_reload_patch.pth"
                if not patch_py.exists():
                    patch_py.write_text(
                        'from pathlib import Path\n'
                        'try:\n'
                        '    from uvicorn.supervisors import watchfilesreload\n'
                        '    _orig = watchfilesreload.FileFilter.__call__\n'
                        '    def _f(self, path: Path):\n'
                        '        name = path.name.lower()\n'
                        '        path_str = str(path).lower()\n'
                        '        if "draft" in name or name.endswith(".ipynb") or "scratch" in path_str:\n'
                        '            return False\n'
                        '        return _orig(self, path)\n'
                        '    watchfilesreload.FileFilter.__call__ = _f\n'
                        'except Exception:\n'
                        '    pass\n'
                        'try:\n'
                        '    import watchfiles.filters\n'
                        '    _orig_wf = watchfiles.filters.BaseFilter.__call__\n'
                        '    def _f_wf(self, change, path: str):\n'
                        '        p = str(path).lower()\n'
                        '        if "draft" in p or p.endswith(".ipynb") or "scratch" in p:\n'
                        '            return False\n'
                        '        return _orig_wf(self, change, path)\n'
                        '    watchfiles.filters.BaseFilter.__call__ = _f_wf\n'
                        'except Exception:\n'
                        '    pass\n',
                        encoding="utf-8"
                    )
                if not pth_file.exists():
                    pth_file.write_text("import _draft_reload_patch\n", encoding="utf-8")
                break
    except Exception:
        pass


_patch_reloader_ignore_draft()

app = FastAPI(title="Quantum Operator Calculator")


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


class OperatorDef(BaseModel):
    kind: str = "boson"  # 'boson' or 'fermion'
    name: str
    display_name: Optional[str] = None
    mode_index: int = 0


class MMOperatorDef(BaseModel):
    name: str
    mm_kind: str = "COP"  # 'COP' (MultimodeCOP) or 'Num2' (MultimodeNum2)


class NamedExprDef(BaseModel):
    name: str
    expr_str: str = ""
    expr_str_t: Optional[str] = None
    expr_str_g: Optional[str] = None
    order: Optional[int] = None
    lowest_order: Optional[int] = None


class PatternFormRequest(BaseModel):
    text: str
    symbols: List[SymbolDef] = []
    operators: List[OperatorDef] = []
    named_expressions: List[NamedExprDef] = []
    use_res: bool = False


class SolveRequest(BaseModel):
    dc: bool
    mismatch: bool
    var_index: int
    order: int
    use_gain: bool = False


class CombinationRequest(BaseModel):
    text: str
    dc: bool = True
    mismatch: bool = False
    order: int = 1
    use_gain: bool = False
    symbols: List[SymbolDef] = []
    operators: List[OperatorDef] = []
    mm_operators: List[MMOperatorDef] = []
    named_expressions: List[NamedExprDef] = []


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/api/single_mode")
def api_single_mode(req: SingleModeRequest):
    try:
        latex_str, is_herm, expr_str = eval_single_mode(
            req.type, req.name, req.first_op, req.exponents, req.action
        )
        return {
            "latex": latex_str,
            "is_hermitian": is_herm,
            "expr_str": expr_str
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/pattern_form")
def api_pattern_form(req: PatternFormRequest):
    try:
        latex_str, expr_str, res_latex, res_str = eval_pattern_form(
            req.text,
            [s.model_dump() for s in req.symbols],
            [op.model_dump() for op in req.operators],
            [ne.model_dump() for ne in req.named_expressions],
            use_res=req.use_res
        )
        return {
            "latex": latex_str,
            "expr_str": expr_str,
            "residue_latex": res_latex,
            "residue_str": res_str
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/solve")
def solve(req: SolveRequest):
    try:
        latex_curr, expr_curr, latex_t, expr_t, latex_g, expr_g = eval_twm_solve(
            req.dc, req.mismatch, req.var_index, req.order, use_gain=req.use_gain
        )
        return {
            "latex": latex_curr,
            "expr_str": expr_curr,
            "latex_t": latex_t,
            "expr_str_t": expr_t,
            "latex_g": latex_g,
            "expr_str_g": expr_g
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/combination")
def api_combination(req: CombinationRequest):
    try:
        latex_curr, expr_curr, latex_t, expr_t, latex_g, expr_g, lowest_order = eval_operator_combination(
            req.text,
            dc=req.dc,
            mismatch=req.mismatch,
            order=req.order,
            use_gain=req.use_gain,
            symbols=[s.model_dump() for s in req.symbols],
            operators=[op.model_dump() for op in req.operators],
            mm_operators=[mm.model_dump() for mm in req.mm_operators],
            named_expressions=[ne.model_dump() for ne in req.named_expressions],
        )
        return {
            "latex": latex_curr,
            "expr_str": expr_curr,
            "latex_t": latex_t,
            "expr_str_t": expr_t,
            "latex_g": latex_g,
            "expr_str_g": expr_g,
            "lowest_order": lowest_order,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_excludes=["*draft*", "draft.py", "*.ipynb", "scratch/*"],
    )



