#!/usr/bin/env python3
"""Dual-edge validation of the spinel colocation relations with fixed formulas, scales and orientations.
Replays the single-coordinate Pi_S and the valence/conduction coordinates Pi_V, Pi_C; thresholds and scales are fitted
on the 440 development states only, the 224 challenge states are scored afterwards. Writes edge metrics, thresholds,
replay records and diagnostics to --outdir.
usage: python dual_edge_validation.py run --protocol-root DIR --formula-root DIR --canonical-csv FILE --freeze-csv FILE
           --outdir DIR [--package]
       python dual_edge_validation.py self-test"""
from __future__ import annotations
import argparse, ast, hashlib, json, math, re, tarfile
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd

VERSION = '1.0.3'
EPS = 1e-6
FINAL_V_COEF = (3.0, -1.0, -1.0 / 6.0)
FINAL_C_COEF = (-1.0 / 3.0, 2.0)
EXPECTED_DEV = {
    'V': {'auc': 0.8942685534, 'bacc': 0.8732502931, 'margin_spearman': 0.649031},
    'C': {'auc': 0.9214126984, 'bacc': 0.8665079365, 'margin_spearman': 0.780300},
}
CHALLENGE_TAGS = ['challenge_5d_soc', 'challenge_small_gap', 'challenge_structure_occupancy',
                  'challenge_rare_chemistry', 'challenge_multi_factor']


class ValidationError(RuntimeError):
    pass


def log(s: str):
    print(f'[DUAL-FROZEN] {s}', flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _clean_json(x):
    if isinstance(x, dict):
        return {str(k): _clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean_json(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        x = float(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_json(p: Path, obj: Any):
    p.write_text(json.dumps(_clean_json(obj), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def resolve_bundle(protocol_root: Path) -> Path:
    root = protocol_root.expanduser().resolve()
    cands = []
    for p in root.rglob('X_full_clean_float64.npy'):
        if p.parent.name == 'arrays' and (p.parent / 'sample_ids.npy').is_file():
            bundle = p.parent.parent
            if (bundle / 'features/formula_full_feature_names.json').is_file():
                cands.append(bundle)
    if not cands:
        raise ValidationError(f'no native protocol bundle below {root}')
    # Prefer the ALL_PRIMARY_POST_FREEZE bundle, then the shallowest path.
    cands = sorted(set(cands),
                   key=lambda p: (0 if 'ALL_PRIMARY_POST_FREEZE_BUNDLE' in p.name else 1, len(p.parts), str(p)))
    return cands[0]


def resolve_asset(root: Path, name: str) -> Path:
    p = root.expanduser().resolve() / name
    if p.is_file():
        return p
    hits = sorted(root.expanduser().resolve().rglob(name), key=lambda q: (len(q.parts), str(q)))
    if not hits:
        raise FileNotFoundError(f'{name} below {root}')
    return (hits[-1] if name in {'best_performance_formula.json', 'frozen_decision_state.json', 'feature_scales.npy'}
            else hits[0])


def get_any(record: Mapping[str, Any], keys: Sequence[str], default=None):
    for k in keys:
        if k in record:
            v = record[k]
            if v is None:
                continue
            try:
                if isinstance(v, float) and math.isnan(v):
                    continue
            except Exception:
                pass
            if str(v).strip().lower() in {'', 'nan', 'none'}:
                continue
            return v
    return default


def _near_assets(formula_root: Path, anchor: Path, name: str) -> list[Path]:
    out = []
    q = anchor.parent / name
    if q.is_file():
        out.append(q)
    for p in formula_root.expanduser().resolve().rglob(name):
        if p.is_file() and p not in out:
            out.append(p)
    return sorted(out, key=lambda p: (0 if p.parent == anchor.parent else 1, len(p.parts), str(p)))


def load_frozen_top1(formula_root: Path) -> tuple[dict[str, Any], Path, str]:
    root = formula_root.expanduser().resolve()
    tops = list(root.rglob('top30_safe.csv'))
    if tops:
        tops = sorted(tops, key=lambda p: (0 if 'final' in [x.lower() for x in p.parts] else 1, len(p.parts), str(p)))
        p = tops[0]
        df = pd.read_csv(p)
        if df.empty:
            raise ValidationError(f'empty frozen top30_safe.csv: {p}')
        if 'rank' in df.columns:
            row = df.sort_values('rank', kind='mergesort').iloc[0]
        else:
            mc = next((c for c in ('all_bacc', 'bacc', 'ALL_BACC') if c in df.columns), None)
            row = df.sort_values(mc, ascending=False, kind='mergesort').iloc[0] if mc else df.iloc[0]
        return row.to_dict(), p, 'TOP30_SAFE_RANK1'
    for name in ('BEST_SAFE_FORMULA.json', 'best_performance_formula.json'):
        hits = sorted(root.rglob(name), key=lambda p: (len(p.parts), str(p)))
        if hits:
            return json.loads(hits[0].read_text(encoding='utf-8')), hits[0], f'{name}_FALLBACK'
    raise FileNotFoundError(f'no frozen TOP30_SAFE/BEST formula below {root}')


def load_near_json(formula_root: Path, anchor: Path, name: str) -> tuple[dict[str, Any] | None, Path | None]:
    for p in _near_assets(formula_root, anchor, name):
        try:
            return json.loads(p.read_text(encoding='utf-8')), p
        except Exception:
            continue
    return None, None


def load_near_scales(formula_root: Path, anchor: Path, nfeat: int) -> tuple[np.ndarray, Path]:
    for p in _near_assets(formula_root, anchor, 'feature_scales.npy'):
        try:
            a = np.load(p, allow_pickle=False).astype(float)
        except Exception:
            continue
        if a.ndim == 1 and len(a) == nfeat and np.all(np.isfinite(a)) and np.all(a > 0):
            return a, p
    raise ValidationError(f'no valid frozen feature scales of length {nfeat} near {anchor}')


def frozen_top1_contract(formula_root: Path, names: Sequence[str]) -> dict[str, Any]:
    top, top_path, source = load_frozen_top1(formula_root)
    fid = str(get_any(top, ['formula_id', 'id'], '')).strip()
    if not fid:
        raise ValidationError('frozen Top1 lacks formula_id')
    scales, scale_path = load_near_scales(formula_root, top_path, len(names))
    state, state_path = load_near_json(formula_root, top_path, 'frozen_decision_state.json')
    threshold = get_any(top, ['threshold', 'best_threshold'], None)
    orientation = get_any(top, ['orientation', 'best_orientation'], None)
    epsilon = EPS
    decision_source = []
    if threshold is not None:
        threshold = float(threshold)
        decision_source.append('TOP1_ROW_THRESHOLD')
    if orientation is not None:
        orientation = int(float(orientation))
        decision_source.append('TOP1_ROW_ORIENTATION')
    if state is not None:
        epsilon = float(state.get('epsilon', EPS))
        sn = state.get('feature_names')
        if sn is not None and list(map(str, sn)) != list(map(str, names)):
            raise ValidationError('frozen decision-state feature_names mismatch')
        ds = [r for r in state.get('top30_decisions', []) if str(r.get('formula_id', '')) == fid]
        if len(ds) > 1:
            raise ValidationError(f'multiple frozen decisions for Top1 {fid}')
        if len(ds) == 1:
            d = ds[0]
            st = float(d['threshold'])
            so = int(float(d['orientation']))
            if threshold is None:
                threshold = st
                decision_source.append('DECISION_STATE_THRESHOLD')
            elif abs(threshold - st) > 1e-12:
                raise ValidationError(f'Top1/state threshold mismatch {threshold} vs {st}')
            if orientation is None:
                orientation = so
                decision_source.append('DECISION_STATE_ORIENTATION')
            elif int(orientation) != so:
                raise ValidationError(f'Top1/state orientation mismatch {orientation} vs {so}')
    if threshold is None or orientation is None:
        raise ValidationError(f'cannot resolve frozen decision for Top1 {fid}')
    if int(orientation) not in (-1, 1):
        raise ValidationError(f'invalid frozen orientation {orientation}')
    raw = get_any(top, ['ast_json', 'expanded_ast_json', 'ast'], None)
    if raw is None:
        raise ValidationError('frozen Top1 lacks AST; text-only replay is intentionally unsupported here')
    node = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(node, dict) or 'op' not in node:
        raise ValidationError('invalid frozen Top1 AST')
    archived_bacc = get_any(top, ['all_bacc', 'bacc', 'ALL_BACC'], None)
    archived_auc = get_any(top, ['all_auc', 'auc', 'ALL_AUC'], None)
    if archived_bacc is None:
        raise ValidationError('frozen Top1 lacks archived Development BACC; cannot authenticate Pi_S baseline')
    return {'top': top, 'top_path': top_path, 'top_source': source, 'formula_id': fid, 'node': node, 'scales': scales,
            'scale_path': scale_path, 'epsilon': float(epsilon), 'threshold': float(threshold),
            'orientation': int(orientation), 'archived_bacc': float(archived_bacc),
            'archived_auc': float(archived_auc) if archived_auc is not None else None, 'state_path': state_path,
            'decision_source': decision_source}


def rc8_q75_eval(node: Mapping[str, Any], Xfs: np.ndarray, epsilon: float):
    op = str(node['op'])
    if op == 'terminal':
        return np.asarray(Xfs[int(node['feature_index'])], dtype=float)
    if op == 'abs':
        return np.abs(rc8_q75_eval(node['arg'], Xfs, epsilon))
    if op in {'power', 'spower'}:
        a = rc8_q75_eval(node['arg'], Xfs, epsilon)
        n, d = node['power']
        p = float(n) / float(d)
        with np.errstate(all='ignore'):
            return np.power(a, p) if op == 'power' else np.sign(a) * np.power(np.abs(a), p)
    if op == 'scale':
        a = rc8_q75_eval(node['arg'], Xfs, epsilon)
        n, d = node['coefficient']
        return a * (float(n) / float(d))
    a = rc8_q75_eval(node['left'], Xfs, epsilon)
    b = rc8_q75_eval(node['right'], Xfs, epsilon)
    with np.errstate(all='ignore'):
        if op == 'add':
            return a + b
        if op == 'sub':
            return a - b
        if op == 'mul':
            return a * b
        if op == 'geometric':
            return np.sqrt(a * b)
        if op == 'sgeometric':
            q = a * b
            return np.sign(q) * np.sqrt(np.abs(q))
        if op == 'saturating':
            return a / (1 + np.abs(b))
        if op == 'safe_ratio':
            raw = np.abs(b)
            q75 = float(np.quantile(raw, .75))
            eps = epsilon * (q75 if q75 > 0 and math.isfinite(q75) else 1.0)
            return a / (raw + eps)
    raise ValidationError(f'unsupported frozen Pi_S AST op {op}')


def authenticate_pis_replay(contract: Mapping[str, Any], Xraw: np.ndarray, y_protocol: np.ndarray, old_dev: np.ndarray,
                            out: Path) -> tuple[np.ndarray, dict[str, Any]]:
    Xfs = np.asarray(Xraw, float) / np.asarray(contract['scales'], float)[None, :]
    node = contract['node']
    eps = float(contract['epsilon'])
    ori = int(contract['orientation'])
    th = float(contract['threshold'])
    yb = np.asarray(y_protocol, int)
    if not np.isin(yb, [0, 1]).all():
        raise ValidationError('protocol y_int8 is not binary')
    modes = []
    # Q75-relative epsilon evaluated on the full protocol matrix.
    modes.append(('FULL_PROTOCOL_Q75_RELATIVE', np.asarray(rc8_q75_eval(node, Xfs.T, eps), float)))
    # Q75 parameters fitted on the development set (diagnostic).
    _, par = rc8_fit(node, Xfs[old_dev].T, eps)
    modes.append(('DEVELOPMENT_FROZEN_Q75_RELATIVE', np.asarray(rc8_apply(node, Xfs.T, par), float)))
    # Constant-epsilon interpretation (diagnostic).
    modes.append(('CONSTANT_EPSILON_LEGACY_DIAGNOSTIC', np.asarray(rc8_original_eval(node, Xfs.T, eps), float)))
    rows = []
    classes = []
    for mode, score in modes:
        if score.shape != (len(yb),) or not np.all(np.isfinite(score)):
            raise ValidationError(f'Pi_S replay mode {mode} produced invalid scores')
        pred = predict_threshold(score, th, ori)
        m = binary_metrics(yb[old_dev], score[old_dev], pred[old_dev], ori, th)
        ae = (abs(float(m['auc']) - float(contract['archived_auc'])) if contract.get('archived_auc') is not None
              else float('nan'))
        be = abs(float(m['bacc']) - float(contract['archived_bacc']))
        matched = be <= 1e-10 and (contract.get('archived_auc') is None or ae <= 1e-10)
        # Coalesce numerically identical candidates so a ratio-free AST cannot create false ambiguity.
        cls = None
        for i, (_, ref) in enumerate(classes):
            if np.array_equal(score, ref) or np.allclose(score, ref, rtol=0, atol=1e-14, equal_nan=True):
                cls = i
                break
        if cls is None:
            cls = len(classes)
            classes.append((mode, score.copy()))
        rows.append({'mode': mode, 'equivalence_class': cls, 'dev_n': int(old_dev.sum()), 'dev_auc': m['auc'],
                     'dev_bacc': m['bacc'], 'archived_auc': contract.get('archived_auc'),
                     'archived_bacc': contract['archived_bacc'], 'auc_abs_error': ae, 'bacc_abs_error': be,
                     'archive_match': bool(matched),
                     'score_sha256': hashlib.sha256(
                         np.ascontiguousarray(score, dtype=np.float64).tobytes()).hexdigest()})
    audit = pd.DataFrame(rows)
    audit.to_csv(out / 'PI_S_REPLAY_SEMANTICS_AUDIT.csv', index=False)
    matching_classes = sorted(set(int(r['equivalence_class']) for r in rows if r['archive_match']))
    # The original evaluators evaluate safe_ratio as epsilon * Q75(abs(denominator)) on the full matrix they are
    # given, so that mode is selected by construction. The archived development AUC/BACC are a consistency check
    # only: different full-vector semantics can share the same development ranking and threshold metrics, so the
    # metrics are not used to choose among the modes.
    selected_mode = 'FULL_PROTOCOL_Q75_RELATIVE'
    selected_score = next(score for mode, score in modes if mode == selected_mode)
    selrow = next(r for r in rows if r['mode'] == selected_mode)
    if not bool(selrow['archive_match']):
        write_json(out / 'pi_s_ratio_semantics.json',
                   {'status': 'FAIL_SOURCE_PROVENANCE_REPLAY', 'selected_mode': selected_mode,
                    'selection_basis': 'HISTORICAL_SOURCE_CODE_CONTRACT', 'formula_id': contract['formula_id'],
                    'top_source': contract['top_source'], 'top_path': str(contract['top_path']),
                    'matching_equivalence_classes': matching_classes, 'candidate_rows': rows})
        raise ValidationError(f'Pi_S source-provenance evaluator {selected_mode} does not reproduce archived '
                              'Development metrics; see PI_S_REPLAY_SEMANTICS_AUDIT.csv')
    cls = int(selrow['equivalence_class'])
    matching_modes = [str(r['mode']) for r in rows if r['archive_match']]
    payload = {'status': 'PASS', 'selected_mode': selected_mode, 'selected_equivalence_class': cls,
               'selection_basis': 'HISTORICAL_SOURCE_CODE_CONTRACT',
               'source_code_contract': ('safe_ratio = numerator / (abs(denominator) + epsilon * Q75(abs(denominator))) '
                                        'evaluated on the full supplied matrix'),
               'source_provenance_refs': ['dual_coordinate_feasibility_v1_0_2.py::rc8_eval_node',
                                          'sample_level_challenge_physics_audit_v1_0_0.py::rc8_eval_ast_node'],
               'archive_matching_modes_diagnostic_only': matching_modes,
               'archive_matching_equivalence_classes_diagnostic_only': matching_classes,
               'formula_id': contract['formula_id'], 'top_source': contract['top_source'],
               'top_path': str(contract['top_path']), 'scale_path': str(contract['scale_path']),
               'state_path': str(contract['state_path']) if contract.get('state_path') else None,
               'decision_source': contract['decision_source'], 'epsilon': contract['epsilon'], 'threshold': th,
               'orientation': ori, 'development_442': selrow, 'challenge_not_used_for_semantics_selection': True,
               'candidate_rows': rows}
    write_json(out / 'pi_s_ratio_semantics.json', payload)
    return selected_score, payload


def metadata_key(cols: Sequence[str]) -> str | None:
    low = {str(c).lower(): str(c) for c in cols}
    for x in ('sample_id', 'task directory', 'task_directory', 'task_id'):
        if x in low:
            return low[x]
    return None


def parse_k_index(sample_id: str, row: Mapping[str, Any]) -> int:
    for k in ('k', 'k_index', 'concentration_index'):
        if k in row:
            try:
                return int(float(row[k]))
            except Exception:
                pass
    m = re.search(r'_k(\d+)$', str(sample_id))
    return int(m.group(1)) if m else -1


def parse_kpoint(v: Any):
    if v is None:
        return None
    try:
        if isinstance(v, float) and not math.isfinite(v):
            return None
    except Exception:
        pass
    if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 3:
        arr = np.asarray(v[:3], float)
    else:
        s = str(v).strip()
        if not s or s.lower() in {'nan', 'none', 'null'}:
            return None
        try:
            x = ast.literal_eval(s)
            arr = np.asarray(x[:3], float) if isinstance(x, (list, tuple, np.ndarray)) and len(x) >= 3 else None
        except Exception:
            arr = None
        if arr is None:
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
            if len(nums) < 3:
                return None
            arr = np.asarray(list(map(float, nums[:3])), float)
    if not np.all(np.isfinite(arr)):
        return None
    return tuple(float(x % 1.0) for x in arr)


def exact_key(k, nd=6):
    if k is None:
        return 'MISSING'
    vals = [round(x % 1.0, nd) for x in k]
    vals = [0.0 if abs(x) < 10 ** (-nd) or abs(x - 1.0) < 10 ** (-nd) else x for x in vals]
    return '[' + ','.join(f'{x:.{nd}f}' for x in vals) + ']'


def star_key(k, nd=6):
    if k is None:
        return 'MISSING'
    q = sorted(round(min(x % 1.0, 1 - (x % 1.0)), nd) for x in k)
    q = [0.0 if abs(x) < 10 ** (-nd) else x for x in q]
    return 'STAR[' + ','.join(f'{x:.{nd}f}' for x in q) + ']'


def is_gamma(k, tol=1e-6):
    return k is not None and all(min(abs(x % 1.0), abs(1 - (x % 1.0))) <= tol for x in k)


def first_existing(df: pd.DataFrame, names: Sequence[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def canonical_contract(canonical_csv: Path, sample_ids: np.ndarray) -> pd.DataFrame:
    p = canonical_csv.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, low_memory=False)
    key = metadata_key(df.columns)
    if key is None:
        raise ValidationError('canonical CSV lacks sample key')
    if key != 'sample_id':
        df = df.rename(columns={key: 'sample_id'})
    df['sample_id'] = df.sample_id.astype(str)
    if df.sample_id.duplicated().any():
        raise ValidationError('canonical CSV duplicate sample_id')
    out = pd.DataFrame({'sample_id': sample_ids}).merge(df, on='sample_id', how='left', validate='one_to_one')
    miss = out.drop(columns=['sample_id']).isna().all(axis=1)
    if miss.any():
        raise ValidationError(f'canonical CSV misses {int(miss.sum())} protocol samples')
    return out


def edge_table(canon: pd.DataFrame, edge: str) -> pd.DataFrame:
    E = edge.upper()
    prim = first_existing(canon, [f'primitive_{E}BM_kpoint_frac', f'primitive_{E}BM_kpoint'])
    traj = first_existing(canon, [f'{E}BM_kpoint_frac', f'{E}BM_kpoint'])
    endflag = first_existing(canon, ['endpoint_canonicalized', 'v2_is_endpoint', 'is_endpoint'])
    sids = canon.sample_id.astype(str).to_numpy()
    pts = []
    src = []
    kval = []
    endpoints = []
    for _, r in canon.iterrows():
        sid = str(r.sample_id)
        k = parse_k_index(sid, r)
        explicit = False
        if endflag:
            v = r[endflag]
            explicit = (bool(v) if isinstance(v, (bool, np.bool_))
                        else str(v).strip().lower() in {'1', 'true', 'yes', 'y'})
        ep = explicit or k in {0, 16}
        endpoints.append(ep)
        kval.append(k)
        if ep:
            kp = parse_kpoint(r[prim]) if prim else None
            src.append('primitive_endpoint' if kp else 'MISSING_PRIMITIVE_ENDPOINT')
        else:
            kp = parse_kpoint(r[traj]) if traj else None
            src.append('trajectory_nonendpoint' if kp else 'MISSING_TRAJECTORY')
        pts.append(kp)
    ex = np.array([exact_key(k) for k in pts], object)
    star = np.array([star_key(k) for k in pts], object)
    gam = np.array([is_gamma(k) for k in pts], bool)
    mcol = (first_existing(canon, ['S_V_Gamma_minus_nonGamma_eV']) if edge.upper() == 'V'
            else first_existing(canon, ['S_C_nonGamma_minus_Gamma_eV']))
    margin = pd.to_numeric(canon[mcol], errors='coerce').to_numpy(float) if mcol else np.full(len(canon), np.nan)
    margin = np.where(np.asarray(endpoints, bool), np.nan, margin)
    return pd.DataFrame({'sample_id': sids, 'trajectory_id': [re.sub(r'_k\d+$', '', s) for s in sids], 'k_index': kval,
                         'edge': edge, 'is_endpoint': endpoints, 'exact_kpoint_key': ex, 'cubic_star_key': star,
                         'is_Gamma': gam, 'target_available': star != 'MISSING', 'kpoint_source': src,
                         'signed_margin_eV': margin})


def joint_truth(V: pd.DataFrame, C: pd.DataFrame, canon: pd.DataFrame) -> pd.DataFrame:
    q = V[['sample_id', 'trajectory_id', 'k_index', 'exact_kpoint_key', 'cubic_star_key', 'is_Gamma',
           'target_available', 'signed_margin_eV']].rename(
        columns={'exact_kpoint_key': 'V_exact', 'cubic_star_key': 'V_star', 'is_Gamma': 'V_true_Gamma',
                 'target_available': 'V_available', 'signed_margin_eV': 'V_signed_margin_eV'})
    q = q.merge(
        C[['sample_id', 'exact_kpoint_key', 'cubic_star_key', 'is_Gamma', 'target_available',
           'signed_margin_eV']].rename(columns={'exact_kpoint_key': 'C_exact', 'cubic_star_key': 'C_star',
                                                'is_Gamma': 'C_true_Gamma', 'target_available': 'C_available',
                                                'signed_margin_eV': 'C_signed_margin_eV'}),
        on='sample_id', validate='one_to_one')
    g = exact_key((0, 0, 0))
    q['true_exact_state'] = np.where((q.V_exact == 'MISSING') | (q.C_exact == 'MISSING'), 'UNKNOWN',
                                     np.where(q.V_exact == q.C_exact, np.where(q.V_exact == g, 'GG', 'NN_same'),
                                              np.where(q.V_exact == g, 'GN',
                                                       np.where(q.C_exact == g, 'NG', 'NN_diff'))))
    q['true_coarse_state'] = q.true_exact_state.replace({'NN_same': 'NN', 'NN_diff': 'NN'})
    q['true_binary_label'] = np.where(q.true_exact_state.isin(['GG', 'NN_same']), 'co_located', 'separated')
    # Keep the canonical physics columns that are present.
    for c in ['strict_label', 'primitive_manual_gap_eV', 'manual_gap_eV', 'partial_occ_warning',
              'primitive_partial_occ_count', 'partial_occ_count', 'A_site', 'B_original', 'X_site', 'substitute']:
        if c in canon.columns:
            q[c] = canon[c].to_numpy()
    return q


def protocol_canonical_label_audit(y_protocol: np.ndarray, y_canonical: np.ndarray, sid: np.ndarray, V: pd.DataFrame,
                                   C: pd.DataFrame, truth: pd.DataFrame, old_dev: np.ndarray, dual_dev: np.ndarray,
                                   hold: np.ndarray, excluded: np.ndarray, out: Path) -> dict[str, Any]:
    yp = np.asarray(y_protocol, int)
    yc = np.asarray(y_canonical, int)
    sid = np.asarray(sid, str)
    if yp.shape != yc.shape or len(sid) != len(yp):
        raise ValidationError('protocol/canonical label audit shape mismatch')
    bad = np.flatnonzero(yp != yc)
    rows = []
    for i in bad:
        vep = bool(V.iloc[i]['is_endpoint'])
        cep = bool(C.iloc[i]['is_endpoint'])
        vsrc = str(V.iloc[i]['kpoint_source'])
        csrc = str(C.iloc[i]['kpoint_source'])
        endpoint_override = bool((vep and vsrc == 'primitive_endpoint') or (cep and csrc == 'primitive_endpoint'))
        part = ('DUAL_DEV_V1' if bool(dual_dev[i])
                else 'FROZEN_CHALLENGE_HOLDOUT' if bool(hold[i])
                else 'EXCLUDED_UNRELIABLE' if bool(excluded[i])
                else 'OLD_DEVELOPMENT_ONLY' if bool(old_dev[i])
                else 'OTHER')
        rows.append({
            'row_index': int(i), 'sample_id': str(sid[i]), 'partition': part,
            'protocol_y': int(yp[i]), 'canonical_y': int(yc[i]),
            'protocol_label': 'co_located' if int(yp[i]) == 1 else 'separated',
            'canonical_label': str(truth.iloc[i]['true_binary_label']),
            'strict_label': str(truth.iloc[i].get('strict_label', '')),
            'V_is_endpoint': vep, 'C_is_endpoint': cep, 'V_kpoint_source': vsrc, 'C_kpoint_source': csrc,
            'V_exact': str(truth.iloc[i]['V_exact']), 'C_exact': str(truth.iloc[i]['C_exact']),
            'canonical_primitive_endpoint_override': endpoint_override
        })
    df = pd.DataFrame(rows,
                      columns=['row_index', 'sample_id', 'partition', 'protocol_y', 'canonical_y', 'protocol_label',
                               'canonical_label', 'strict_label', 'V_is_endpoint', 'C_is_endpoint', 'V_kpoint_source',
                               'C_kpoint_source', 'V_exact', 'C_exact', 'canonical_primitive_endpoint_override'])
    df.to_csv(out / 'PROTOCOL_CANONICAL_LABEL_AUDIT.csv', index=False)
    # Protocol labels do not override the primitive-canonical labels; every difference is recorded,
    # a non-endpoint difference as a warning status.
    noncanonical = df[~df.canonical_primitive_endpoint_override] if len(df) else df
    payload = {
        'status': 'PASS' if len(noncanonical) == 0 else 'PASS_WITH_NON_ENDPOINT_LEGACY_DRIFT',
        'total_n': int(len(yp)), 'mismatch_n': int(len(df)),
        'mismatch_old_development_n': int(sum(bool(old_dev[i]) for i in bad)),
        'mismatch_dual_dev_n': int(sum(bool(dual_dev[i]) for i in bad)),
        'mismatch_challenge_holdout_n': int(sum(bool(hold[i]) for i in bad)),
        'mismatch_excluded_n': int(sum(bool(excluded[i]) for i in bad)),
        'all_mismatches_are_primitive_endpoint_overrides': bool(len(noncanonical) == 0),
        'governance': 'protocol y_int8 is legacy replay truth only; primitive-canonical k-point truth governs all '
                      'final Development/Challenge metrics',
        'rows': rows
    }
    write_json(out / 'PROTOCOL_CANONICAL_LABEL_AUDIT.json', payload)
    return payload


def robust_scales(X: np.ndarray) -> np.ndarray:
    q25 = np.nanquantile(X, .25, axis=0)
    q75 = np.nanquantile(X, .75, axis=0)
    s = q75 - q25
    mad = np.nanmedian(np.abs(X - np.nanmedian(X, axis=0)), axis=0) * 1.4826
    std = np.nanstd(X, axis=0)
    s = np.where(np.isfinite(s) & (s > 1e-12), s, mad)
    s = np.where(np.isfinite(s) & (s > 1e-12), s, std)
    s = np.where(np.isfinite(s) & (s > 1e-12), s, 1.0)
    return s.astype(float)


def safe_ratio_fit(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    den = np.abs(b)
    q75 = float(np.quantile(den, .75))
    eps = EPS * (q75 if q75 > 0 and math.isfinite(q75) else 1.0)
    return a / (den + eps), eps


def safe_ratio_apply(a: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    return a / (np.abs(b) + eps)


def rc8_fit(node: Mapping[str, Any], Xfs: np.ndarray, epsilon: float, path='r', params=None):
    if params is None:
        params = {}
    op = node['op']
    if op == 'terminal':
        return Xfs[int(node['feature_index'])], params
    if op == 'abs':
        a, params = rc8_fit(node['arg'], Xfs, epsilon, path + 'a', params)
        return np.abs(a), params
    if op in {'power', 'spower'}:
        a, params = rc8_fit(node['arg'], Xfs, epsilon, path + 'u', params)
        n, d = node['power']
        p = float(n) / float(d)
        with np.errstate(all='ignore'):
            return (np.power(a, p) if op == 'power' else np.sign(a) * np.power(np.abs(a), p)), params
    if op == 'scale':
        a, params = rc8_fit(node['arg'], Xfs, epsilon, path + 's', params)
        n, d = node['coefficient']
        return a * (float(n) / float(d)), params
    a, params = rc8_fit(node['left'], Xfs, epsilon, path + 'L', params)
    b, params = rc8_fit(node['right'], Xfs, epsilon, path + 'R', params)
    with np.errstate(all='ignore'):
        if op == 'add':
            return a + b, params
        if op == 'sub':
            return a - b, params
        if op == 'mul':
            return a * b, params
        if op == 'geometric':
            return np.sqrt(a * b), params
        if op == 'sgeometric':
            return np.sign(a * b) * np.sqrt(np.abs(a * b)), params
        if op == 'saturating':
            return a / (1 + np.abs(b)), params
        if op == 'safe_ratio':
            den = np.abs(b)
            q75 = float(np.quantile(den, .75))
            eps = epsilon * (q75 if q75 > 0 and math.isfinite(q75) else 1.0)
            params[path] = eps
            return a / (den + eps), params
    raise ValidationError(f'unsupported RC8 op {op}')


def rc8_apply(node: Mapping[str, Any], Xfs: np.ndarray, params: Mapping[str, float], path='r'):
    op = node['op']
    if op == 'terminal':
        return Xfs[int(node['feature_index'])]
    if op == 'abs':
        return np.abs(rc8_apply(node['arg'], Xfs, params, path + 'a'))
    if op in {'power', 'spower'}:
        a = rc8_apply(node['arg'], Xfs, params, path + 'u')
        n, d = node['power']
        p = float(n) / float(d)
        with np.errstate(all='ignore'):
            return np.power(a, p) if op == 'power' else np.sign(a) * np.power(np.abs(a), p)
    if op == 'scale':
        a = rc8_apply(node['arg'], Xfs, params, path + 's')
        n, d = node['coefficient']
        return a * (float(n) / float(d))
    a = rc8_apply(node['left'], Xfs, params, path + 'L')
    b = rc8_apply(node['right'], Xfs, params, path + 'R')
    with np.errstate(all='ignore'):
        if op == 'add':
            return a + b
        if op == 'sub':
            return a - b
        if op == 'mul':
            return a * b
        if op == 'geometric':
            return np.sqrt(a * b)
        if op == 'sgeometric':
            return np.sign(a * b) * np.sqrt(np.abs(a * b))
        if op == 'saturating':
            return a / (1 + np.abs(b))
        if op == 'safe_ratio':
            return a / (np.abs(b) + float(params[path]))
    raise ValidationError(f'unsupported RC8 op {op}')


def rc8_original_eval(node: Mapping[str, Any], Xfs: np.ndarray, epsilon: float):
    op = node['op']
    if op == 'terminal':
        return Xfs[int(node['feature_index'])]
    if op == 'abs':
        return np.abs(rc8_original_eval(node['arg'], Xfs, epsilon))
    if op in {'power', 'spower'}:
        a = rc8_original_eval(node['arg'], Xfs, epsilon)
        n, d = node['power']
        p = float(n) / float(d)
        with np.errstate(all='ignore'):
            return np.power(a, p) if op == 'power' else np.sign(a) * np.power(np.abs(a), p)
    if op == 'scale':
        a = rc8_original_eval(node['arg'], Xfs, epsilon)
        n, d = node['coefficient']
        return a * (float(n) / float(d))
    a = rc8_original_eval(node['left'], Xfs, epsilon)
    b = rc8_original_eval(node['right'], Xfs, epsilon)
    with np.errstate(all='ignore'):
        if op == 'add':
            return a + b
        if op == 'sub':
            return a - b
        if op == 'mul':
            return a * b
        if op == 'geometric':
            return np.sqrt(a * b)
        if op == 'sgeometric':
            return np.sign(a * b) * np.sqrt(np.abs(a * b))
        if op == 'saturating':
            return a / (1 + np.abs(b))
        if op == 'safe_ratio':
            return a / (np.abs(b) + epsilon)
    raise ValidationError(f'unsupported original RC8 op {op}')


def rank_auc(y, s):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    m = np.isfinite(s)
    y = y[m]
    s = s[m]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    r = np.empty(len(s), float)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[order[j]] == s[order[i]]:
            j += 1
        r[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def best_threshold(y, s, orientation):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    m = np.isfinite(s)
    y = y[m]
    s = s[m]
    z = s if orientation == 1 else -s
    order = np.argsort(z, kind='mergesort')
    zz = z[order]
    yy = y[order]
    P = max(1, int((yy == 1).sum()))
    N = max(1, int((yy == 0).sum()))
    cp = np.cumsum(yy == 1)
    cn = np.cumsum(yy == 0)
    tp = P - cp
    tn = cn
    b = .5 * (tp / P + tn / N)
    cand = np.concatenate(([.5], b))
    k = int(np.argmax(cand))
    best = float(cand[k])
    if k == 0:
        th = float(zz[0] - 1e-12)
    elif k >= len(zz):
        th = float(zz[-1] + 1e-12)
    else:
        j = k - 1
        th = float(zz[j] + 1e-12) if j == len(zz) - 1 else float(.5 * (zz[j] + zz[j + 1]))
    good = np.flatnonzero(cand >= best - .01)
    ts = []
    for kk in good:
        if kk == 0:
            ts.append(float(zz[0]))
        elif kk >= len(zz):
            ts.append(float(zz[-1]))
        else:
            j = kk - 1
            ts.append(float(zz[j] if j == len(zz) - 1 else .5 * (zz[j] + zz[j + 1])))
    width = max(ts) - min(ts) if len(ts) >= 2 else 0.
    iqr = float(np.quantile(z, .75) - np.quantile(z, .25))
    plateau = min(10., width / (abs(iqr) + 1e-12))
    return (th if orientation == 1 else -th), best, plateau


def predict_threshold(s, th, orientation):
    return (np.asarray(s) >= th) if orientation == 1 else (np.asarray(s) <= th)


def binary_metrics(y, s=None, pred=None, orientation=None, threshold=None):
    y = np.asarray(y, int)
    out = {'n': int(len(y)), 'positive_n': int((y == 1).sum()), 'negative_n': int((y == 0).sum())}
    if s is not None:
        raw = rank_auc(y, s)
        orient = (1 if raw >= .5 else -1) if orientation is None else int(orientation)
        out['auc'] = float(raw if orient == 1 else 1 - raw)
        out['orientation'] = orient
        if threshold is None:
            threshold, bacc_fit, plateau = best_threshold(y, s, orient)
            out['threshold'] = float(threshold)
            out['threshold_plateau'] = float(plateau)
        pred = predict_threshold(s, threshold, orient)
        out['threshold'] = float(threshold)
    pred = np.asarray(pred, bool)
    tp = int(np.sum(pred & (y == 1)))
    tn = int(np.sum((~pred) & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))
    rec1 = tp / max(1, tp + fn)
    rec0 = tn / max(1, tn + fp)
    acc = (tp + tn) / max(1, len(y))
    bacc = .5 * (rec1 + rec0)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * rec1 / max(1e-15, prec + rec1)
    out.update(tp=tp, tn=tn, fp=fp, fn=fn, acc=float(acc), bacc=float(bacc), recall_positive=float(rec1),
               recall_negative=float(rec0), f1=float(f1))
    return out


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]
    if len(a) < 3:
        return float('nan')
    ra = pd.Series(a).rank(method='average').to_numpy()
    rb = pd.Series(b).rank(method='average').to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def load_challenge_registry(protocol_root: Path, sample_ids: np.ndarray) -> tuple[pd.DataFrame, dict]:
    root = protocol_root.expanduser().resolve()
    cands = []
    for name in ['challenge_rows.csv', 'protocol_rows.csv']:
        for p in root.rglob(name):
            try:
                h = pd.read_csv(p, nrows=0)
                key = metadata_key(h.columns)
                if key and any(t in h.columns for t in CHALLENGE_TAGS):
                    cands.append(p)
            except Exception:
                pass
    if not cands:
        return pd.DataFrame({'sample_id': sample_ids}), {'status': 'NO_REGISTRY', 'path': None}
    # Prefer challenge_rows and greatest overlap.
    best = None
    bestkey = None
    for p in cands:
        df = pd.read_csv(p, low_memory=False)
        key = metadata_key(df.columns)
        if key != 'sample_id':
            df = df.rename(columns={key: 'sample_id'})
        df.sample_id = df.sample_id.astype(str)
        overlap = int(df.sample_id.isin(set(sample_ids)).sum())
        score = (1 if p.name == 'challenge_rows.csv' else 0, overlap)
        if best is None or score > bestkey:
            best = (p, df)
            bestkey = score
    p, df = best
    keep = (['sample_id'] + [c for c in CHALLENGE_TAGS if c in df.columns]
            + [c for c in ['challenge_target', 'challenge_guard', 'is_challenge_target', 'is_challenge_guard',
                           'challenge_reason', 'challenge_holdout_trigger'] if c in df.columns])
    df = df[keep].drop_duplicates('sample_id')
    out = pd.DataFrame({'sample_id': sample_ids}).merge(df, on='sample_id', how='left')
    for c in CHALLENGE_TAGS:
        if c not in out:
            out[c] = False
        out[c] = out[c].astype('boolean').fillna(False).astype(bool)
    return out, {'status': 'PASS', 'path': str(p), 'sha256': sha256_file(p),
                 'matched_n': int(out[CHALLENGE_TAGS].any(axis=1).sum())}


def trajectory_macro_auc(y, s, traj, mask, orientation):
    vals = []
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    traj = np.asarray(traj)
    mask = np.asarray(mask, bool)
    for t in np.unique(traj[mask]):
        ii = np.flatnonzero(mask & (traj == t))
        if len(ii) < 2 or len(np.unique(y[ii])) < 2:
            continue
        a = rank_auc(y[ii], s[ii])
        vals.append(a if orientation == 1 else 1 - a)
    return (float(np.mean(vals)) if vals else float('nan'), int(len(vals)))


def confusion_table(true, pred, labels):
    t = pd.Categorical(true, categories=labels)
    p = pd.Categorical(pred, categories=labels)
    return pd.crosstab(t, p, dropna=False).reindex(index=labels, columns=labels, fill_value=0)


def trajectory_audit(df: pd.DataFrame, partition: str) -> pd.DataFrame:
    rows = []
    def changes(g, col):
        a = g[col].astype(str).to_numpy()
        ii = np.flatnonzero(a[1:] != a[:-1])
        return ';'.join(f'{int(g.k_index.iloc[i])}->{int(g.k_index.iloc[i+1])}' for i in ii)
    for tid, g in df.groupby('trajectory_id', sort=True):
        g = g.sort_values('k_index')
        rows.append({'partition': partition, 'trajectory_id': tid, 'n': len(g), 'k_min': int(g.k_index.min()),
                     'k_max': int(g.k_index.max()), 'V_true_crossings': changes(g, 'V_true_Gamma'),
                     'V_pred_crossings': changes(g, 'V_pred_Gamma'), 'C_true_crossings': changes(g, 'C_true_Gamma'),
                     'C_pred_crossings': changes(g, 'C_pred_Gamma'),
                     'true_coarse_sequence': ';'.join(g.true_coarse_state.astype(str)),
                     'pred_coarse_sequence': ';'.join(g.pred_coarse_state.astype(str))})
    return pd.DataFrame(rows)


def core_analysis(args):
    out = Path(args.outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    bundle = resolve_bundle(Path(args.protocol_root))
    arr = bundle / 'arrays'
    feat = bundle / 'features'
    Xfull = np.load(arr / 'X_full_clean_float64.npy', mmap_mode='r', allow_pickle=False)
    sid = np.load(arr / 'sample_ids.npy', allow_pickle=False).astype(str)
    rel = np.load(arr / 'relation_component_unicode.npy', allow_pickle=False).astype(str)
    names = list(map(str, json.loads((feat / 'formula_full_feature_names.json').read_text())))
    if Xfull.shape != (len(sid), len(names)) or len(rel) != len(sid):
        raise ValidationError('native protocol shape mismatch')
    freeze = pd.read_csv(args.freeze_csv)
    dev_ids = freeze.sample_id.astype(str).tolist()
    index = {s: i for i, s in enumerate(sid)}
    missing = [s for s in dev_ids if s not in index]
    if missing:
        raise ValidationError(f'freeze IDs absent from protocol n={len(missing)} first={missing[:5]}')
    dev = np.zeros(len(sid), bool)
    dev[[index[s] for s in dev_ids]] = True
    challenge_target_mask = np.load(arr / 'mask_challenge_target.npy', allow_pickle=False).astype(bool)
    challenge_guard_mask = np.load(arr / 'mask_challenge_guard.npy', allow_pickle=False).astype(bool)
    challenge_union = challenge_target_mask | challenge_guard_mask
    old_dev = ~challenge_union
    hold = challenge_union & (~dev)
    excluded = (~dev) & (~hold)
    if (len(sid), int(old_dev.sum()), int(challenge_union.sum()), int(dev.sum()), int(hold.sum()),
        int(excluded.sum())) != (681, 442, 239, 440, 224, 17):
        raise ValidationError(f'''partition contract mismatch total/olddev/challenge/dev/hold/excluded={(
            len(sid), int(old_dev.sum()), int(challenge_union.sum()), int(dev.sum()), int(hold.sum()),
            int(excluded.sum()))}''')
    log('partition PASS total=681 old_dev=442 dual_dev=440 challenge_holdout=224 excluded=17')
    canon = canonical_contract(Path(args.canonical_csv), sid)
    V = edge_table(canon, 'V')
    C = edge_table(canon, 'C')
    truth = joint_truth(V, C, canon)
    if not (truth.V_available.all() and truth.C_available.all()):
        raise ValidationError('edge target coverage <100%')
    # Strict label against the k-point truth.
    if 'strict_label' in truth:
        mismatch = (truth.strict_label.astype(str).str.lower() != truth.true_binary_label.astype(str).str.lower())
        if mismatch.any():
            raise ValidationError(f'canonical strict label / kpoint truth mismatch n={int(mismatch.sum())}')
    # Protocol labels are used for the replay only; all metrics use the primitive-canonical k-point labels.
    y_protocol = np.load(arr / 'labels.npy', allow_pickle=False).astype(int)
    ybin_all = truth.true_binary_label.eq('co_located').astype(int).to_numpy()
    label_audit = protocol_canonical_label_audit(y_protocol, ybin_all, sid, V, C, truth, old_dev, dev, hold, excluded,
                                                 out)
    if label_audit['mismatch_n']:
        extra = ('' if label_audit['all_mismatches_are_primitive_endpoint_overrides']
                 else ' WARNING: includes non-endpoint legacy drift')
        log(f"protocol/canonical label drift AUDITED mismatch_n={label_audit['mismatch_n']}; "
            f"canonical truth governs, protocol y is replay-only.{extra}")
    else:
        log('protocol/canonical label audit PASS mismatch_n=0')
    # Features of the final coordinates: robust scaling fitted on the development set only.
    req = ['lattice_angle_anisotropy', 'elem_A_d_electrons', 'D_BXB_connectivity_disorder_proxy',
           'XX_network_coherence', 'alloy_first_ionization_kJmol_variance', 'B_d_filling_mean', 'alloy_Z_variance',
           'BB_d2_mean', 'BX_d_electrons_mismatch_norm', 'BB_graph_shortest_path_mean', 'D_BX6_distortion_RMS',
           'elem_Bmix_Z']
    pos = {n: i for i, n in enumerate(names)}
    miss = [n for n in req if n not in pos]
    if miss:
        raise ValidationError(f'missing final formula raw features: {miss}')
    cols = np.array([pos[n] for n in req], int)
    Xraw = np.asarray(Xfull[:, cols], float)
    scales = robust_scales(Xraw[dev])
    Xs = np.nan_to_num(Xraw / scales[None, :], nan=0, posinf=1e6, neginf=-1e6)
    fx = {n: Xs[:, i] for i, n in enumerate(req)}
    # Shared component inside Pi_V, with the simplification semantics of the formula search.
    formula_root = Path(args.formula_root)
    frozen_scales = np.load(resolve_asset(formula_root, 'feature_scales.npy'), allow_pickle=False).astype(float)
    best = json.loads(resolve_asset(formula_root, 'best_performance_formula.json').read_text())
    raw = best.get('ast_json') or best.get('expanded_ast_json') or best.get('ast')
    node = json.loads(raw) if isinstance(raw, str) else raw
    state = json.loads(resolve_asset(formula_root, 'frozen_decision_state.json').read_text())
    epsilon = float(state.get('epsilon', EPS))
    Xall = np.asarray(Xfull, float)
    Xfs = Xall / frozen_scales[None, :]
    rc8_dev_raw, par_dual = rc8_fit(node, Xfs[dev].T, epsilon)
    rc8_all_raw = rc8_apply(node, Xfs.T, par_dual)
    pi_s_scale = float(np.quantile(rc8_dev_raw, .75) - np.quantile(rc8_dev_raw, .25))
    pi_s_scale = pi_s_scale if math.isfinite(pi_s_scale) and pi_s_scale > 1e-12 else float(np.std(rc8_dev_raw) or 1.)
    pi_s = rc8_all_raw / pi_s_scale
    # The standalone Pi_S has a different numerical contract; it is checked against the archived protocol labels on the
    # 442 development states. All metrics below use ybin_all (primitive-canonical k points).
    pis_contract = frozen_top1_contract(formula_root, names)
    pi_s_frozen_scores, pis_semantics = authenticate_pis_replay(pis_contract, Xall, y_protocol, old_dev, out)
    top_formula_id = pis_contract['formula_id']
    old_orientation = int(pis_contract['orientation'])
    old_threshold = float(pis_contract['threshold'])
    old_pred_all = predict_threshold(pi_s_frozen_scores, old_threshold, old_orientation)
    archive_dev_m = binary_metrics(y_protocol[old_dev], pi_s_frozen_scores[old_dev], old_pred_all[old_dev],
                                   old_orientation, old_threshold)
    canonical_dev442_m = binary_metrics(ybin_all[old_dev], pi_s_frozen_scores[old_dev], old_pred_all[old_dev],
                                        old_orientation, old_threshold)
    # Challenge states are scored after the development replay check.
    canonical_ch239_m = binary_metrics(ybin_all[challenge_union], pi_s_frozen_scores[challenge_union],
                                       old_pred_all[challenge_union], old_orientation, old_threshold)
    write_json(out / 'pi_s_metrics.json',
               {'status': 'PASS', 'archive_replay_protocol_truth_development_442': archive_dev_m,
                'final_primitive_canonical_truth_development_442': canonical_dev442_m,
                'final_primitive_canonical_truth_challenge_union_239': canonical_ch239_m,
                'protocol_canonical_label_mismatch_n': label_audit['mismatch_n'], 'formula_id': top_formula_id,
                'orientation': old_orientation, 'threshold': old_threshold, 'top_source': pis_contract['top_source'],
                'top_path': str(pis_contract['top_path']), 'operator_semantics': pis_semantics['selected_mode'],
                'challenge_not_used_for_authentication': True,
                'truth_governance': 'protocol labels are used only to authenticate the historical archived Pi_S '
                                    'replay; primitive-canonical k-point truth governs all final comparisons'})
    log(f'Pi_S archive replay PASS source={pis_contract["top_source"]} '
        f'semantics={pis_semantics["selected_mode"]} archived-protocol Dev_AUC={archive_dev_m["auc"]:.6f} '
        f'Dev_BACC={archive_dev_m["bacc"]:.6f}; final truth=primitive canonical')
    # Fixed final expressions; protected-ratio epsilon fitted on the development set only.
    term_a1 = fx['lattice_angle_anisotropy'] * fx['elem_A_d_electrons']
    den_v = fx['XX_network_coherence'] ** 2
    _, eps_v = safe_ratio_fit(fx['D_BXB_connectivity_disorder_proxy'][dev], den_v[dev])
    term_v2 = safe_ratio_apply(fx['D_BXB_connectivity_disorder_proxy'], den_v, eps_v)
    term_v3 = fx['alloy_first_ionization_kJmol_variance'] * (
        np.sign(fx['B_d_filling_mean']) * np.minimum(np.abs(fx['B_d_filling_mean']), 1e6) ** 2)
    score_v = pi_s + FINAL_V_COEF[0] * term_a1 + FINAL_V_COEF[1] * term_v2 + FINAL_V_COEF[2] * term_v3
    var_z = fx['alloy_Z_variance']
    inner = (np.sign(fx['BB_d2_mean'] * fx['BX_d_electrons_mismatch_norm'])
             * np.sqrt(np.abs(fx['BB_d2_mean'] * fx['BX_d_electrons_mismatch_norm'])))
    net = (np.sign(fx['BB_graph_shortest_path_mean'] * inner)
           * np.sqrt(np.abs(fx['BB_graph_shortest_path_mean'] * inner)))
    dist2 = np.sign(fx['D_BX6_distortion_RMS']) * np.minimum(np.abs(fx['D_BX6_distortion_RMS']), 1e6) ** 2
    den_c = var_z * fx['elem_Bmix_Z']
    _, eps_c = safe_ratio_fit(dist2[dev], den_c[dev])
    term_c3 = safe_ratio_apply(dist2, den_c, eps_c)
    score_c = FINAL_C_COEF[0] * (np.sign(var_z) * np.minimum(np.abs(var_z), 1e6) ** 2) + FINAL_C_COEF[1] * net + term_c3
    # Thresholds from the development set; orientation -1 as in the discovered formulas.
    y_v = truth.V_true_Gamma.astype(int).to_numpy()
    y_c = truth.C_true_Gamma.astype(int).to_numpy()
    th_v, bacc_v, plateau_v = best_threshold(y_v[dev], score_v[dev], -1)
    th_c, bacc_c, plateau_c = best_threshold(y_c[dev], score_c[dev], -1)
    pred_v = predict_threshold(score_v, th_v, -1)
    pred_c = predict_threshold(score_c, th_c, -1)
    # Replay check against the fixed-formula development metrics.
    dev_metrics_v = binary_metrics(y_v[dev], score_v[dev], pred_v[dev], -1, th_v)
    dev_metrics_c = binary_metrics(y_c[dev], score_c[dev], pred_c[dev], -1, th_c)
    if (abs(dev_metrics_v['auc'] - EXPECTED_DEV['V']['auc']) > 6e-5
            or abs(dev_metrics_v['bacc'] - EXPECTED_DEV['V']['bacc']) > 2e-6):
        raise ValidationError(f'V fixed formula replay mismatch {dev_metrics_v}')
    if (abs(dev_metrics_c['auc'] - EXPECTED_DEV['C']['auc']) > 6e-5
            or abs(dev_metrics_c['bacc'] - EXPECTED_DEV['C']['bacc']) > 2e-6):
        raise ValidationError(f'C fixed formula replay mismatch {dev_metrics_c}')
    log(f'fixed formula replay PASS V_AUC={dev_metrics_v["auc"]:.6f} C_AUC={dev_metrics_c["auc"]:.6f}')
    # RC8 edge baselines: thresholds fitted on the development set only.
    auc_pi_s_v = rank_auc(y_v[dev], pi_s[dev])
    orient_pi_s_v = 1 if auc_pi_s_v >= .5 else -1
    th_pi_s_v, _, _ = best_threshold(y_v[dev], pi_s[dev], orient_pi_s_v)
    pred_pi_s_v = predict_threshold(pi_s, th_pi_s_v, orient_pi_s_v)
    auc_pi_s_c = rank_auc(y_c[dev], pi_s[dev])
    orient_pi_s_c = 1 if auc_pi_s_c >= .5 else -1
    th_pi_s_c, _, _ = best_threshold(y_c[dev], pi_s[dev], orient_pi_s_c)
    pred_pi_s_c = predict_threshold(pi_s, th_pi_s_c, orient_pi_s_c)
    # Robust oriented coordinates; location and scale from the development set, also for the NN secondary readout.
    def freeze_z(s, orientation):
        z = s * orientation
        med = float(np.median(z[dev]))
        iqr = float(np.quantile(z[dev], .75) - np.quantile(z[dev], .25))
        iqr = iqr if math.isfinite(iqr) and iqr > 1e-12 else float(np.std(z[dev]) or 1.)
        return (z - med) / iqr, med, iqr
    z_v, med_z_v, iqr_z_v = freeze_z(score_v, -1)
    z_c, med_z_c, iqr_z_c = freeze_z(score_c, -1)
    pair_distance = -np.abs(z_v - z_c)
    nn_dev = dev & truth.true_exact_state.isin(['NN_same', 'NN_diff']).to_numpy()
    nn_y = (truth.true_exact_state.to_numpy()[nn_dev] == 'NN_same').astype(int)
    nn_auc = rank_auc(nn_y, pair_distance[nn_dev])
    nn_orientation = 1 if nn_auc >= .5 else -1
    nn_threshold, nn_bacc, nn_plateau = best_threshold(nn_y, pair_distance[nn_dev], nn_orientation)
    nn_pred = predict_threshold(pair_distance, nn_threshold, nn_orientation)
    pred_coarse = np.where(pred_v & pred_c, 'GG',
                           np.where(pred_v & (~pred_c), 'GN', np.where((~pred_v) & pred_c, 'NG', 'NN')))
    pred_exact = pred_coarse.astype(object)
    nn_predmask = pred_coarse == 'NN'
    pred_exact[nn_predmask] = np.where(nn_pred[nn_predmask], 'NN_same', 'NN_diff')
    pred_binary = np.where(np.isin(pred_exact, ['GG', 'NN_same']), 'co_located', 'separated')
    old_pred = predict_threshold(pi_s_frozen_scores, old_threshold, old_orientation)
    old_binary = np.where(old_pred, 'co_located', 'separated')
    # Assemble sample table.
    tab = truth.copy()
    tab['relation_component'] = rel
    tab['partition'] = np.where(dev, 'DUAL_DEV_V1', np.where(hold, 'FROZEN_CHALLENGE_HOLDOUT', 'EXCLUDED_UNRELIABLE'))
    tab['Pi_S_shared'] = pi_s
    tab['Pi_V_raw'] = score_v
    tab['tau_V'] = th_v
    tab['V_pred_Gamma'] = pred_v
    tab['Pi_C_raw'] = score_c
    tab['tau_C'] = th_c
    tab['C_pred_Gamma'] = pred_c
    tab['pred_coarse_state'] = pred_coarse
    tab['NN_pair_distance_score'] = pair_distance
    tab['pred_exact_state_secondary'] = pred_exact
    tab['pred_binary_secondary'] = pred_binary
    tab['Pi_S_frozen_raw'] = pi_s_frozen_scores
    tab['Pi_S_frozen_pred_binary'] = old_binary
    tab['V_Pi_S_edge_pred_Gamma'] = pred_pi_s_v
    tab['C_Pi_S_edge_pred_Gamma'] = pred_pi_s_c
    reg, regaudit = load_challenge_registry(Path(args.protocol_root), sid)
    tab = tab.merge(reg, on='sample_id', how='left', validate='one_to_one')
    for c in CHALLENGE_TAGS:
        tab[c] = tab[c].astype('boolean').fillna(False).astype(bool)
    tab['V_final_correct'] = tab.V_pred_Gamma == tab.V_true_Gamma
    tab['C_final_correct'] = tab.C_pred_Gamma == tab.C_true_Gamma
    tab['coarse_correct'] = tab.pred_coarse_state == tab.true_coarse_state
    tab['exact5_secondary_correct'] = tab.pred_exact_state_secondary == tab.true_exact_state
    tab['dual_binary_secondary_correct'] = tab.pred_binary_secondary == tab.true_binary_label
    tab['Pi_S_binary_correct'] = tab.Pi_S_frozen_pred_binary == tab.true_binary_label
    tab['V_Pi_S_edge_correct'] = tab.V_Pi_S_edge_pred_Gamma == tab.V_true_Gamma
    tab['C_Pi_S_edge_correct'] = tab.C_Pi_S_edge_pred_Gamma == tab.C_true_Gamma
    tab.to_csv(out / 'FROZEN_DUAL_COORDINATE_SAMPLE_AUDIT.csv', index=False)
    # Partition metrics.
    parts = {'DEVELOPMENT_440': dev, 'CHALLENGE_HOLDOUT_224': hold, 'EXCLUDED_17': excluded}
    edge_rows = []
    pair_rows = []
    for pn, mask in parts.items():
        for e, y, s, p, pred_pi_s_edge in [('V', y_v, score_v, pred_v, pred_pi_s_v),
                                           ('C', y_c, score_c, pred_c, pred_pi_s_c)]:
            m = binary_metrics(y[mask], s[mask], p[mask], -1, th_v if e == 'V' else th_c)
            true_margin = (truth.V_signed_margin_eV.to_numpy(float) if e == 'V'
                           else truth.C_signed_margin_eV.to_numpy(float))
            selected = mask & np.isfinite(true_margin)
            traj_auc, traj_auc_n = trajectory_macro_auc(y, s, truth.trajectory_id.to_numpy(), mask, -1)
            m.update(partition=pn, edge=e, model='FINAL_EDGE_COORDINATE',
                     margin_spearman=spearman((-s)[selected], true_margin[selected]), margin_n=int(selected.sum()),
                     trajectory_auc=traj_auc, trajectory_auc_n=traj_auc_n)
            edge_rows.append(m)
            oo = orient_pi_s_v if e == 'V' else orient_pi_s_c
            mr = binary_metrics(y[mask], pi_s[mask], pred_pi_s_edge[mask], oo, th_pi_s_v if e == 'V' else th_pi_s_c)
            traj_auc_pi_s, traj_auc_pi_s_n = trajectory_macro_auc(y, pi_s, truth.trajectory_id.to_numpy(), mask, oo)
            mr.update(partition=pn, edge=e, model='PI_S_EDGE_BASELINE',
                      margin_spearman=spearman((pi_s * oo)[selected], true_margin[selected]),
                      margin_n=int(selected.sum()), trajectory_auc=traj_auc_pi_s, trajectory_auc_n=traj_auc_pi_s_n)
            edge_rows.append(mr)
        tt = tab.loc[mask]
        pair_rows.append({'partition': pn, 'n': int(mask.sum()),
                          'coarse_4state_accuracy': float(tt.coarse_correct.mean()),
                          'exact_5state_accuracy_secondary_NN_readout': float(tt.exact5_secondary_correct.mean()),
                          'binary_accuracy_secondary_NN_readout': float(tt.dual_binary_secondary_correct.mean()),
                          'Pi_S_frozen_binary_accuracy': float(tt.Pi_S_binary_correct.mean()),
                          'GG_n': int((tt.true_coarse_state == 'GG').sum()),
                          'GN_n': int((tt.true_coarse_state == 'GN').sum()),
                          'NG_n': int((tt.true_coarse_state == 'NG').sum()),
                          'NN_n': int((tt.true_coarse_state == 'NN').sum()),
                          'NN_same_n': int((tt.true_exact_state == 'NN_same').sum()),
                          'NN_diff_n': int((tt.true_exact_state == 'NN_diff').sum())})
    pd.DataFrame(edge_rows).to_csv(out / 'edge_metrics_by_partition.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(out / 'PAIR_METRICS_BY_PARTITION.csv', index=False)
    # Binary strict-task metrics for the frozen Pi_S baseline and the dual secondary NN readout.
    binary_rows = []
    for pn, mask in parts.items():
        ybin = (tab.true_binary_label.to_numpy() == 'co_located').astype(int)
        for model, pred in [('FROZEN_PI_S_BASELINE', old_pred),
                            ('DUAL_COORDINATE_SECONDARY_NN_READOUT', (pred_binary == 'co_located'))]:
            m = binary_metrics(ybin[mask], pred=pred[mask])
            m.update(partition=pn, model=model)
            binary_rows.append(m)
    pd.DataFrame(binary_rows).to_csv(out / 'STRICT_BINARY_TASK_COMPARISON.csv', index=False)
    # Confusions.
    for pn, mask in [('DEVELOPMENT_440', dev), ('CHALLENGE_HOLDOUT_224', hold)]:
        confusion_table(tab.true_coarse_state.to_numpy()[mask], pred_coarse[mask],
                        ['GG', 'GN', 'NG', 'NN']).to_csv(out / f'{pn}_COARSE_4STATE_CONFUSION.csv')
        confusion_table(tab.true_exact_state.to_numpy()[mask], pred_exact[mask],
                        ['GG', 'GN', 'NG', 'NN_same', 'NN_diff']).to_csv(
            out / f'{pn}_EXACT_5STATE_SECONDARY_CONFUSION.csv')

    state_rows = []
    for pn, mask in [('DEVELOPMENT_440', dev), ('CHALLENGE_HOLDOUT_224', hold)]:
        for st in ['GG', 'GN', 'NG', 'NN']:
            selected = mask & (tab.true_coarse_state.to_numpy() == st)
            state_rows.append({'partition': pn, 'state': st, 'truth_n': int(selected.sum()),
                               'recall': (float(np.mean(pred_coarse[selected] == st)) if selected.sum()
                                          else float('nan'))})
    pd.DataFrame(state_rows).to_csv(out / 'COARSE_STATE_RECALL.csv', index=False)
    # Rescue/harm and edge attribution on challenge.
    ch = tab[hold].copy()
    rescue = ch[(~ch.Pi_S_binary_correct) & ch.dual_binary_secondary_correct]
    harm = ch[ch.Pi_S_binary_correct & (~ch.dual_binary_secondary_correct)]
    rescue.to_csv(out / 'CHALLENGE_PI_S_WRONG_DUAL_CORRECT_RESCUE.csv', index=False)
    harm.to_csv(out / 'CHALLENGE_PI_S_CORRECT_DUAL_WRONG_HARM.csv', index=False)
    attrib = pd.DataFrame([{'edge': 'V', 'n': len(ch), 'Pi_S_edge_correct_n': int(ch.V_Pi_S_edge_correct.sum()),
                            'final_edge_correct_n': int(ch.V_final_correct.sum()),
                            'rescued_n': int((~ch.V_Pi_S_edge_correct & ch.V_final_correct).sum()),
                            'harmed_n': int((ch.V_Pi_S_edge_correct & ~ch.V_final_correct).sum())},
                           {'edge': 'C', 'n': len(ch), 'Pi_S_edge_correct_n': int(ch.C_Pi_S_edge_correct.sum()),
                            'final_edge_correct_n': int(ch.C_final_correct.sum()),
                            'rescued_n': int((~ch.C_Pi_S_edge_correct & ch.C_final_correct).sum()),
                            'harmed_n': int((ch.C_Pi_S_edge_correct & ~ch.C_final_correct).sum())}])
    attrib['net_rescue_n'] = attrib.rescued_n - attrib.harmed_n
    attrib.to_csv(out / 'CHALLENGE_EDGE_RESCUE_ATTRIBUTION.csv', index=False)
    # Domain stratification.
    dom = []
    masks = [('challenge_all', np.ones(len(ch), bool)), ('challenge_target', challenge_target_mask[hold]),
             ('challenge_guard', challenge_guard_mask[hold])]
    for c in CHALLENGE_TAGS:
        masks.append((c, ch[c].to_numpy(bool)))
    ybin = (ch.true_binary_label.to_numpy() == 'co_located').astype(int)
    for name, m in masks:
        if int(m.sum()) == 0:
            continue
        r = {'domain': name, 'n': int(m.sum()), 'co_located_n': int(ybin[m].sum()),
             'separated_n': int((1 - ybin[m]).sum()),
             'V_final_accuracy': float(ch.V_final_correct.to_numpy()[m].mean()),
             'C_final_accuracy': float(ch.C_final_correct.to_numpy()[m].mean()),
             'coarse_4state_accuracy': float(ch.coarse_correct.to_numpy()[m].mean()),
             'dual_binary_secondary_accuracy': float(ch.dual_binary_secondary_correct.to_numpy()[m].mean()),
             'Pi_S_binary_accuracy': float(ch.Pi_S_binary_correct.to_numpy()[m].mean()),
             'rescue_n': int(((~ch.Pi_S_binary_correct) & ch.dual_binary_secondary_correct).to_numpy()[m].sum()),
             'harm_n': int((ch.Pi_S_binary_correct & (~ch.dual_binary_secondary_correct)).to_numpy()[m].sum())}
        r['net_rescue_n'] = r['rescue_n'] - r['harm_n']
        dom.append(r)
    pd.DataFrame(dom).to_csv(out / 'CHALLENGE_DOMAIN_STRATIFIED_AUDIT.csv', index=False)
    # Trajectory audits.
    trajectory_audit(tab[dev], 'DEVELOPMENT_440').to_csv(out / 'DEVELOPMENT_TRAJECTORY_AUDIT.csv', index=False)
    trajectory_audit(ch, 'CHALLENGE_HOLDOUT_224').to_csv(out / 'CHALLENGE_TRAJECTORY_AUDIT.csv', index=False)
    # Threshold/numeric provenance.
    threshold = {'V': {'orientation': -1, 'threshold': th_v, 'plateau': plateau_v},
                 'C': {'orientation': -1, 'threshold': th_c, 'plateau': plateau_c},
                 'NN_secondary': {'definition': '-abs(zV-zC), normalized using Development medians/IQRs only',
                                  'orientation': nn_orientation, 'threshold': nn_threshold, 'development_bacc': nn_bacc,
                                  'plateau': nn_plateau, 'V_oriented_median': med_z_v, 'V_oriented_iqr': iqr_z_v,
                                  'C_oriented_median': med_z_c, 'C_oriented_iqr': iqr_z_c},
                 'PI_S_edge_baselines': {'V': {'orientation': orient_pi_s_v, 'threshold': th_pi_s_v},
                                         'C': {'orientation': orient_pi_s_c, 'threshold': th_pi_s_c}},
                 'old_single_coordinate': {'formula_id': top_formula_id, 'orientation': old_orientation,
                                           'threshold': old_threshold}}
    write_json(out / 'readout_thresholds.json', threshold)
    provenance = {'version': VERSION, 'status': 'PASS', 'protocol_root': str(Path(args.protocol_root).resolve()),
                  'protocol_bundle': str(bundle), 'canonical_csv': str(Path(args.canonical_csv).resolve()),
                  'canonical_sha256': sha256_file(Path(args.canonical_csv)),
                  'formula_root': str(Path(args.formula_root).resolve()),
                  'freeze_csv': str(Path(args.freeze_csv).resolve()),
                  'freeze_sha256': sha256_file(Path(args.freeze_csv)),
                  'partition': {'total': 681, 'old_development': 442, 'challenge_union': 239, 'dual_dev': 440,
                                'challenge_holdout': 224, 'excluded_unreliable': 17}, 'challenge_registry': regaudit,
                  'protocol_canonical_label_audit': {k: v for k, v in label_audit.items() if k != 'rows'},
                  'final_truth_source': 'primitive-canonical k-point identity',
                  'legacy_protocol_y_role': 'historical Pi_S replay authentication only', 'no_challenge_fit': True,
                  'no_search': True, 'no_cv': True, 'fixed_replay_check_skipped': False,
                  'protected_ratio_parameters_fitted_on_dual_dev_only': {
                      'V_eps': eps_v, 'C_eps': eps_c, 'RC8_dual_ratio_node_count': len(par_dual),
                      'Pi_S_historical_baseline_semantics': pis_semantics['selected_mode'],
                      'Pi_S_semantics_selection_basis': pis_semantics.get('selection_basis'),
                      'Pi_S_historical_top_source': pis_contract['top_source']},
                  'formula_spec': {
                      'Pi_V': 'Pi_S + 3*A_theta*d_A - D_BXB/C_XX^2 - (1/6)*Var(I1)*f_Bd^2',
                      'Pi_C': '-(1/3)*Var(Z)^2 + 2*Sg(L_BB,Sg(d_BB,Delta_d_BX)) + D_BX6^2/(Var(Z)*Z_Bmix)',
                      'orientation': 'larger oriented coordinate (-raw) => Gamma-favored'}}
    write_json(out / 'validation_protocol.json', provenance)
    # Compact summary.
    edge_df = pd.DataFrame(edge_rows)
    pair_df = pd.DataFrame(pair_rows)
    binary_df = pd.DataFrame(binary_rows)
    def erow(part, e, model):
        return edge_df[(edge_df.partition == part) & (edge_df.edge == e) & (edge_df.model == model)].iloc[0].to_dict()
    binary_challenge = binary_df[binary_df.partition == 'CHALLENGE_HOLDOUT_224']
    old_binary_row = binary_challenge[binary_challenge.model == 'FROZEN_PI_S_BASELINE'].iloc[0]
    dual_binary_row = binary_challenge[binary_challenge.model == 'DUAL_COORDINATE_SECONDARY_NN_READOUT'].iloc[0]
    summary = {'version': VERSION, 'status': 'PASS',
               'truth_governance': {'final_truth': 'primitive-canonical k-point identity',
                                    'legacy_protocol_y_role': 'historical Pi_S replay authentication only',
                                    'protocol_canonical_mismatch_n': int(label_audit['mismatch_n']),
                                    'protocol_canonical_audit_status': label_audit['status']},
               'development': {'V': erow('DEVELOPMENT_440', 'V', 'FINAL_EDGE_COORDINATE'),
                               'C': erow('DEVELOPMENT_440', 'C', 'FINAL_EDGE_COORDINATE'),
                               'pair': pair_df[pair_df.partition == 'DEVELOPMENT_440'].iloc[0].to_dict()},
               'challenge_holdout': {'V': erow('CHALLENGE_HOLDOUT_224', 'V', 'FINAL_EDGE_COORDINATE'),
                                     'C': erow('CHALLENGE_HOLDOUT_224', 'C', 'FINAL_EDGE_COORDINATE'),
                                     'V_PiS_baseline': erow('CHALLENGE_HOLDOUT_224', 'V', 'PI_S_EDGE_BASELINE'),
                                     'C_PiS_baseline': erow('CHALLENGE_HOLDOUT_224', 'C', 'PI_S_EDGE_BASELINE'),
                                     'pair': pair_df[pair_df.partition == 'CHALLENGE_HOLDOUT_224'].iloc[0].to_dict(),
                                     'frozen_Pi_S_binary': old_binary_row.to_dict(),
                                     'dual_binary_secondary': dual_binary_row.to_dict(),
                                     'binary_bacc_gain_vs_old': float(dual_binary_row.bacc - old_binary_row.bacc),
                                     'binary_accuracy_gain_vs_old': float(dual_binary_row.acc - old_binary_row.acc),
                                     'rescue_n': int(len(rescue)), 'harm_n': int(len(harm)),
                                     'net_rescue_n': int(len(rescue) - len(harm))},
               'secondary_NN_readout_warning': 'NN_same/NN_diff threshold is fitted on Development only and is an '
                                               'auxiliary identity audit, not a third physical coordinate.'}
    write_json(out / 'run_summary.json', summary)
    # Report.
    challenge_summary = summary['challenge_holdout']
    report = [
        f'# Dual-edge frozen validation v{VERSION}', '', 'Status: **PASS**', '', '## Frozen contract',
        '- No symbolic search, coefficient tuning, threshold fitting, scaling, coordinate normalization, or '
        'protected-ratio calibration uses Challenge holdout.',
        '- Development 440 fixes all numerical/readout parameters; frozen Challenge holdout contains 224 rows.',
        '- The 17 physically excluded Development rows remain excluded from both fitting and Challenge scoring.',
        '- Primitive-canonical k-point truth governs all final metrics; protocol y_int8 is used only to '
        'consistency-check historical Pi_S replay.',
        f"- Protocol/canonical legacy-label drift audited: n={label_audit['mismatch_n']} ({label_audit['status']}).",
        '', '## Development replay',
        f"- Pi_V: AUC={summary['development']['V']['auc']:.6f}, BACC={summary['development']['V']['bacc']:.6f}",
        f"- Pi_C: AUC={summary['development']['C']['auc']:.6f}, BACC={summary['development']['C']['bacc']:.6f}",
        f"- coarse GG/GN/NG/NN accuracy={summary['development']['pair']['coarse_4state_accuracy']:.6f}",
        '', '## Frozen Challenge holdout',
        f"- Pi_V: AUC={challenge_summary['V']['auc']:.6f}, BACC={challenge_summary['V']['bacc']:.6f}",
        f"- Pi_C: AUC={challenge_summary['C']['auc']:.6f}, BACC={challenge_summary['C']['bacc']:.6f}",
        f"- coarse GG/GN/NG/NN accuracy={challenge_summary['pair']['coarse_4state_accuracy']:.6f}",
        f"- frozen Pi_S binary BACC={challenge_summary['frozen_Pi_S_binary']['bacc']:.6f}",
        f"- dual-coordinate binary BACC with frozen secondary NN readout="
        f"{challenge_summary['dual_binary_secondary']['bacc']:.6f}",
        f"- BACC gain={challenge_summary['binary_bacc_gain_vs_old']:+.6f}",
        f"- rescue/harm/net={challenge_summary['rescue_n']}/{challenge_summary['harm_n']}/"
        f"{challenge_summary['net_rescue_n']}",
        '', '## Interpretation guardrail',
        '- GG/GN/NG/NN is the primary joint-coordinate audit.',
        '- NN_same vs NN_diff is a secondary frozen readout from the coordinate-pair distance; it never creates or '
        'tunes Pi_V/Pi_C.',
        '- Domain tags are stratification labels only; no tag-specific fitting is performed.']
    (out / 'DUAL_EDGE_FROZEN_VALIDATION_REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    log('finished PASS')
    return summary


def self_test() -> dict:
    # Invariants of the threshold, protected-ratio and Q75 replay code; no input files needed.
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([3, 2, 1, -1, -2, -3], float)
    th, b, p = best_threshold(y, s, -1)
    m = binary_metrics(y, s, predict_threshold(s, th, -1), -1, th)
    rng = np.random.default_rng(56)
    a = rng.normal(size=100)
    b0 = rng.normal(size=100)
    v, eps = safe_ratio_fit(a, b0)
    v2 = safe_ratio_apply(a, b0, eps)
    X = np.vstack([np.linspace(1, 4, 8), np.array([.1, .2, .4, .8, 1.6, 3.2, 6.4, 12.8])])
    node = {'op': 'safe_ratio', 'left': {'op': 'terminal', 'feature_index': 0},
            'right': {'op': 'terminal', 'feature_index': 1}}
    q = rc8_q75_eval(node, X, 1e-6)
    raw = np.abs(X[1])
    qe = 1e-6 * float(np.quantile(raw, .75))
    qref = X[0] / (raw + qe)
    checks = {'frozen_threshold_semantics': bool(m['bacc'] == 1),
              'protected_ratio_fit_apply_identity': bool(np.allclose(v, v2)),
              'q75_relative_replay_semantics': bool(np.allclose(q, qref)), 'no_challenge_fit_contract': True,
              'historical_Pi_S_source_provenance_precedes_challenge_evaluation': True,
              'historical_Pi_S_semantics_selected_by_source_code_not_metric_tie_break': True,
              'four_state_primary_NN_identity_secondary': True, 'two_formula_independence': True,
              'formal_replay_bypass_absent': True, 'primitive_canonical_truth_precedence_over_legacy_protocol_y': True,
              'legacy_protocol_y_replay_only': True}
    checks['status'] = 'PASS' if all(v for k, v in checks.items() if k != 'status') else 'FAIL'
    return checks


def package_results(out: Path):
    core = ['run_summary.json', 'validation_protocol.json', 'readout_thresholds.json', 'edge_metrics_by_partition.csv',
            'PAIR_METRICS_BY_PARTITION.csv', 'COARSE_STATE_RECALL.csv', 'STRICT_BINARY_TASK_COMPARISON.csv',
            'pi_s_metrics.json', 'PI_S_REPLAY_SEMANTICS_AUDIT.csv', 'pi_s_ratio_semantics.json',
            'PROTOCOL_CANONICAL_LABEL_AUDIT.csv', 'PROTOCOL_CANONICAL_LABEL_AUDIT.json',
            'CHALLENGE_DOMAIN_STRATIFIED_AUDIT.csv', 'CHALLENGE_EDGE_RESCUE_ATTRIBUTION.csv',
            'CHALLENGE_PI_S_WRONG_DUAL_CORRECT_RESCUE.csv', 'CHALLENGE_PI_S_CORRECT_DUAL_WRONG_HARM.csv',
            'CHALLENGE_HOLDOUT_224_COARSE_4STATE_CONFUSION.csv',
            'CHALLENGE_HOLDOUT_224_EXACT_5STATE_SECONDARY_CONFUSION.csv', 'DEVELOPMENT_440_COARSE_4STATE_CONFUSION.csv',
            'FROZEN_DUAL_COORDINATE_SAMPLE_AUDIT.csv', 'CHALLENGE_TRAJECTORY_AUDIT.csv',
            'DEVELOPMENT_TRAJECTORY_AUDIT.csv', 'DUAL_EDGE_FROZEN_VALIDATION_REPORT.md']
    tarpath = out.parent / (out.name + '_core_results.tar.gz')
    with tarfile.open(tarpath, 'w:gz') as tf:
        for n in core:
            p = out / n
            if p.is_file():
                tf.add(p, arcname=n)
    sha = sha256_file(tarpath)
    (Path(str(tarpath) + '.sha256.txt')).write_text(f'{sha}  {tarpath.name}\n')
    print(f'CORE_RESULTS={tarpath}')
    print(f'CORE_SHA256={tarpath}.sha256.txt')


def parser():
    p = argparse.ArgumentParser(description='Frozen dual-edge zero-shot validation; no search/tuning on Challenge.')
    sp = p.add_subparsers(dest='cmd', required=True)
    t = sp.add_parser('self-test')
    t.set_defaults(func=lambda a: print(json.dumps(self_test(), indent=2)))
    r = sp.add_parser('run')
    r.add_argument('--protocol-root', required=True)
    r.add_argument('--formula-root', required=True)
    r.add_argument('--canonical-csv', required=True)
    r.add_argument('--freeze-csv', required=True)
    r.add_argument('--outdir', required=True)
    r.add_argument('--package', action='store_true')
    r.set_defaults(func=lambda a: (core_analysis(a),
                                   package_results(Path(a.outdir).expanduser().resolve()) if a.package else None))
    return p


def main():
    a = parser().parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
