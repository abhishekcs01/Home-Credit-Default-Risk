from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import replace_day_sentinel


def _months_recent_mask(months_balance: pd.Series, last_n: int) -> pd.Series:
    return months_balance >= -last_n


def aggregate_bureau_balance(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    bb = bureau_balance.copy()
    if bb.empty or "SK_ID_BUREAU" not in bb.columns:
        return pd.DataFrame(columns=["SK_ID_BUREAU"])

    if "MONTHS_BALANCE" in bb.columns:
        bb["MONTHS_BALANCE_NEG"] = -bb["MONTHS_BALANCE"]
    if "STATUS" in bb.columns:
        bb["STATUS_NUM"] = pd.to_numeric(bb["STATUS"], errors="coerce")

    num_cols = [c for c in bb.select_dtypes(include=[np.number]).columns if c != "SK_ID_BUREAU"]
    if num_cols:
        agg_dict = {c: ["mean", "max", "min", "sum", "count", "std"] for c in num_cols}
        g_base = bb.groupby("SK_ID_BUREAU").agg(agg_dict)
        g_base.columns = ["BB_" + "_".join(col).strip() for col in g_base.columns.values]
        g_base = g_base.reset_index()
    else:
        g_base = pd.DataFrame({"SK_ID_BUREAU": bb["SK_ID_BUREAU"].unique()})

    extra_rows = []
    for bid, gx in bb.groupby("SK_ID_BUREAU"):
        er = {"SK_ID_BUREAU": bid}
        mb = gx["MONTHS_BALANCE"] if "MONTHS_BALANCE" in gx.columns else None
        if mb is not None and len(gx):
            rw = np.exp(np.clip(mb.astype(float).to_numpy(), -96.0, 0.0) / 20.0)
            if "STATUS_NUM" in gx.columns:
                sn = gx["STATUS_NUM"].astype(float).to_numpy()
                wsum = np.nansum(sn * rw)
                wden = np.nansum(rw)
                er["BB_RECENCY_WMEAN_STATUS"] = float(wsum / wden) if wden > 0 else np.nan
            else:
                er["BB_RECENCY_WMEAN_STATUS"] = np.nan
        else:
            er["BB_RECENCY_WMEAN_STATUS"] = np.nan

        for wm in (3, 6, 12, 13):
            pre = f"BB_W{wm}_"
            if mb is None:
                er[f"{pre}CNT"] = 0
                er[f"{pre}STATUS_MEAN"] = np.nan
                er[f"{pre}DELINQ_SHARE"] = np.nan
                continue
            sub = gx.loc[_months_recent_mask(mb, wm)]
            er[f"{pre}CNT"] = int(len(sub))
            if len(sub) and "STATUS_NUM" in sub.columns:
                er[f"{pre}STATUS_MEAN"] = float(sub["STATUS_NUM"].mean())
                er[f"{pre}DELINQ_SHARE"] = float((sub["STATUS_NUM"].fillna(0) > 1).mean())
            else:
                er[f"{pre}STATUS_MEAN"] = np.nan
                er[f"{pre}DELINQ_SHARE"] = np.nan
        extra_rows.append(er)

    extra_df = pd.DataFrame(extra_rows)
    return g_base.merge(extra_df, on="SK_ID_BUREAU", how="left")


def aggregate_bureau(bureau: pd.DataFrame, bb_per_bureau: pd.DataFrame | None) -> pd.DataFrame:
    bu = replace_day_sentinel(bureau, [c for c in bureau.columns if "DAYS" in c.upper()])
    if bb_per_bureau is not None and len(bb_per_bureau):
        bu = bu.merge(bb_per_bureau, on="SK_ID_BUREAU", how="left")

    num_for_agg = [c for c in bu.select_dtypes(include=[np.number]).columns if c not in ("SK_ID_CURR", "SK_ID_BUREAU")]
    agg_dict = {c: ["mean", "max", "min", "sum", "count", "std"] for c in num_for_agg}
    g = bu.groupby("SK_ID_CURR").agg(agg_dict)
    g.columns = ["BURO_" + "_".join(col).strip() for col in g.columns.values]
    g = g.reset_index()

    if "CREDIT_ACTIVE" in bu.columns:
        active_dummies = pd.get_dummies(bu["CREDIT_ACTIVE"], prefix="BURO_ACTIVE")
        active_dummies["SK_ID_CURR"] = bu["SK_ID_CURR"].values
        active_sum = active_dummies.groupby("SK_ID_CURR").sum().add_prefix("BURO_CNT_ACTIVE_").reset_index()
        g = g.merge(active_sum, on="SK_ID_CURR", how="left")

    if "CREDIT_TYPE" in bu.columns:
        n_types = bu.groupby("SK_ID_CURR")["CREDIT_TYPE"].nunique().rename("BURO_N_CREDIT_TYPES")
        g = g.merge(n_types.reset_index(), on="SK_ID_CURR", how="left")

    return g


def aggregate_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    p = replace_day_sentinel(prev, [c for c in prev.columns if "DAYS" in c.upper()])
    num_cols = [c for c in p.select_dtypes(include=[np.number]).columns if c not in ("SK_ID_CURR", "SK_ID_PREV")]
    agg_dict = {c: ["mean", "max", "min", "sum", "count", "std"] for c in num_cols}
    gg = p.groupby("SK_ID_CURR").agg(agg_dict)
    gg.columns = ["PREV_" + "_".join(col).strip() for col in gg.columns.values]
    g = gg.reset_index()

    if "NAME_CONTRACT_STATUS" in p.columns:
        appr = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(np.float32).groupby(p["SK_ID_CURR"]).mean()
        refused = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(np.float32).groupby(p["SK_ID_CURR"]).mean()
        g = g.merge(appr.rename("PREV_APPROVED_RATE").reset_index(), on="SK_ID_CURR", how="left")
        g = g.merge(refused.rename("PREV_REFUSED_RATE").reset_index(), on="SK_ID_CURR", how="left")
        appr_n = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(np.int32).groupby(p["SK_ID_CURR"]).sum()
        ref_n = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(np.int32).groupby(p["SK_ID_CURR"]).sum()
        decided = (appr_n + ref_n).replace(0, np.nan)
        g = g.merge((appr_n / decided).rename("PREV_APPROVED_AMONG_DECIDED").reset_index(), on="SK_ID_CURR", how="left")
        g = g.merge((ref_n / decided).rename("PREV_REFUSED_AMONG_DECIDED").reset_index(), on="SK_ID_CURR", how="left")
        g = g.merge((ref_n > 0).astype(np.float32).rename("PREV_HAS_REFUSAL").reset_index(), on="SK_ID_CURR", how="left")
        if "Canceled" in p["NAME_CONTRACT_STATUS"].unique():
            canc = (p["NAME_CONTRACT_STATUS"] == "Canceled").astype(np.float32).groupby(p["SK_ID_CURR"]).mean()
            g = g.merge(canc.rename("PREV_CANCELED_RATE").reset_index(), on="SK_ID_CURR", how="left")

    if "FLAG_LAST_APPL_PER_CONTRACT" in p.columns:
        last_y = (p["FLAG_LAST_APPL_PER_CONTRACT"] == "Y").astype(np.float32).groupby(p["SK_ID_CURR"]).mean()
        g = g.merge(last_y.rename("PREV_LAST_APPL_PER_CONTRACT_RATE").reset_index(), on="SK_ID_CURR", how="left")

    return g


def _installments_window_agg(ins: pd.DataFrame, days: int, prefix: str) -> pd.DataFrame:
    if ins.empty or "SK_ID_CURR" not in ins.columns or "DAYS_INSTALMENT" not in ins.columns:
        return pd.DataFrame(columns=["SK_ID_CURR"])
    sub = ins[ins["DAYS_INSTALMENT"] > -days]
    if sub.empty:
        return pd.DataFrame(columns=["SK_ID_CURR"])
    num_cols = [c for c in sub.select_dtypes(include=[np.number]).columns if c not in ("SK_ID_CURR", "SK_ID_PREV") and not str(c).startswith("_")]
    if not num_cols:
        return sub.groupby("SK_ID_CURR").size().rename(f"{prefix}CNT").reset_index()
    agg_dict = {c: ["mean", "sum", "max"] for c in num_cols}
    agg = sub.groupby("SK_ID_CURR").agg(agg_dict)
    agg.columns = [prefix + "_".join(col).strip() for col in agg.columns.values]
    return agg.reset_index()


def aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    ins = replace_day_sentinel(inst, [c for c in inst.columns if "DAYS" in c.upper()])
    if {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(ins.columns):
        ins["INST_PAYMENT_DIFF"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
        ins["INST_PAYMENT_DIFF_POS"] = ins["INST_PAYMENT_DIFF"].clip(lower=0)
        ins["INST_IS_LATE"] = (ins["INST_PAYMENT_DIFF"] > 0).astype(np.float32)
        ins["INST_IS_SEVERE_LATE"] = (ins["INST_PAYMENT_DIFF"] > 30).astype(np.float32)
    if {"AMT_PAYMENT", "AMT_INSTALMENT"}.issubset(ins.columns):
        ins["INST_PAYMENT_RATIO"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"].replace(0, np.nan)
    if "INST_PAYMENT_DIFF_POS" in ins.columns and "DAYS_INSTALMENT" in ins.columns:
        ins["_REC_W"] = np.exp(np.clip(ins["DAYS_INSTALMENT"].astype(float).to_numpy(), -8000.0, 0.0) / 450.0)

    num_cols = [c for c in ins.select_dtypes(include=[np.number]).columns if c not in ("SK_ID_CURR", "SK_ID_PREV") and not str(c).startswith("_")]
    agg_dict = {c: ["mean", "max", "min", "sum", "count", "std"] for c in num_cols}
    g = ins.groupby("SK_ID_CURR").agg(agg_dict)
    g.columns = ["INST_" + "_".join(col).strip() for col in g.columns.values]
    g = g.reset_index()

    if "INST_IS_LATE" in ins.columns:
        g = g.merge(ins.groupby("SK_ID_CURR")["INST_IS_LATE"].mean().rename("INST_LATE_PAYMENT_RATE").reset_index(), on="SK_ID_CURR", how="left")
    if "INST_IS_SEVERE_LATE" in ins.columns:
        g = g.merge(
            ins.groupby("SK_ID_CURR")["INST_IS_SEVERE_LATE"].mean().rename("INST_SEVERE_LATE_PAYMENT_RATE").reset_index(),
            on="SK_ID_CURR",
            how="left",
        )

    if "_REC_W" in ins.columns and "INST_PAYMENT_DIFF_POS" in ins.columns:
        num = (ins["INST_PAYMENT_DIFF_POS"].fillna(0) * ins["_REC_W"]).groupby(ins["SK_ID_CURR"]).sum()
        den = ins["_REC_W"].groupby(ins["SK_ID_CURR"]).sum()
        g = g.merge((num / den.replace(0, np.nan)).rename("INST_RECENCY_WMEAN_LATE_DAYS").reset_index(), on="SK_ID_CURR", how="left")

    for dwin, lab in ((90, "INST_W90_"), (180, "INST_W180_"), (365, "INST_W365_")):
        w = _installments_window_agg(ins, dwin, lab)
        if len(w):
            g = g.merge(w, on="SK_ID_CURR", how="left")
    return g


def _balance_table_window_agg(df: pd.DataFrame, prefix: str, months: int, id_col: str) -> pd.DataFrame:
    if df.empty or "MONTHS_BALANCE" not in df.columns:
        return pd.DataFrame(columns=[id_col])
    sub = df.loc[_months_recent_mask(df["MONTHS_BALANCE"], months)]
    if sub.empty:
        return pd.DataFrame(columns=[id_col])
    num_cols = [c for c in sub.select_dtypes(include=[np.number]).columns if c not in (id_col, "SK_ID_PREV") and not str(c).startswith("_")]
    if not num_cols:
        return sub.groupby(id_col).size().rename(f"{prefix}W{months}_CNT").reset_index()
    agg_dict = {c: ["mean", "max", "sum"] for c in num_cols}
    agg = sub.groupby(id_col).agg(agg_dict)
    agg.columns = [f"{prefix}W{months}_" + "_".join(col).strip() for col in agg.columns.values]
    return agg.reset_index()


def aggregate_pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    p = replace_day_sentinel(pos, [c for c in pos.columns if "DAYS" in c.upper() or c.startswith("SK_DPD")])
    if "SK_DPD" in p.columns:
        p["POS_DPD_POS"] = (p["SK_DPD"].fillna(0) > 0).astype(np.float32)

    num_cols = [c for c in p.select_dtypes(include=[np.number]).columns if c not in ("SK_ID_CURR", "SK_ID_PREV") and not str(c).startswith("_")]
    agg_dict = {c: ["mean", "max", "min", "sum", "count", "std"] for c in num_cols}
    g = p.groupby("SK_ID_CURR").agg(agg_dict)
    g.columns = ["POS_" + "_".join(col).strip() for col in g.columns.values]
    g = g.reset_index()

    if "POS_DPD_POS" in p.columns:
        g = g.merge(p.groupby("SK_ID_CURR")["POS_DPD_POS"].mean().rename("POS_DELINQ_FREQ").reset_index(), on="SK_ID_CURR", how="left")
    if "MONTHS_BALANCE" in p.columns:
        p = p.copy()
        p["_REC_W"] = np.exp(np.clip(p["MONTHS_BALANCE"].astype(float).to_numpy(), -96.0, 0.0) / 20.0)
        if "SK_DPD" in p.columns:
            x = p["SK_DPD"].fillna(0).astype(float)
            num = (x * p["_REC_W"]).groupby(p["SK_ID_CURR"]).sum()
            den = p["_REC_W"].groupby(p["SK_ID_CURR"]).sum()
            g = g.merge((num / den.replace(0, np.nan)).rename("POS_RECENCY_WMEAN_DPD").reset_index(), on="SK_ID_CURR", how="left")

    for wm in (3, 6, 12, 13):
        w = _balance_table_window_agg(p, "POS_", wm, "SK_ID_CURR")
        if len(w):
            g = g.merge(w, on="SK_ID_CURR", how="left")
    if "SK_DPD" in p.columns:
        g = g.merge(p.groupby("SK_ID_CURR")["SK_DPD"].max().rename("POS_MAX_SK_DPD").reset_index(), on="SK_ID_CURR", how="left")
        g = g.merge(((p.groupby("SK_ID_CURR")["SK_DPD"].max() > 0).astype(np.float32)).rename("POS_ANY_HIST_DELINQ").reset_index(), on="SK_ID_CURR", how="left")
    return g


def aggregate_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    c = cc.copy()
    if {"AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"}.issubset(c.columns):
        c["CC_UTILIZATION"] = c["AMT_BALANCE"] / c["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    if "CC_UTILIZATION" in c.columns:
        c["CC_HIGH_UTIL"] = (c["CC_UTILIZATION"] > 0.9).astype(np.float32)
    if "SK_DPD" in c.columns:
        c["CC_DPD_POS"] = (c["SK_DPD"].fillna(0) > 0).astype(np.float32)

    num_cols = [col for col in c.select_dtypes(include=[np.number]).columns if col not in ("SK_ID_CURR", "SK_ID_PREV") and not str(col).startswith("_")]
    agg_dict = {col: ["mean", "max", "min", "sum", "count", "std"] for col in num_cols}
    g = c.groupby("SK_ID_CURR").agg(agg_dict)
    g.columns = ["CC_" + "_".join(col).strip() for col in g.columns.values]
    g = g.reset_index()

    if "CC_HIGH_UTIL" in c.columns:
        g = g.merge(c.groupby("SK_ID_CURR")["CC_HIGH_UTIL"].mean().rename("CC_HIGH_UTILIZATION_FREQ").reset_index(), on="SK_ID_CURR", how="left")
    if "CC_UTILIZATION" in c.columns:
        u80 = (c["CC_UTILIZATION"] > 0.8).astype(np.float32).groupby(c["SK_ID_CURR"]).mean().rename("CC_UTIL_OVER80_FREQ")
        g = g.merge(u80.reset_index(), on="SK_ID_CURR", how="left")
        any80 = (c.groupby("SK_ID_CURR")["CC_UTILIZATION"].max() > 0.8).astype(np.float32).rename("CC_ANY_UTIL_OVER80")
        g = g.merge(any80.reset_index(), on="SK_ID_CURR", how="left")
    if "CC_DPD_POS" in c.columns:
        g = g.merge(c.groupby("SK_ID_CURR")["CC_DPD_POS"].mean().rename("CC_DELINQ_FREQ").reset_index(), on="SK_ID_CURR", how="left")

    if "MONTHS_BALANCE" in c.columns and "CC_UTILIZATION" in c.columns:
        cc2 = c.copy()
        cc2["_REC_W"] = np.exp(np.clip(cc2["MONTHS_BALANCE"].astype(float).to_numpy(), -96.0, 0.0) / 20.0)
        u = cc2["CC_UTILIZATION"].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
        num = (u * cc2["_REC_W"]).groupby(cc2["SK_ID_CURR"]).sum()
        den = cc2["_REC_W"].groupby(cc2["SK_ID_CURR"]).sum()
        g = g.merge((num / den.replace(0, np.nan)).rename("CC_RECENCY_WMEAN_UTIL").reset_index(), on="SK_ID_CURR", how="left")

    for wm in (3, 6, 12, 13):
        w = _balance_table_window_agg(c, "CC_", wm, "SK_ID_CURR")
        if len(w):
            g = g.merge(w, on="SK_ID_CURR", how="left")
    return g


def merge_stage(
    app: pd.DataFrame,
    stage: str,
    *,
    bureau: pd.DataFrame | None = None,
    bureau_balance: pd.DataFrame | None = None,
    previous_application: pd.DataFrame | None = None,
    installments_payments: pd.DataFrame | None = None,
    pos_cash_balance: pd.DataFrame | None = None,
    credit_card_balance: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = app.copy()
    if stage in ("app", "application"):
        return out

    if stage in ("bureau", "previous", "installments", "pos_cash", "credit_card", "full"):
        bb_agg = aggregate_bureau_balance(bureau_balance) if bureau_balance is not None else None
        bu_agg = aggregate_bureau(bureau, bb_agg) if bureau is not None else None
        if bu_agg is not None:
            out = out.merge(bu_agg, on="SK_ID_CURR", how="left")
    if stage in ("previous", "installments", "pos_cash", "credit_card", "full"):
        prev_agg = aggregate_previous_application(previous_application) if previous_application is not None else None
        if prev_agg is not None:
            out = out.merge(prev_agg, on="SK_ID_CURR", how="left")
    if stage in ("installments", "pos_cash", "credit_card", "full"):
        ins_agg = aggregate_installments(installments_payments) if installments_payments is not None else None
        if ins_agg is not None:
            out = out.merge(ins_agg, on="SK_ID_CURR", how="left")
    if stage in ("pos_cash", "credit_card", "full"):
        pos_agg = aggregate_pos_cash(pos_cash_balance) if pos_cash_balance is not None else None
        if pos_agg is not None:
            out = out.merge(pos_agg, on="SK_ID_CURR", how="left")
    if stage in ("credit_card", "full"):
        cc_agg = aggregate_credit_card(credit_card_balance) if credit_card_balance is not None else None
        if cc_agg is not None:
            out = out.merge(cc_agg, on="SK_ID_CURR", how="left")
    return out

