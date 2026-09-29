"""
pavement_iri_pipeline.py  — Complete IRI Prediction Pipeline
=============================================================
Stages: EDA → Data Prep → Scenarios → Train/Eval → Tuning → SHAP → Figures
"""

import os, warnings, copy, ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import (GroupKFold, KFold,
                                     RandomizedSearchCV)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import shap
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "Final Data - R01.xlsx")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR  = os.path.join(RESULTS_DIR, "tables")
SHAP_DIR    = os.path.join(RESULTS_DIR, "shap")
INTERIM_DIR = os.path.join(RESULTS_DIR, "interim_data")
for d in [FIGURES_DIR, TABLES_DIR, SHAP_DIR, INTERIM_DIR]:
    os.makedirs(d, exist_ok=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
TARGET       = "IRI"
GROUP        = "section_id"
N_FOLDS      = 10
RANDOM_STATE = 42

MODELS = {
    "Linear Regression" : LinearRegression(),
    "Random Forest"     : RandomForestRegressor(n_estimators=200,
                            random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting" : GradientBoostingRegressor(n_estimators=200,
                            random_state=RANDOM_STATE),
    "XGBoost"           : XGBRegressor(n_estimators=200,
                            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1),
}

FEATURE_LABELS = {
    "age":"Pavement Age (yrs)", "time_from_MRR":"Time from Last MRR (yrs)",
    "ac_thickness":"AC Layer Thickness (in)", "base_thickness":"Base Thickness (in)",
    "esal":"ESAL", "precip":"Precipitation (mm)", "temp":"Mean Temperature (°C)",
    "fatigue_crack_pct":"Fatigue Cracking (%)", "wheel_crack_pct":"Wheel Path Cracking (%)",
    "bsg":"Bulk Specific Gravity", "mean_ac_pct":"Mean Asphalt Content (%)",
    "IRI_prev":"Previous IRI (m/km)",
}

STYLE = {"font.family":"Arial","font.size":11,"axes.titlesize":13,
         "axes.labelsize":11,"axes.spines.top":False,"axes.spines.right":False,
         "figure.dpi":150}
plt.rcParams.update(STYLE)

C_RED="#E05252"; C_BLUE="#2E75B6"; C_GREEN="#70AD47"
C_DARK="#1F4E79"; C_ORG="#ED7D31"

# ── HELPERS ───────────────────────────────────────────────────────────────────
def mape(y_true, y_pred):
    mask = np.array(y_true) != 0
    return np.mean(np.abs((np.array(y_true)[mask]-np.array(y_pred)[mask])
                          /np.array(y_true)[mask]))*100

def eval_group_kfold(model, X, y, groups):
    # Single pooled-OOF loop: R2/RMSE/MAE/MAPE are all computed on the
    # concatenated out-of-fold predictions (same quantity Stage 5 uses for
    # r2_oof), NOT sklearn's mean-of-per-fold-scores. Those are two different,
    # both-valid statistics — mixing them is what caused table vs. figure
    # numbers to disagree. "Std" still comes from the per-fold R2 scores,
    # since that's a genuinely different (and still useful) stability metric.
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.zeros(len(y))
    fold_r2 = []
    for tr,te in gkf.split(X,y,groups=groups):
        m = copy.deepcopy(model); m.fit(X.iloc[tr],y.iloc[tr])
        pred = m.predict(X.iloc[te])
        oof[te] = pred
        fold_r2.append(r2_score(y.iloc[te],pred))
    fold_r2 = np.array(fold_r2)
    return {"CV R2":round(r2_score(y,oof),4),
            "CV R2 Std":round(fold_r2.std(),4),
            "CV RMSE":round(np.sqrt(mean_squared_error(y,oof)),4),
            "CV MAE":round(mean_absolute_error(y,oof),4),
            "CV MAPE":round(mape(y.values,oof),2),
            "fold_r2":fold_r2.tolist()}

def eval_random_kfold(model, X, y):
    # Same pooled-OOF approach as eval_group_kfold — see note above.
    rkf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    fold_r2 = []
    for tr,te in rkf.split(X,y):
        m = copy.deepcopy(model); m.fit(X.iloc[tr],y.iloc[tr])
        pred = m.predict(X.iloc[te])
        oof[te] = pred
        fold_r2.append(r2_score(y.iloc[te],pred))
    fold_r2 = np.array(fold_r2)
    return {"RKF R2":round(r2_score(y,oof),4),
            "RKF R2 Std":round(fold_r2.std(),4),
            "RKF RMSE":round(np.sqrt(mean_squared_error(y,oof)),4),
            "RKF MAE":round(mean_absolute_error(y,oof),4),
            "fold_r2":fold_r2.tolist()}

def style_ws(ws, df, title):
    ws.insert_rows(1); ws.insert_rows(1)
    ws["A1"] = title
    ws["A1"].font = Font(name="Arial",size=12,bold=True)
    ws.merge_cells(f"A1:{get_column_letter(df.shape[1])}1")
    ws["A1"].alignment = Alignment(horizontal="center")
    side = Side(style="thin",color="AAAAAA")
    bdr  = Border(left=side,right=side,top=side,bottom=side)
    hdr  = PatternFill("solid",fgColor="1F4E79")
    alt  = PatternFill("solid",fgColor="EBF3FB")
    best_val = df["CV R2"].max() if "CV R2" in df.columns else None
    for ci,col in enumerate(df.columns,1):
        cell=ws.cell(row=2,column=ci,value=col)
        cell.font=Font(name="Arial",size=10,bold=True,color="FFFFFF")
        cell.fill=hdr; cell.border=bdr
        cell.alignment=Alignment(horizontal="center",wrap_text=True)
        ws.column_dimensions[get_column_letter(ci)].width=max(len(col)+2,12)
    best_fill=PatternFill("solid",fgColor="E2EFDA")
    for ri,row in enumerate(df.itertuples(index=False),start=3):
        is_best=(best_val is not None and "CV R2" in df.columns and
                 row[df.columns.tolist().index("CV R2")]==best_val)
        fill=best_fill if is_best else(alt if ri%2==0
              else PatternFill("solid",fgColor="FFFFFF"))
        for ci,val in enumerate(row,1):
            cell=ws.cell(row=ri,column=ci,value=val)
            cell.font=Font(name="Arial",size=10,bold=True if is_best else False)
            cell.fill=fill; cell.border=bdr
            cell.alignment=Alignment(horizontal="center")
    ws.freeze_panes="A3"

def savefig(name):
    plt.savefig(os.path.join(FIGURES_DIR,name),dpi=300,bbox_inches="tight")
    plt.close()
    print(f"  Saved: {name}")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*60+"\n  STAGE 0 — EDA\n"+"="*60)

raw = pd.read_excel(DATA_PATH, sheet_name="Final_Data", parse_dates=["VISIT_DATE"])
raw = raw.sort_values(["SECTION_ID_SHORT","VISIT_DATE"]).reset_index(drop=True)
print(f"  Raw shape: {raw.shape}")

# ── EDA Fig 1: Missing Values (missingno-style, manual) ─────────────────────
miss_matrix = ~raw.isnull()   # True=present(blue), False=missing(white)
fig, ax = plt.subplots(figsize=(18, 6))
ax.imshow(miss_matrix.T.values, aspect="auto", cmap="Blues",
          interpolation="none", vmin=0, vmax=1)
# Column labels
ax.set_xticks([])
ax.set_yticks(range(len(raw.columns)))
ax.set_yticklabels(raw.columns, fontsize=8)
ax.yaxis.set_tick_params(length=0)
ax.tick_params(axis="y", pad=4)
# Rotated column labels at top
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(np.linspace(0, len(raw)-1, len(raw.columns)))
ax2.set_xticklabels(raw.columns, rotation=45, ha="left", fontsize=8)
ax2.tick_params(length=0)
# Row count labels
ax.set_ylabel("")
ax.text(-0.005, 1.0, "1",           transform=ax.transAxes,
        fontsize=9, ha="right", va="top",   fontweight="bold")
ax.text(-0.005, 0.0, str(len(raw)), transform=ax.transAxes,
        fontsize=9, ha="right", va="bottom", fontweight="bold")
# Missing % annotations on right
for i, col in enumerate(raw.columns):
    pct = raw[col].isnull().mean() * 100
    if pct > 0:
        ax.text(len(raw)+10, i, f"{pct:.0f}%", va="center",
                fontsize=7, color="#C0392B")
ax.set_title("Missing Values Distribution across Final_Data",
             fontsize=13, fontweight="bold", pad=20)
ax.spines[["top","right","bottom","left"]].set_visible(False)
plt.tight_layout()
savefig("EDA_01_missing_values.png")

# ── EDA Fig 2: IRI Distribution & Outliers ──────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(12,5))
axes[0].hist(raw["IRI"].dropna(), bins=40, color="#6B82C4",
             edgecolor="white", alpha=0.85)
axes[0].set_xlabel("IRI"); axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of IRI (m/km)", fontweight="bold")
from scipy.stats import gaussian_kde
iri_vals = raw["IRI"].dropna()
kde = gaussian_kde(iri_vals)
xs  = np.linspace(iri_vals.min(), iri_vals.max(), 200)
ax2 = axes[0].twinx(); ax2.plot(xs, kde(xs), color="#3D5AA3", lw=2)
ax2.set_yticks([])
bp = axes[1].boxplot(iri_vals, patch_artist=True, vert=False,
                     boxprops=dict(facecolor="#C0392B",alpha=0.75),
                     medianprops=dict(color="#2C3E50",lw=2),
                     flierprops=dict(marker="o",markerfacecolor="gray",
                                     markersize=3,alpha=0.5))
axes[1].set_xlabel("IRI"); axes[1].set_yticks([])
axes[1].set_title("Boxplot for IRI (Outlier Detection)", fontweight="bold")
plt.tight_layout()
savefig("EDA_02_iri_distribution.png")

# ── EDA Fig 3: Correlation Matrix ───────────────────────────────────────────
eda_cols = ["IRI","Age","Time from last MRR","AC Thickness","Base Thickness",
            "Subbase Thickness","ESAL","BSG","Mean Asphalt Content",
            "% Fatigue & Long. Cracking","% Wheel Path Cracking","Precip","Temp"]
corr = raw[eda_cols].corr()
fig, ax = plt.subplots(figsize=(13,11))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax, annot_kws={"size":8})
ax.set_title("Correlation Matrix for Selected Pavement Features",
             fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("EDA_03_correlation_matrix.png")

# ── EDA Fig 4: Impact of Age & ESAL on IRI (scatter) ────────────────────────
fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].scatter(raw["Age"], raw["IRI"], alpha=0.3, s=12,
                color="#8B2635", edgecolors="none")
axes[0].set_xlabel("Age (Years)"); axes[0].set_ylabel("IRI")
axes[0].set_title("Impact of Pavement Age on IRI", fontweight="bold")
axes[1].scatter(raw["ESAL"], raw["IRI"], alpha=0.3, s=12,
                color="#2D6A4F", edgecolors="none")
axes[1].set_xlabel("ESAL"); axes[1].set_ylabel("IRI")
axes[1].set_title("Impact of ESAL on IRI", fontweight="bold")
plt.tight_layout()
savefig("EDA_04_age_esal_scatter.png")

# ── EDA Fig 5: IRI TrendLines (6 subplots) ──────────────────────────────────
pairs = [("Age","Age"), ("ESAL","ESAL"), ("AC Thickness","AC Thickness"),
         ("Temp","Temp"), ("Precip","Precip"),
         ("% Fatigue & Long. Cracking","% Fatigue & Long. Cracking")]
fig, axes = plt.subplots(2,3,figsize=(15,9))
for ax,(col,lbl) in zip(axes.flatten(), pairs):
    sub = raw[[col,"IRI"]].dropna()
    ax.scatter(sub[col], sub["IRI"], alpha=0.2, s=8, color=C_BLUE)
    ax.set_xlabel(lbl); ax.set_ylabel("IRI (m/km)")
    ax.set_title(f"IRI vs {lbl}", fontweight="bold")
    try:
        m,b = np.polyfit(sub[col], sub["IRI"], 1)
        xs  = np.linspace(sub[col].min(), sub[col].max(), 100)
        ax.plot(xs, m*xs+b, "r-", lw=1.8)
    except Exception:
        pass
plt.suptitle("IRI TrendLines", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("EDA_05_trendlines.png")

# ── EDA Fig 6: Deterioration Curves ─────────────────────────────────────────
sample_secs = (raw.groupby("SECTION_ID_SHORT")["IRI"]
               .count().sort_values(ascending=False).head(10).index)
fig, ax = plt.subplots(figsize=(13,6))
for sec in sample_secs:
    sub = raw[raw["SECTION_ID_SHORT"]==sec].sort_values("Age")
    ax.plot(sub["Age"], sub["IRI"], marker="o", ms=4, alpha=0.85, label=sec)
ax.set_xlabel("Pavement Age (Years)"); ax.set_ylabel("IRI (m/km)")
ax.set_title("Pavement Deterioration Curves (IRI Evolution Over Time)",
             fontweight="bold")
ax.legend(fontsize=8, ncol=1, loc="upper left",
          title="Section ID", title_fontsize=8)
ax.yaxis.grid(True, alpha=0.4, ls="--")
plt.tight_layout()
savefig("EDA_06_deterioration_curves.png")

# ── EDA Fig 7: Average IRI by State ─────────────────────────────────────────
raw["state"] = raw["SECTION_ID_SHORT"].str.split("_").str[0].astype(int)
state_iri = raw.groupby("state")["IRI"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(11,5))
colors_st = plt.cm.viridis(np.linspace(0.85, 0.15, len(state_iri)))
ax.bar(state_iri.index.astype(str), state_iri.values,
       color=colors_st, edgecolor="white", width=0.7)
ax.set_xlabel("State Code (Derived from Section ID)")
ax.set_ylabel("Average IRI (m/km)")
ax.set_title("Average Network IRI by State", fontweight="bold")
ax.yaxis.grid(True, alpha=0.3, ls="--")
plt.tight_layout()
savefig("EDA_07_iri_by_state.png")

# ── EDA Stats ────────────────────────────────────────────────────────────────
eda_stat_cols = ["IRI","Age","AC Thickness","Base Thickness","Subbase Thickness",
                 "ESAL","Precip","Temp","% Fatigue & Long. Cracking",
                 "% Wheel Path Cracking","BSG","Mean Asphalt Content"]
eda_stats = raw[eda_stat_cols].describe().T.round(3)
eda_stats["missing"]   = raw[eda_stat_cols].isnull().sum()
eda_stats["missing_%"] = (eda_stats["missing"]/len(raw)*100).round(1)
eda_stats.to_excel(os.path.join(TABLES_DIR,"EDA_summary_stats.xlsx"))
print("  EDA complete.")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*60+"\n  STAGE 1 — DATA PREPARATION\n"+"="*60)

df = raw.copy()
df = df.rename(columns={
    "SECTION_ID_SHORT":"section_id","VISIT_DATE":"visit_date",
    "Age":"age","Time from last MRR":"time_from_MRR",
    "AC Thickness":"ac_thickness","Base Thickness":"base_thickness",
    "ESAL":"esal","BSG":"bsg","Mean Asphalt Content":"mean_ac_pct",
    "% Fatigue & Long. Cracking":"fatigue_crack_pct",
    "% Wheel Path Cracking":"wheel_crack_pct",
    "Precip":"precip","Temp":"temp",
})
df = df.sort_values(["section_id","visit_date"]).reset_index(drop=True)
df["IRI_prev"] = df.groupby("section_id")["IRI"].shift(1)
print(f"  IRI_prev: {df['IRI_prev'].notna().sum()} non-null "
      f"({df['IRI_prev'].isnull().sum()} first-visit NaN)")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — SCENARIO BUILDING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*60+"\n  STAGE 2 — SCENARIO BUILDING\n"+"="*60)

FEAT_BASE = ["age","time_from_MRR","precip","temp","esal",
             "ac_thickness","base_thickness","fatigue_crack_pct","wheel_crack_pct"]
PREV      = ["IRI_prev"]

scenario_defs = {
    "S1":{"features":FEAT_BASE,                          "row_filter":None},
    "S2":{"features":FEAT_BASE+["bsg"],                  "row_filter":"bsg"},
    "S3":{"features":FEAT_BASE+["mean_ac_pct"],          "row_filter":"mean_ac_pct"},
    "S4":{"features":FEAT_BASE+["bsg","mean_ac_pct"],    "row_filter":"both"},
}

for sc, sdef in scenario_defs.items():
    rf = sdef["row_filter"]
    if rf is None:      sub = df.copy()
    elif rf=="both":    sub = df[df["bsg"].notna()&df["mean_ac_pct"].notna()].copy()
    else:               sub = df[df[rf].notna()].copy()

    for ver, feat_cols in {"A":sdef["features"], "B":sdef["features"]+PREV}.items():
        cols = [GROUP,TARGET]+feat_cols
        out  = sub[cols].dropna()
        out.to_csv(os.path.join(INTERIM_DIR,f"{sc}_{ver}.csv"),index=False)
        print(f"  {sc}_{ver}: {len(out):>4} rows | "
              f"{out[GROUP].nunique()} sections | {len(feat_cols)} features")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*60+"\n  STAGE 3 — TRAINING & EVALUATION\n"+"="*60)

all_results = []
for sc in ["S1","S2","S3","S4"]:
    for ver in ["A","B"]:
        key = f"{sc}_{ver}"
        csv = os.path.join(INTERIM_DIR,f"{key}.csv")
        if not os.path.exists(csv): continue
        data   = pd.read_csv(csv)
        groups = data[GROUP]; X = data.drop(columns=[GROUP,TARGET]); y = data[TARGET]
        print(f"\n  {key} ({len(data)} rows, {X.shape[1]} features)")
        for mname, model in MODELS.items():
            print(f"    {mname} ...", end=" ")
            gkf = eval_group_kfold(model, X, y, groups)
            rkf = eval_random_kfold(model, X, y)
            row = {"Scenario":sc,"Version":ver,"Key":key,"Model":mname,
                   "Rows":len(data),"Features":X.shape[1],
                   "Sections":groups.nunique()}
            row.update(gkf); row.update(rkf)
            all_results.append(row)
            print(f"Group R2={gkf['CV R2']:.4f} | Random R2={rkf['RKF R2']:.4f}")

results_df = pd.DataFrame(all_results)
export_cols = ["Key","Model","Rows","Features","Sections",
               "CV R2","CV R2 Std","CV RMSE","CV MAE","CV MAPE",
               "RKF R2","RKF R2 Std","RKF RMSE","RKF MAE"]
export = results_df[export_cols].sort_values("CV R2",ascending=False)
export.to_csv(os.path.join(TABLES_DIR,"all_results.csv"),index=False)

wb = openpyxl.Workbook(); ws = wb.active; ws.title="All Results"
for row in [export.columns.tolist()]+export.values.tolist(): ws.append(row)
style_ws(ws, export, "All Experiments — Group KFold vs Random KFold")
wb.save(os.path.join(TABLES_DIR,"results_all.xlsx"))

print("\n  Best Group KFold per scenario:")
best_sum = (results_df.sort_values("CV R2",ascending=False)
            .groupby("Key",as_index=False).first()
            [["Key","Model","Rows","CV R2","CV R2 Std","RKF R2"]])
print(best_sum.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — HYPERPARAMETER TUNING 
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*60+"\n  STAGE 4 — HYPERPARAMETER TUNING\n"+"="*60)

TUNE_GRIDS = {
    "XGBoost":(XGBRegressor(random_state=RANDOM_STATE,verbosity=0,n_jobs=-1),
               {"n_estimators":[200,300,500,700],"max_depth":[3,4,5,6],
                "learning_rate":[0.01,0.05,0.08,0.1],"subsample":[0.7,0.8,0.9,1.0],
                "colsample_bytree":[0.6,0.7,0.8,1.0],"min_child_weight":[1,3,5],
                "gamma":[0,0.1,0.2]}),
    "Random Forest":(RandomForestRegressor(random_state=RANDOM_STATE,n_jobs=-1),
                    {"n_estimators":[200,300,500],"max_depth":[None,10,15,20],
                     "min_samples_split":[2,5,10],"min_samples_leaf":[1,2,4],
                     "max_features":["sqrt","log2",0.5]}),
    "Gradient Boosting":(GradientBoostingRegressor(random_state=RANDOM_STATE),
                         {"n_estimators":[200,300,500],"max_depth":[3,4,5],
                          "learning_rate":[0.01,0.05,0.1],"subsample":[0.7,0.8,0.9],
                          "min_samples_split":[2,5,10]}),
}

kf_tune     = KFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE)
tune_results = []
c_scenarios  = [k for k in results_df["Key"].unique() if k.endswith("_B")]

for sc_key in c_scenarios:
    data_sc = pd.read_csv(os.path.join(INTERIM_DIR,f"{sc_key}.csv"))
    X_sc = data_sc.drop(columns=[GROUP,TARGET])
    y_sc = data_sc[TARGET]; g_sc = data_sc[GROUP]
    print(f"\n  -- {sc_key} ({len(data_sc)} rows) --")
    for mname,(base_model,param_grid) in TUNE_GRIDS.items():
        print(f"    Tuning {mname} ...", end=" ")
        search = RandomizedSearchCV(base_model,param_distributions=param_grid,
                                    n_iter=50,cv=kf_tune,scoring="r2",
                                    n_jobs=-1,random_state=RANDOM_STATE)
        search.fit(X_sc,y_sc)
        tuned = base_model.set_params(**search.best_params_)
        res   = eval_group_kfold(tuned,X_sc,y_sc,g_sc)
        rkf   = eval_random_kfold(tuned,X_sc,y_sc)
        print(f"Group R2={res['CV R2']:.4f} | Random R2={rkf['RKF R2']:.4f}")
        tune_results.append({
            "Scenario":sc_key,"Model":f"Tuned {mname}",
            "Best Params":str(search.best_params_),**res,
            "RKF R2":rkf["RKF R2"],"RKF R2 Std":rkf["RKF R2 Std"],
        })

tune_df = pd.DataFrame(tune_results)
tune_df.drop(columns=["fold_r2"],errors="ignore").to_csv(
    os.path.join(TABLES_DIR,"tuning_results.csv"),index=False)

# Compare tuned vs untuned — pick the true best.
# Selection is on "RKF R2" (Random K-Fold), the official metric for maintenance
# planning (Chapter 3, Section 3.5.1) — every candidate here is a "_B" scenario.
# "CV R2" (Group K-Fold) is still computed/reported alongside it, but must not
# drive the selection.
best_tune_row_cand = tune_df.sort_values("RKF R2",ascending=False).iloc[0]
best_untune_row    = results_df[results_df["Version"]=="B"].sort_values(
                        "RKF R2",ascending=False).iloc[0]

if best_tune_row_cand["RKF R2"] >= best_untune_row["RKF R2"]:
    best_tune_row  = best_tune_row_cand
    best_tune_name = best_tune_row["Model"].replace("Tuned ","")
    best_key       = best_tune_row["Scenario"]
    best_params_str= best_tune_row["Best Params"]
    is_tuned       = True
    print(f"\n  Overall best: TUNED — {best_tune_row['Model']} | {best_key} | "
          f"RKF R2={best_tune_row['RKF R2']:.4f} (CV/GKF R2={best_tune_row['CV R2']:.4f})")
else:
    best_tune_row  = best_untune_row
    best_tune_name = best_untune_row["Model"]
    best_key       = best_untune_row["Key"]
    best_params_str= None
    is_tuned       = False
    print(f"\n  Overall best: UNTUNED — {best_tune_name} | {best_key} | "
          f"RKF R2={best_tune_row['RKF R2']:.4f} (CV/GKF R2={best_tune_row['CV R2']:.4f})")

# Load best scenario data
data_best = pd.read_csv(os.path.join(INTERIM_DIR,f"{best_key}.csv"))
X_best    = data_best.drop(columns=[GROUP,TARGET])
y_best    = data_best[TARGET]
g_best    = data_best[GROUP]

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — OOF PREDICTIONS (for consistent figures)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*60+"\n  STAGE 5 — OOF PREDICTIONS\n"+"="*60)

if is_tuned and best_params_str:
    best_params = ast.literal_eval(best_params_str)
else:
    best_params = {"n_estimators":200}
if "XGBoost" in best_tune_name:
    oof_model = XGBRegressor(**best_params,
                              random_state=RANDOM_STATE,verbosity=0,n_jobs=-1)
elif "Random Forest" in best_tune_name:
    oof_model = RandomForestRegressor(**best_params,
                                       random_state=RANDOM_STATE,n_jobs=-1)
else:
    oof_model = GradientBoostingRegressor(**best_params,
                                           random_state=RANDOM_STATE)

# NOTE: best_key is always a "_B" (maintenance planning) scenario — Chapter 3,
# Section 3.5.1 specifies Random K-Fold (KFold, shuffle=True) as the validation
# protocol for maintenance planning, NOT Group K-Fold. Group K-Fold is reserved
# for the network planning use case (Section 3.5.2), computed separately below
# for r2_net/oof_a. Using GroupKFold here was the source of the Fig 2/3/5 and
# use_case_comparison contradiction with Chapter 3.
rkf_best = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_pred = np.zeros(len(y_best))
for tr,te in rkf_best.split(X_best,y_best):
    oof_model.fit(X_best.iloc[tr],y_best.iloc[tr])
    oof_pred[te] = oof_model.predict(X_best.iloc[te])

r2_oof   = r2_score(y_best,oof_pred)
rmse_oof = np.sqrt(mean_squared_error(y_best,oof_pred))
mae_oof  = mean_absolute_error(y_best,oof_pred)
print(f"  OOF R²   = {r2_oof:.4f}")
print(f"  OOF RMSE = {rmse_oof:.4f} m/km")
print(f"  OOF MAE  = {mae_oof:.4f} m/km")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*60+"\n  STAGE 6 — SHAP ANALYSIS\n"+"="*60)
print(f"  SHAP on: {best_tune_name} | {best_key}")

X_shap = X_best.copy()
X_shap.columns = [FEATURE_LABELS.get(c,c) for c in X_shap.columns]
feat_names = list(X_shap.columns)

shap_model = copy.deepcopy(oof_model)
shap_model.fit(X_shap,y_best)
explainer   = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(X_shap)
mean_shap   = pd.Series(np.abs(shap_values).mean(axis=0),
                         index=feat_names).sort_values(ascending=False)
total_shap  = mean_shap.sum()

print("  SHAP feature importance:")
for feat,val in mean_shap.items():
    print(f"    {feat:<40} {val:.4f}  ({val/total_shap*100:.1f}%)")

pd.DataFrame(shap_values,columns=feat_names).to_csv(
    os.path.join(SHAP_DIR,"shap_values.csv"),index=False)
X_shap.to_csv(os.path.join(SHAP_DIR,"X_for_shap.csv"),index=False)

# SHAP Bar
fig,ax = plt.subplots(figsize=(9,7))
colors_shap=[C_DARK if i==0 else C_BLUE if i<3 else "#9DC3E6"
             for i in range(len(mean_shap))]
ax.barh(range(len(mean_shap)),mean_shap.values[::-1],
        color=colors_shap[::-1],edgecolor="white")
ax.set_yticks(range(len(mean_shap)))
ax.set_yticklabels(mean_shap.index[::-1],fontsize=9)
ax.set_xlabel("Mean |SHAP Value|")
for i,(val,feat) in enumerate(zip(mean_shap.values[::-1],mean_shap.index[::-1])):
    ax.text(val+0.001,i,f"{val/total_shap*100:.1f}%",va="center",
            fontsize=8,color=C_DARK)
plt.tight_layout(); savefig("SHAP_01_bar.png")

# SHAP Beeswarm
fig,_ = plt.subplots(figsize=(10,7))
shap.summary_plot(shap_values,X_shap,show=False,max_display=len(feat_names))
plt.tight_layout(); savefig("SHAP_02_beeswarm.png")

# SHAP Dependence — top 2
for feat in mean_shap.index[:2]:
    fig,ax = plt.subplots(figsize=(8,5))
    fi  = feat_names.index(feat)
    int_feat = mean_shap.index[1] if feat==mean_shap.index[0] else mean_shap.index[0]
    sc2 = ax.scatter(X_shap[feat],shap_values[:,fi],
                     c=X_shap[int_feat],cmap="coolwarm",alpha=0.5,s=15)
    plt.colorbar(sc2,ax=ax,label=int_feat)
    ax.axhline(0,color="gray",ls="--",lw=0.8)
    ax.set_xlabel(feat); ax.set_ylabel(f"SHAP value for\n{feat}")
    plt.tight_layout()
    safe=feat.replace(" ","_").replace("/","-").replace("(","").replace(")","")
    savefig(f"SHAP_03_dependence_{safe}.png")

# SHAP Table — Maintenance Planning
shap_table = pd.DataFrame([{"Rank":i+1,"Feature":f,
    "Mean |SHAP|":round(v,4),"% of Total":round(v/total_shap*100,1),
    "Cumulative %":round(mean_shap.iloc[:i+1].sum()/total_shap*100,1)}
    for i,(f,v) in enumerate(mean_shap.items())])
shap_table.to_excel(os.path.join(TABLES_DIR,"shap_importance_maintenance.xlsx"),
                    index=False)
print("  SHAP (Maintenance Planning) complete.")

# ── SHAP: Network Planning (best _A scenario, no IRI_prev) ───────────────
print("\n  Computing SHAP for Network Planning (_A scenarios) ...")

# Find best _A model
best_a_row = results_df[results_df["Version"]=="A"].sort_values(
                "CV R2",ascending=False).iloc[0]
best_a_key = best_a_row["Key"]
print(f"  Network Planning best: {best_a_row['Model']} | {best_a_key} | "
      f"CV R2={best_a_row['CV R2']:.4f}")

data_a  = pd.read_csv(os.path.join(INTERIM_DIR,f"{best_a_key}.csv"))
X_a     = data_a.drop(columns=[GROUP,TARGET])
y_a     = data_a[TARGET]
g_a     = data_a[GROUP]

X_a.columns = [FEATURE_LABELS.get(c,c) for c in X_a.columns]
feat_names_a = list(X_a.columns)

if "XGBoost" in best_a_row["Model"]:
    net_model = XGBRegressor(n_estimators=200,random_state=RANDOM_STATE,
                              verbosity=0,n_jobs=-1)
elif "Random Forest" in best_a_row["Model"]:
    net_model = RandomForestRegressor(n_estimators=200,random_state=RANDOM_STATE,
                                       n_jobs=-1)
else:
    net_model = GradientBoostingRegressor(n_estimators=200,random_state=RANDOM_STATE)

net_model.fit(X_a, y_a)
explainer_a   = shap.TreeExplainer(net_model)
shap_values_a = explainer_a.shap_values(X_a)
mean_shap_a   = pd.Series(np.abs(shap_values_a).mean(axis=0),
                           index=feat_names_a).sort_values(ascending=False)
total_shap_a  = mean_shap_a.sum()

print("  Network Planning SHAP importance:")
for feat,val in mean_shap_a.items():
    print(f"    {feat:<40} {val:.4f}  ({val/total_shap_a*100:.1f}%)")

# Network Planning SHAP Bar
fig,ax = plt.subplots(figsize=(9,6))
colors_net=[C_ORG if i==0 else "#F4A261" if i<3 else "#FDDCB5"
            for i in range(len(mean_shap_a))]
ax.barh(range(len(mean_shap_a)),mean_shap_a.values[::-1],
        color=colors_net[::-1],edgecolor="white")
ax.set_yticks(range(len(mean_shap_a)))
ax.set_yticklabels(mean_shap_a.index[::-1],fontsize=9)
ax.set_xlabel("Mean |SHAP Value|")
for i,(val,feat) in enumerate(zip(mean_shap_a.values[::-1],
                                   mean_shap_a.index[::-1])):
    ax.text(val+0.001,i,f"{val/total_shap_a*100:.1f}%",va="center",
            fontsize=8,color=C_ORG)
plt.tight_layout()
savefig("SHAP_04_network_planning_bar.png")

# Save Network Planning SHAP table
shap_table_a = pd.DataFrame([{"Rank":i+1,"Feature":f,
    "Mean |SHAP|":round(v,4),"% of Total":round(v/total_shap_a*100,1),
    "Cumulative %":round(mean_shap_a.iloc[:i+1].sum()/total_shap_a*100,1)}
    for i,(f,v) in enumerate(mean_shap_a.items())])
shap_table_a.to_excel(os.path.join(TABLES_DIR,"shap_importance_network.xlsx"),
                       index=False)

# ── Comparison Table: Maintenance vs Network Planning ────────────────────
print("\n  Building Use Case Comparison Table ...")

# OOF for Network Planning
gkf_a = GroupKFold(n_splits=N_FOLDS)
oof_a = np.zeros(len(y_a))
for tr,te in gkf_a.split(X_a,y_a,groups=g_a):
    net_model.fit(X_a.iloc[tr],y_a.iloc[tr])
    oof_a[te] = net_model.predict(X_a.iloc[te])

r2_net   = r2_score(y_a,oof_a)
rmse_net = np.sqrt(mean_squared_error(y_a,oof_a))
mae_net  = mean_absolute_error(y_a,oof_a)

comparison = pd.DataFrame([
    {"Use Case":"Maintenance Planning",
     "Description":"Known road sections — predict next IRI measurement",
     "Validation":"Random K-Fold",
     "IRI_prev Used":"Yes","Rows":len(data_best),
     "Sections":g_best.nunique(),
     "Best Model":best_tune_row["Model"],
     "R²":round(r2_oof,4),"RMSE":round(rmse_oof,4),"MAE":round(mae_oof,4),
     "Top Feature":f"{mean_shap.index[0]} ({mean_shap.iloc[0]/total_shap*100:.1f}%)"},
    {"Use Case":"Network Planning",
     "Description":"Unseen road sections — predict IRI for new/unmonitored roads",
     "Validation":"Group K-Fold",
     "IRI_prev Used":"No","Rows":len(data_a),
     "Sections":g_a.nunique(),
     "Best Model":best_a_row["Model"],
     "R²":round(r2_net,4),"RMSE":round(rmse_net,4),"MAE":round(mae_net,4),
     "Top Feature":f"{mean_shap_a.index[0]} ({mean_shap_a.iloc[0]/total_shap_a*100:.1f}%)"},
])
comparison.to_excel(os.path.join(TABLES_DIR,"use_case_comparison.xlsx"),index=False)
print(f"  Network Planning: R²={r2_net:.4f}, RMSE={rmse_net:.4f}")
print("  SHAP complete.")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — PUBLICATION FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*60+"\n  STAGE 7 — PUBLICATION FIGURES\n"+"="*60)

# ── Fig 1: Data Leakage ──────────────────────────────────────────────────────
s1a = results_df[results_df["Key"]=="S1_A"].copy()
model_names = list(MODELS.keys())
x = np.arange(len(model_names)); w = 0.35

fig,ax = plt.subplots(figsize=(11,6))
rkf_vals = [s1a[s1a["Model"]==m]["RKF R2"].values[0] for m in model_names]
gkf_vals = [s1a[s1a["Model"]==m]["CV R2"].values[0]  for m in model_names]
rkf_std  = [s1a[s1a["Model"]==m]["RKF R2 Std"].values[0] for m in model_names]
gkf_std  = [s1a[s1a["Model"]==m]["CV R2 Std"].values[0]  for m in model_names]

bars_r=ax.bar(x-w/2,rkf_vals,w,label="Random K-Fold",
              color=C_RED,alpha=0.85,yerr=rkf_std,capsize=4,
              error_kw={"color":"#555","lw":1.2})
bars_g=ax.bar(x+w/2,gkf_vals,w,label="Group K-Fold",
              color=C_BLUE,alpha=0.85,yerr=gkf_std,capsize=4,
              error_kw={"color":"#555","lw":1.2})
for bar,v,std in zip(bars_r,rkf_vals,rkf_std):
    ax.text(bar.get_x()+bar.get_width()/2,v+std+0.012,f"{v:.4f}",
            ha="center",fontsize=9,color=C_RED,fontweight="bold")
for bar,v,std in zip(bars_g,gkf_vals,gkf_std):
    ax.text(bar.get_x()+bar.get_width()/2,v+std+0.012,f"{v:.4f}",
            ha="center",fontsize=9,color=C_BLUE,fontweight="bold")
for i,(rv,gv,rs,gs) in enumerate(zip(rkf_vals,gkf_vals,rkf_std,gkf_std)):
    gap=rv-gv
    ax.annotate("",xy=(x[i]+w/2,rv),xytext=(x[i]+w/2,gv),
                arrowprops=dict(arrowstyle="<->",color="#666",lw=1.2))
    ax.text(x[i]+w/2+0.03,(rv+gv)/2,f"Δ={gap:.4f}",
            fontsize=8,color="#444",va="center")
ax.set_xticks(x); ax.set_xticklabels(model_names,fontsize=10)
ax.set_ylabel("Cross-Validated R²"); ax.set_ylim(0,1.08)
ax.legend(fontsize=10); ax.yaxis.grid(True,alpha=0.3,ls="--")
plt.tight_layout(); savefig("FIG_01_data_leakage.png")

# ── Fig 2: Progressive Improvement ──────────────────────────────────────────
# Steps 1–2 illustrate the leakage gap on S1_A (network planning, Group K-Fold
# is the honest protocol there — Section 3.5.2). Steps 3+ are "_B" (maintenance
# planning) scenarios, whose official protocol is Random K-Fold (Section 3.5.1),
# so they must be sorted/reported on "RKF R2", not "CV R2" (Group K-Fold).
s1a_best_rkf = s1a.sort_values("RKF R2",ascending=False).iloc[0]
s1a_best_gkf = s1a.sort_values("CV R2",ascending=False).iloc[0]
s1c = results_df[results_df["Key"]=="S1_B"].sort_values("RKF R2",ascending=False).iloc[0]
best_c = results_df[results_df["Version"]=="B"].sort_values("RKF R2",ascending=False).iloc[0]

steps = [
    {"label":"Prior work\n(Random K-Fold)","r2":s1a_best_rkf["RKF R2"],
     "std":s1a_best_rkf["RKF R2 Std"],"color":C_RED},
    {"label":"Honest baseline\n(Group K-Fold, S1_A)","r2":s1a_best_gkf["CV R2"],
     "std":s1a_best_gkf["CV R2 Std"],"color":C_BLUE},
    {"label":"+ IRI_prev\n(S1_B, Random K-Fold)","r2":s1c["RKF R2"],
     "std":s1c["RKF R2 Std"],"color":C_DARK},
]
if best_c["Key"]!="S1_B":
    steps.append({"label":f"Best scenario\n({best_c['Key']}, Random K-Fold)","r2":best_c["RKF R2"],
                  "std":best_c["RKF R2 Std"],"color":"#7030A0"})
steps.append({"label":f"+ Tuning\n({best_tune_name}, Random K-Fold)","r2":r2_oof,
              "std":best_tune_row["RKF R2 Std"],"color":C_ORG})

fig,ax = plt.subplots(figsize=(12,6))
for i,s in enumerate(steps):
    ax.bar(i,s["r2"],color=s["color"],alpha=0.85,edgecolor="white",width=0.6,
           yerr=s["std"],capsize=4,error_kw={"color":"#555","lw":1.5})
    ax.text(i,s["r2"]+s["std"]+0.015,f"R²={s['r2']:.4f}",
            ha="center",va="bottom",fontsize=9,fontweight="bold",color=s["color"])
ax.set_xticks(range(len(steps)))
ax.set_xticklabels([s["label"] for s in steps],fontsize=10)
ax.set_ylabel("Cross-Validated R²")
ax.set_ylim(0,1.05)
ax.yaxis.grid(True,alpha=0.3,ls="--")
plt.tight_layout(); savefig("FIG_02_progressive.png")

# ── Fig 3: Actual vs Predicted (OOF) ────────────────────────────────────────
fig,ax = plt.subplots(figsize=(8,7))
ax.scatter(y_best,oof_pred,alpha=0.3,s=12,color=C_BLUE,edgecolors="none")
lims=[min(y_best.min(),oof_pred.min())-0.1,max(y_best.max(),oof_pred.max())+0.1]
ax.plot(lims,lims,"r--",lw=1.5,label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual IRI (m/km)"); ax.set_ylabel("Predicted IRI (m/km)")
ax.text(0.05,0.92,f"R² = {r2_oof:.4f}\nRMSE = {rmse_oof:.4f} m/km",
        transform=ax.transAxes,fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4",facecolor="white",
                  edgecolor="#CCC",alpha=0.9))
ax.legend(fontsize=10)
plt.tight_layout(); savefig("FIG_03_actual_vs_predicted.png")

# ── Fig 4: Per-fold R² ──────────────────────────────────────────────────────
# X_best/y_best is a maintenance-planning ("_B") scenario, so folds must be
# Random K-Fold (Section 3.5.1), same splitter as the Stage 5 OOF predictions.
fold_scores={name:[] for name in MODELS}
for mname,model in MODELS.items():
    for tr,te in rkf_best.split(X_best,y_best):
        m=copy.deepcopy(model); m.fit(X_best.iloc[tr],y_best.iloc[tr])
        fold_scores[mname].append(r2_score(y_best.iloc[te],m.predict(X_best.iloc[te])))

fig,ax = plt.subplots(figsize=(9,6))
bp=ax.boxplot([fold_scores[n] for n in MODELS],patch_artist=True,
              medianprops=dict(color="white",lw=2))
colors_bp=[C_BLUE,C_GREEN,C_ORG,C_DARK]
for patch,color in zip(bp["boxes"],colors_bp):
    patch.set_facecolor(color); patch.set_alpha(0.8)
for elem in ["whiskers","fliers","caps"]: plt.setp(bp[elem],color="#555")
ax.set_xticks(range(1,len(MODELS)+1))
ax.set_xticklabels(list(MODELS.keys()),fontsize=10)
ax.set_ylabel("R² per Fold (Random K-Fold, Maintenance Planning)")
ax.yaxis.grid(True,alpha=0.3,ls="--")
ax.axhline(0,color="#AAA",lw=0.8,ls=":")
plt.tight_layout(); savefig("FIG_04_per_fold.png")

# ── Fig 5: Use Case Comparison (Maintenance vs Network Planning) ─────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: R² comparison
use_cases  = ["Maintenance Planning\n(Random K-Fold)\nKnown sections + IRI_prev",
               "Network Planning\n(Group K-Fold)\nUnseen sections, no IRI_prev"]
r2_vals    = [r2_oof, r2_net]
rmse_vals  = [rmse_oof, rmse_net]
colors_uc  = [C_BLUE, C_ORG]

ax = axes[0]
bars = ax.bar(range(2), r2_vals, color=colors_uc, alpha=0.85,
              edgecolor="white", width=0.5)
for bar,v in zip(bars,r2_vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"R²={v:.4f}",
            ha="center", fontsize=12, fontweight="bold",
            color=bar.get_facecolor())
ax.set_xticks(range(2))
ax.set_xticklabels(use_cases, fontsize=9, multialignment="center")
ax.set_ylabel("OOF R² (Cross-Validated)", fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title("Predictive Accuracy by Use Case", fontweight="bold")
ax.yaxis.grid(True, alpha=0.3, ls="--")

# Right: SHAP comparison (top 5 features each)
ax = axes[1]
top_maint = mean_shap.head(5)
top_net   = mean_shap_a.head(5)
y_pos_m   = np.arange(5)
y_pos_n   = np.arange(5) - 0.35
h         = 0.32

bars_m = ax.barh(y_pos_m, top_maint.values/total_shap*100,
                  h, label="Maintenance Planning", color=C_BLUE, alpha=0.8)
bars_n = ax.barh(y_pos_m - h, top_net.values/total_shap_a*100,
                  h, label="Network Planning", color=C_ORG, alpha=0.8)

ax.set_yticks(y_pos_m - h/2)
ax.set_yticklabels([f"Rank {i+1}" for i in range(5)], fontsize=9)
ax.set_xlabel("SHAP Importance (%)", fontsize=10)
ax.set_title("Top 5 Features by Use Case", fontweight="bold")
ax.legend(fontsize=9)

# Annotate feature names
for i,(vm,vn,fm,fn) in enumerate(zip(
        top_maint.values/total_shap*100,
        top_net.values/total_shap_a*100,
        top_maint.index, top_net.index)):
    ax.text(vm+0.3, y_pos_m[i],     f"{fm[:20]}", va="center", fontsize=7, color=C_BLUE)
    ax.text(vn+0.3, y_pos_m[i]-h,   f"{fn[:20]}", va="center", fontsize=7, color=C_ORG)

ax.yaxis.grid(True, alpha=0.3, ls="--")
plt.tight_layout()
savefig("FIG_05_use_case_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*60+"\n  FINAL SUMMARY\n"+"="*60)
print("\n  Progressive improvement:")
for s in steps:
    print(f"    {s['label'].replace(chr(10),' '):<35} R²={s['r2']:.4f}")
print(f"\n  Best tuned model : {best_tune_row['Model']}")
print(f"  Scenario         : {best_key}  (Maintenance Planning)")
print(f"  OOF R² (Random K-Fold, official) = {r2_oof:.4f}  (Fig 2 & Fig 3)")
print(f"  RKF R² (mean)    = {best_tune_row['RKF R2']:.4f} ± {best_tune_row['RKF R2 Std']:.4f}")
print(f"  CV/GKF R² (mean, reference only) = {best_tune_row['CV R2']:.4f} ± {best_tune_row['CV R2 Std']:.4f}")
print(f"  OOF RMSE         = {rmse_oof:.4f} m/km")
print(f"  OOF MAE          = {mae_oof:.4f} m/km")
print(f"\n  Top SHAP features:")
for feat,val in mean_shap.head(5).items():
    print(f"    {feat:<40} {val/total_shap*100:.1f}%")
print(f"\n  Network Planning (no IRI_prev):")
print(f"    Best model : {best_a_row['Model']} | {best_a_key}")
print(f"    OOF R²     = {r2_net:.4f}")
print(f"    OOF RMSE   = {rmse_net:.4f} m/km")
print(f"    Top feature: {mean_shap_a.index[0]} "
      f"({mean_shap_a.iloc[0]/total_shap_a*100:.1f}%)")
print(f"\n  All outputs: {RESULTS_DIR}")
print("  Pipeline complete.")
