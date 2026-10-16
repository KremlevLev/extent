"""Rebuild Figures 4-7 and their audit from archived EXP129-132 JSONs.
Offline only: no model execution, checkpoints, or remote access.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "figures"
COLORS = ["#32799c", "#d18b35", "#7260a7", "#279079"]

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

def report(data, worker):
    return next(w["report"] for w in data["workers"] if w["worker"] == worker)

def evaluation(r, step):
    return next(e for e in r["fits"]["MAIN"]["evaluations"] if e["step"] == step)

def nll(e, panel):
    return e[{"A": "A64", "B": "B32", "C": "C64"}[panel]]["nll"]

def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    data = {e: json.loads((REPO / f"results/EXP-{e}-session-1.json").read_text(encoding="utf-8")) for e in range(129, 133)}
    reference = report(data[129], 0)
    audit = {"date": "2026-10-17", "scope": "Single seed123; reused validation panels; partial horizons; no independent confirmation", "sources": [], "runs": [], "points": []}
    for exp, d in data.items():
        path = REPO / f"results/EXP-{exp}-session-1.json"
        audit["sources"].append(dict(experiment=exp, path=path.relative_to(REPO).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), status=d["status"], wall_hours=d["wall_hours"]))
        for w in d["workers"]:
            r = w["report"]; f = r["fits"]["MAIN"]
            assert w["exit"] == 0 and r["source_replay"]["passed"] and r["hook_background"]["passed"]
            assert r["data"]["panels"] == reference["data"]["panels"]
            assert r["contract"]["source_pin_sha"] == reference["contract"]["source_pin_sha"]
            assert not r["pending_fit_states"] and f["step"] == f["durable_step"]
            assert f["finalization_status"] == "completed"
            failed_profiles = []
            for label, endpoint in f["final"].items():
                for layer in endpoint["layer_profile"]["layers"]:
                    if not layer["accepted"]:
                        failed_profiles.append(dict(endpoint=label, layer=layer["layer"], discrepancy=layer["relative_output_discrepancy"]))
            audit["runs"].append(dict(experiment=exp, worker=w["worker"], arm=r["contract"]["arm"]["name"], step=f["step"], best_step=f["best_step"], cost=r["cost"], final_profile_discrepancies=failed_profiles, last_nll={p:v["nll"] for p,v in f["final"]["last"]["panels"].items()}))
            for e in f["evaluations"]:
                if "A64" not in e: continue
                for panel in "ABC":
                    audit["points"].append(dict(experiment=exp, worker=w["worker"], step=e["step"], input_tokens=e["step"]*256, panel=panel, nll=nll(e,panel), kind="registered"))
            last = f["final"]["last"]
            for panel in "ABC":
                audit["points"].append(dict(experiment=exp, worker=w["worker"], step=last["step"], input_tokens=last["step"]*256, panel=panel, nll=last["panels"][panel]["nll"], kind="last"))
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False, "svg.hashsalt":"extent-recent-20261017"})
    # Matched cells show absolute NLL and change from the norm+tanh control.
    import numpy as np
    arms=[(130,0,"Norm + tanh"),(131,0,"Unlock dynamics"),(131,1,"Unlock + MLP"),(130,1,"Combined reset")]
    values=np.array([[nll(evaluation(report(data[e],w),3072),p) for p in "ABC"] for e,w,_ in arms])
    delta=values-values[0]
    fig,ax=plt.subplots(figsize=(9,4.1))
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
    cmap=LinearSegmentedColormap.from_list("benefit",["#58a891","#f7f7f4","#d28a68"])
    im=ax.imshow(delta,cmap=cmap,norm=TwoSlopeNorm(vmin=-.4,vcenter=0,vmax=.06),aspect="auto")
    for row in range(4):
        for col in range(3):
            change="reference" if row==0 else f"change {delta[row,col]:+.3f}"
            ax.text(col,row,f"{values[row,col]:.3f} NLL\n{change}",ha="center",va="center",fontsize=11)
    ax.set_xticks(range(3),["Validation A (64 windows)","Validation B (32 windows)","Validation C (64 windows)"])
    ax.xaxis.tick_top();ax.tick_params(length=0,pad=10)
    ax.set_yticks(range(4),[z[2] for z in arms]);ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True);ax.grid(which="minor",color="white",linewidth=3);ax.tick_params(which="minor",length=0)
    cb=fig.colorbar(im,ax=ax,pad=.035,fraction=.04);cb.set_label("NLL change vs reference; negative is better")
    fig.suptitle("The combined reset improves every validation panel",fontsize=14,y=1.04)
    fig.text(.5,.015,"EXP-130/131 | same source and data | 3,072 updates | one seed; reused validation",ha="center",fontsize=9,color="#555555")
    fig.tight_layout(rect=(0,.06,1,.95));save(fig,"recent_matched_ablations")
    # A64 alone: independent axes avoid empty ranges and mixing distinct panels.
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    names={129:("Rank32 control","Pre-gate norm"),130:("Norm + tanh","Combined reset"),131:("Unlock dynamics","Unlock + MLP"),132:("True-token CE","Teacher KL")}
    titles={129:"129 | Normalization alone",130:"130 | Resetting dynamics + factors",131:"131 | Unlocking weights + MLP",132:"132 | Changing the training objective"}
    for ax,exp in zip(axes.flat,range(129,133)):
        for worker in (0,1):
            r=report(data[exp],worker);es=[e for e in r["fits"]["MAIN"]["evaluations"] if "A64" in e]
            xs=[e["step"]/1000 for e in es];ys=[nll(e,"A") for e in es]
            ax.plot(xs,ys,marker="o",ms=5,lw=2.2,color=COLORS[worker],label=names[exp][worker])
            last=r["fits"]["MAIN"]["final"]["last"];lx=last["step"]/1000;ly=last["panels"]["A"]["nll"]
            ax.plot([xs[-1],lx],[ys[-1],ly],ls=":",lw=1.6,color=COLORS[worker])
            ax.scatter(lx,ly,marker="X",s=65,color=COLORS[worker],zorder=4)
            ax.annotate(f"{ly:.3f}",(lx,ly),xytext=(6,5 if worker==0 else -12),textcoords="offset points",fontsize=9,color=COLORS[worker])
        ax.margins(x=.17,y=.22);ax.set_title(titles[exp],loc="left",fontsize=11,fontweight="bold",pad=32);ax.grid(alpha=.2)
        for w in (0,1):ax.text(.01 + .52*w,1.025,names[exp][w],transform=ax.transAxes,color=COLORS[w],fontsize=9,fontweight="bold")
        ax.set_xlabel("Additional updates (thousands)");ax.set_ylabel("Validation A NLL (lower is better)")
    fig.suptitle("What happens after the intervention reaches full strength?",fontsize=14)
    fig.text(.5,.015,"Each panel has its own axis range. Circles: registered evaluations. X: final deadline evaluation; dotted link.\nOnly the same A64 panel is plotted; no mixing with the earlier A16 monitor. All runs are partial, seed 123.",ha="center",fontsize=9,color="#555555")
    fig.tight_layout(rect=(0,.075,1,.95));save(fig,"recent_trajectories")
    # Two separately matched pairs: direct before/after comparisons, no scatter inference.
    fig,axes=plt.subplots(2,3,figsize=(12,5.6));profile_rows=[]
    for row,(exp,step,labels) in enumerate([(129,4096,("Control","+ norm")),(130,3072,("Norm + tanh","+ reset"))]):
        pair=[]
        for worker in (0,1):
            r=report(data[exp],worker);f=r["fits"]["MAIN"];assert f["final"]["best"]["step"]==step
            layer=f["final"]["best"]["layer_profile"]["layers"][26];assert layer["accepted"]
            y=nll(evaluation(r,step),"A")
            pair.append([layer["output_rms"],100*layer["mlp_addition_absorbed_fraction"],y])
            profile_rows.append(dict(experiment=exp,worker=worker,step=step,layer=26,output_rms=layer["output_rms"],absorbed_fraction=layer["mlp_addition_absorbed_fraction"],panel_A_nll=y))
        for col in range(3):
            ax=axes[row,col];v=[pair[0][col],pair[1][col]]
            for worker in (0,1):
                ax.scatter(v[worker],1-worker,color=COLORS[worker],s=75,zorder=3)
                text=(f"{v[worker]:.2e}" if col==0 else f"{v[worker]:.2f}%" if col==1 else f"{v[worker]:.3f}")
                ax.annotate(text,(v[worker],1-worker),xytext=(0,10),textcoords="offset points",ha="center",fontsize=10)
            ax.set_yticks([1,0],labels if col==0 else ["",""]);ax.set_ylim(-.45,1.6)
            if col==0:ax.set_xscale("log")
            ax.margins(x=.28);ax.grid(axis="x",alpha=.18);ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0)
            if row==0:ax.set_title(["Layer 26 output RMS (log)","Absorbed MLP additions","Validation A NLL"][col],fontsize=11)
            ax.set_xlabel(f"EXP-{exp} | {step:,} updates",fontsize=9)
    fig.suptitle("Smaller activations are not sufficient: compare quality in the right column",fontsize=13)
    fig.text(.5,.01,"Rows are separate matched comparisons. Layer measurements: one diagnostic window; NLL: all 64 validation windows.\nThe normalization reduces RMS by millions of times, yet NLL slightly worsens. Reset improves NLL as well.",ha="center",fontsize=9,color="#555555")
    fig.tight_layout(rect=(0,.09,1,.94));save(fig,"scale_quality_dissociation")
    audit["profile_points"]=profile_rows
    # Categorical panels are paired comparisons, not a continuous line.
    fig,axes=plt.subplots(1,2,figsize=(10,4),gridspec_kw={"width_ratios":[1.5,1]})
    for j,panel in enumerate("ABC"):
        vals=[nll(evaluation(report(data[132],w),2560),panel) for w in (0,1)]
        axes[0].plot(vals,[j,j],color="#adb5bc",lw=3,zorder=1)
        for w in (0,1):
            axes[0].scatter(vals[w],j,color=COLORS[w],s=65,zorder=3,label=("True-token CE" if w==0 else "Teacher KL") if j==0 else None)
            axes[0].annotate(f"{vals[w]:.3f}",(vals[w],j),xytext=(0,-17 if w==0 else 9),textcoords="offset points",ha="center",fontsize=9)
        axes[0].text(7.395,j,f"+{vals[1]-vals[0]:.3f}",va="center",fontsize=10,color="#a06a24")
    axes[0].set_yticks(range(3),["Validation A","Validation B","Validation C"]);axes[0].invert_yaxis();axes[0].set_ylim(2.55,-.6);axes[0].set_xlim(6.99,7.46)
    axes[0].set_xlabel("NLL at 2,560 updates (lower is better)");axes[0].set_title("KL is worse on every panel",loc="left",fontsize=12);axes[0].legend(loc="upper left",fontsize=8);axes[0].grid(axis="x",alpha=.18)
    for w in (0,1):
        c=report(data[132],w)["cost"];v=c["training_seconds"]/c["accepted_updates"]
        axes[1].barh(w,v,color=COLORS[w],height=.45);axes[1].text(v+.12,w,f"{v:.2f} s",va="center",fontsize=11)
    axes[1].set_yticks([0,1],["CE","KL + teacher"]);axes[1].invert_yaxis();axes[1].set_xlim(0,8.5);axes[1].set_xlabel("Seconds per accepted update");axes[1].set_title("KL costs about 28% more time",loc="left",fontsize=12);axes[1].grid(axis="x",alpha=.18)
    fig.suptitle("EXP-132 | Distillation from the same prepared checkpoint",fontsize=14)
    fig.text(.5,.015,"Same data and trainable capacity; one seed. NLL is measured against the true next token.\nRuntime averages span each partial session and include training/compilation overhead. No claim about all distillation recipes.",ha="center",fontsize=9,color="#555555")
    fig.tight_layout(rect=(0,.12,1,.93));save(fig,"distillation_quality_cost")
    dump(REPO/"results/EXP-129-132-session-1-analysis.json",audit)
    with (OUT/"recent_points.csv").open("w",encoding="utf-8",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(audit["points"][0]));writer.writeheader();writer.writerows(audit["points"])
    dump(OUT/"recent_provenance.json",{k:v for k,v in audit.items() if k!="points"})
    print(f"Rebuilt four figures from {len(audit['points'])} measured points; eight workers audited.")

if __name__=="__main__":main()
