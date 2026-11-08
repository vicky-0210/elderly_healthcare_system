"""Development-only factorial removals of the full proposed teaching framework."""
from study import *

def main():
    st=Study(); dev=A/"development_amp"
    teachers=[json.loads(p.read_text()) for p in dev.glob("E2*/done.json")]
    st.teacher=choose(teachers)["name"]
    kd=choose([json.loads(p.read_text()) for p in dev.glob("E4*/done.json")])["config"]
    joint=dict(T=kd["T"],kd=kd["kd"],rep=.15,gate=True,label="weighted",aux=[0,1,2,3],fusion="all")
    variants={"full":joint,"hard_labels":joint|{"label":"hard"},"no_repr":joint|{"rep":0},
              "no_conf_gate":joint|{"gate":False},"no_logit":joint|{"kd":0},"no_aux":joint|{"aux":[]},
              "no_morphology":{k:v for k,v in joint.items() if k!="fusion"},
              "fixed_fusion":joint|{"fusion":"no_quality"},
              "no_teacher":{k:v for k,v in joint.items() if k not in ("T","kd","gate","rep")}}
    rows=[]
    for name,cfg in variants.items():
        r=st.train("joint_"+name,cfg)
        rows.append({k:r[k] for k in ("name","ap","sens90","f1","brier","ece","params","train_n")})
    pd.DataFrame(rows).to_csv(R/"joint_ablation.csv",index=False)
    # Configuration manifest is aggregate, containing no patient identifiers.
    save_json(R/"configurations.json",{p.parent.name:json.loads(p.read_text())["config"] for p in dev.glob("*/done.json")})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

if __name__=="__main__":main()
