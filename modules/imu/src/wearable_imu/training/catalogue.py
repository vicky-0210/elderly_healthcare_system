"""Fixed primary candidates and controlled sensitivities. Test folds never select settings."""
from dataclasses import dataclass,field


@dataclass(frozen=True)
class Experiment:
    name:str
    model:str
    role:str="primary"
    view:str="ipmv"
    seconds:int=2
    overlap:float=.5
    scaler:str="robust"
    settings:dict=field(default_factory=dict)
    network:dict=field(default_factory=dict)
    epochs:int=5
    bag_size:int=8


def catalogue():
    base=("rf","svm","xgb","cnn","lstm","gru","bigru","tcn","inception","cnn_lstm","cnn_gru","cnn_bigru",
          "cnn_tcn","multicnn_tcn","cnn_transformer","cnn_ssm","cnn_channel","tcn_attention",
          "multitcn_attention","dual_cnn_tcn","dm_tcn","cross_modal")
    exams=[Experiment(n,n) for n in base]
    def add(n,model,**opts): exams.append(Experiment(n,model,role="ablation",**opts))
    for v in ("axes","axes_ipmv","impact","body","no_wave","no_body","no_peaks","sym4","wave2"):
        add("dm_"+v,"dm_tcn",view=v)
    for f in ("add","concat","attention"):
        add("dm_fuse_"+f,"dm_tcn",network={"fusion":f})
    for p in ("avg","max","avgmax"):
        add("dm_pool_"+p,"dm_tcn",network={"pool":p})
    add("dm_short_kernel","dm_tcn",network={"kernels":[3,5]})
    add("dm_shallow_dilation","dm_tcn",network={"dilations":[1,2]})
    add("dm_standard","dm_tcn",scaler="standard")
    add("dm_longer","dm_tcn",epochs=12)
    for v in ("axes","acc","gyro"): add("cnn_"+v,"cnn",view=v)
    for seconds in (1,2,3,5):
        for overlap in (0,.5,.75):
            if (seconds,overlap)!=(2,.5):
                add(f"grid_{seconds}s_{int(100*overlap)}","xgb",seconds=seconds,overlap=overlap)
    for n in (2,4):
        for hz in (20,30,40):
            add(f"lp{n}_{hz}","xgb",settings={"order":n,"acc_hz":hz,"gyro_hz":hz,"adaptive":False})
    add("raw_filter","xgb",settings={"acc_hz":0,"gyro_hz":0,"adaptive":False})
    for hz in (20,40):
        add(f"adaptive_{hz}","xgb",settings={"acc_hz":hz,"gyro_hz":hz})
    add("separate_20_40","xgb",settings={"acc_hz":20,"gyro_hz":40,"adaptive":False})
    add("separate_40_20","xgb",settings={"acc_hz":40,"gyro_hz":20,"adaptive":False})
    add("moving_gravity","xgb",settings={"gravity":"mean"})
    add("native_100","xgb",settings={"fs":100})
    return {s.name:s for s in exams}


def available():
    return {**catalogue(),"dm_distilled":Experiment("dm_distilled","dm_tcn",role="followup",scaler="standard")}
