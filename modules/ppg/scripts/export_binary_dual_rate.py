"""Refit the validation-selected dual-rate candidate and verify offline and streaming parity."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wearable_ppg.datasets.osasud import CACHE
from wearable_ppg.datasets.windows import spans
from wearable_ppg.evaluation.compare import join, read_scores
from wearable_ppg.evaluation.metrics import choose_threshold
from wearable_ppg.inference.predictor import PPGPredictor
from wearable_ppg.preprocessing.optical import NAMES
from wearable_ppg.training.dual_rate import REPORT_DIR, RUN_DIR, fit, load_data, window_batch
from wearable_ppg.training.study import SEED, calibration, norm_fit


class EventLogits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, views):
        return self.model(views)[:, :, :1]


def main():
    import onnx
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = json.loads((REPORT_DIR / "selection.json").read_text())["dual_candidate"]
    roots = [RUN_DIR / name / f"fold{k}" for k in range(5)]
    epochs = int(np.median([json.loads((r / "done.json").read_text())["epoch"] for r in roots]))
    oof = join([read_scores(r, "test") for r in roots])
    a, b = calibration(oof["y"], oof["logit"])
    threshold = choose_threshold(oof["y"], expit(a*oof["logit"]+b), oof["t"], oof["patient"], oof["arrival"])
    features, y, t, ids, ends, starts = load_data()
    center, scale = norm_fit(features, ends)
    x = torch.asinh((torch.as_tensor(features, device=device)-torch.as_tensor(center, device=device))
                    /torch.as_tensor(scale, device=device))
    starts = torch.as_tensor(starts, device=device)
    model, history, _ = fit(name, x, starts, y, ids, ends, None, SEED, epochs)
    model = model.cpu().eval()
    wrapper = EventLogits(model).eval()
    folder = Path("artifacts/ppg/exports") / REPORT_DIR.name / name
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / "weights.pt")
    rng = np.random.default_rng(671)
    samples = window_batch(x, starts, rng.choice(ends, 128, replace=False), model.context).cpu()
    torch.onnx.export(wrapper, samples[:1], str(folder / "model.onnx"), input_names=["views"],
                      output_names=["event_logits"], opset_version=17)
    onnx.checker.check_model(str(folder / "model.onnx"))
    meta = dict(model=name, task="binary_event", warmup=30, context=model.context, hop=5, features=NAMES,
                target="any physician-annotated apnea or hypopnea in the scored second", labels=["NONE","ANY_EVENT"],
                calibration_version="osasud_binary_dual_rate_v1_oof", calibration=[a, b], threshold=threshold,
                center=center.tolist(), scale=scale.tolist(), epochs=epochs, history=history,
                graph_input="standardized features plus final real-history validity bit")
    (folder / "runtime.json").write_text(json.dumps(meta, indent=2))
    runtime = PPGPredictor(folder)
    with torch.inference_mode():
        expected = wrapper(samples).numpy()
    actual = np.concatenate([runtime.session.run(None, {"views": v[None].numpy()})[0] for v in samples])
    np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=2e-4)

    errors = []
    for pid in (1, 15, 28):
        filename = f"p{pid:02}.npz"
        with np.load(CACHE / filename) as src, np.load(CACHE / "views" / filename) as cache:
            run = max(spans(src["t"], src["valid"]), key=len)[:180]
            raw, f = src["raw"][run].ravel(), cache["x"][run]
        norm = torch.from_numpy(np.arcsinh((f-center)/scale))
        e = np.arange(29, len(f), 5)
        inputs = window_batch(norm, torch.zeros(len(f),dtype=torch.long), e, model.context)
        with torch.inference_mode():
            predicted = expit(a*wrapper(inputs).squeeze(-1).numpy().ravel()+b)
        runtime.reset()
        rows = []
        for i in range(0, len(raw), 137):
            rows.extend(runtime.push(raw[i:i+137], i/80))
        scores = [r["p_event"] for r in rows]
        np.testing.assert_equal([r["timestamp"] for r in rows], (e[:, None]+np.arange(-4,1)).ravel())
        np.testing.assert_allclose(scores, predicted, atol=1e-5)
        errors.append(float(np.max(np.abs(np.asarray(scores)-predicted))))
    verified = dict(model=name, participants=30, epochs=epochs, parameters=sum(p.numel() for p in model.parameters()),
                    onnx_bytes=(folder/"model.onnx").stat().st_size, parity_blocks=len(samples),
                    max_logit_error=float(np.max(abs(actual-expected))), max_stream_probability_errors=errors)
    (REPORT_DIR / "export.json").write_text(json.dumps(verified, indent=2))
    print(verified)


if __name__ == "__main__":
    main()
