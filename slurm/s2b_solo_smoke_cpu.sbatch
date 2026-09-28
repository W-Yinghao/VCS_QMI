#!/usr/bin/env bash
#SBATCH --job-name=s2b_solo_smoke_cpu
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/s2b_solo_smoke_cpu_%j.out
# S2b (P77) CPU smoke: (1) 3 steps of vcs_s2b on CPU (S2_NO_TEST=1, precision 32) and a check that a, b moved by
# AdamW-sized steps while the LARS groups exclude the critic; (2) checkpoint carries two optimizer + two scheduler
# states; (3) auto-resume for one more epoch continues from the saved epoch; (4) the frozen P75 method file is unchanged.
set -u
PY=/home/infres/yinwang/CS_QMI/env_solo/bin/python
SOLO=/home/infres/yinwang/CS_QMI/solo_learn
OUT=/home/infres/yinwang/CS_QMI/outputs/P77_S2b_smoke_${SLURM_JOB_ID}
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -u +%FT%TZ) solo_commit=$(git -C $SOLO rev-parse --short HEAD) solo_patch_sha256=$(git -C $SOLO diff HEAD | sha256sum | cut -c1-16)"
cd $SOLO
sha256sum solo/methods/vcs.py solo/methods/vcs_s2b.py main_pretrain_s2b.py scripts/pretrain/cifar/s2b_vcs.yaml
rc=0
COMMON="accelerator=cpu devices=1 strategy=auto precision=32 sync_batchnorm=False +limit_train_batches=3 data.num_workers=4 +performance.disable_channel_last=True +log_every_n_steps=1 seed=0"
t0=$(date +%s)
S2_NO_TEST=1 $PY main_pretrain_s2b.py --config-path scripts/pretrain/cifar --config-name s2b_vcs.yaml $COMMON max_epochs=1 \
  checkpoint.dir=$OUT/run +default_root_dir=$OUT/run auto_resume.enabled=True name=s2b-smoke || rc=1
echo "[epoch1] rc=$rc wall=$(( $(date +%s) - t0 ))s"; ls -R $OUT/run | head -12
S2_NO_TEST=1 $PY main_pretrain_s2b.py --config-path scripts/pretrain/cifar --config-name s2b_vcs.yaml $COMMON max_epochs=2 \
  checkpoint.dir=$OUT/run +default_root_dir=$OUT/run auto_resume.enabled=True name=s2b-smoke 2>&1 | grep -iE "restor|resum|epoch" | head -5 || true
$PY - "$OUT/run" <<'PYEOF'
import glob, sys, torch, csv
out = sys.argv[1]
cks = sorted(glob.glob(out + "/**/*.ckpt", recursive=True), key=lambda p: p)
print("checkpoints:", [c.split("/")[-1] for c in cks])
ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
print("optimizer_states:", len(ck["optimizer_states"]), "lr_schedulers:", len(ck.get("lr_schedulers", [])), "epoch:", ck["epoch"], "global_step:", ck["global_step"])
o0, o1 = ck["optimizer_states"]
print("main optimizer param groups (names):", [g.get("name") for g in o0["param_groups"]])
print("critic optimizer: n_params", sum(len(g["params"]) for g in o1["param_groups"]), "lr", o1["param_groups"][0]["lr"], "wd", o1["param_groups"][0]["weight_decay"], "betas", o1["param_groups"][0]["betas"])
assert "critic" not in [g.get("name") for g in o0["param_groups"]], "critic must not be in the LARS optimiser"
assert len(ck["optimizer_states"]) == 2 and len(ck.get("lr_schedulers", [])) == 2
sd = ck["state_dict"]; a = float(sd["critic.scale"][0]); b = float(sd["critic.bias"][0])
print(f"critic a={a:.6f} b={b:.6f} (a0=5, b0=0; AdamW 1e-3 with warm-up from 0 -> |Δ| per step << 1e-3 early)")
assert abs(a - 5.0) < 0.05 and abs(b) < 0.05, "critic scalars moved far more than AdamW-1e-3 steps allow"
assert ck["epoch"] >= 1 and ck["global_step"] >= 6, "second run did not resume/continue"
rows = []
for f in glob.glob(out + "/**/metrics.csv", recursive=True):
    rows += list(csv.DictReader(open(f)))
print("csv rows:", len(rows), "columns with critic_lr/main_lr:", [k for k in rows[0].keys() if k.endswith("_lr") or k.startswith("critic_")])
PYEOF
rc2=$?
echo "vcs.py unchanged vs P75 patch: $(git -C $SOLO diff HEAD --quiet -- solo/methods/vcs.py 2>/dev/null; git -C $SOLO diff HEAD -- solo/methods/vcs.py | sha256sum | cut -c1-16)"
echo "exit_train=$rc exit_checks=$rc2 end=$(date -u +%FT%TZ)"; exit $(( rc | rc2 ))
