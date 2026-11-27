import math, torch
from cam_live.nn.fast import FastPrior, FastBeliefState
from cam_live.predict.policy import SelectionCalibration, policy_weight, apply_floor, selection_probs
from cam_live.reward.inner import InnerConfig
from cam_live.reward.integrity import scale_normalise
torch.manual_seed(0)
# 1 NIG predictive variance = b/(a-1) (1+q)
p = FastPrior(4); st = FastBeliefState(p)
phi = torch.randn(4); phi = phi / phi.norm()
for y in (0.3, -0.2, 0.5): st.update(phi, y, "real")
post = st._post["real"]; m, e, n = st.predict(phi, "real")
q = float(phi.double() @ post["cov"] @ phi.double())
print("1 nig var", float(e + n), float(post["b"] / (post["a"] - 1) * (1 + q)))
# 2 calibration prior lambda
cal = SelectionCalibration(); c = cal.sample(20000, torch.Generator().manual_seed(1)); lam = policy_weight(c)
print("2 prior beta mean", c.mean(0).tolist(), "lambda mean", float(lam.mean()), "share>0", float((lam > 0).double().mean()))
# 3 floor
pr = apply_floor(torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64), torch.tensor([[True]*4]), 0.02)
print("3 floor", pr.tolist())
# 4 discount
ic = InnerConfig(); print("4 beta", ic.beta, math.log(2)/86400)
# 5 scale invariance
r = torch.randn(10, dtype=torch.float64); print("5 scale-inv", float((scale_normalise(3*r+2) - scale_normalise(r)).abs().max()))
