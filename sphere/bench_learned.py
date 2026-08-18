# Tokut -- a compiled controller for improvement loops.
# Copyright (C) 2026 devkancheti4-design
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.  See <https://www.gnu.org/licenses/> for details.

"""Does a LEARNED model, given the identical 21 events, generalise as well as a
141-character weightless law? Same data in, same 256 situations out."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,".")
from sphere.law import Law, MEASURED
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

LC = Law("C")
idx = {n:i for i,n in enumerate(LC.bits)}
X=[]; y=[]
for names, act in MEASURED:
    s = sum(1<<idx[n] for n in names)
    X.append([(s>>b)&1 for b in range(8)]); y.append(act)
X=np.array(X); y=np.array(y)
ALL = np.array([[(s>>b)&1 for b in range(8)] for s in range(256)])
GT  = np.array([LC.decide(s) for s in range(256)])          # the law's own ruling
novel = [17,19,22,68,72,82,104,132,136,150,160,196,224,0]   # seen in transfer, unseen in training

def params(m):
    if hasattr(m,"tree_"): return int(m.tree_.node_count)
    if hasattr(m,"coefs_"): return sum(c.size for c in m.coefs_)+sum(b.size for b in m.intercepts_)
    if hasattr(m,"coef_"): return m.coef_.size + m.intercept_.size
    if hasattr(m,"estimators_"):
        try: return sum(int(e.tree_.node_count) for e in np.ravel(m.estimators_))
        except Exception: return -1
    if hasattr(m,"_fit_X"): return m._fit_X.size
    return -1

MODELS = [
 ("DecisionTree",        DecisionTreeClassifier(random_state=0)),
 ("RandomForest x200",   RandomForestClassifier(n_estimators=200,random_state=0)),
 ("GradientBoosting",    GradientBoostingClassifier(random_state=0)),
 ("k-NN (k=1)",          KNeighborsClassifier(n_neighbors=1)),
 ("LogisticRegression",  LogisticRegression(max_iter=5000)),
 ("MLP 64x64",           MLPClassifier((64,64),max_iter=20000,random_state=0)),
 ("MLP 256x256x256",     MLPClassifier((256,256,256),max_iter=20000,random_state=0)),
]
print("trained on the SAME 21 events. scored against the law on all 256 situations.\n")
print("%-22s %-10s %-14s %-16s %s"%("model","params","21 events","all 256","the 14 novel"))
print("-"*82)
for name,m in MODELS:
    m.fit(X,y)
    tr = (m.predict(X)==y).sum()
    pr = m.predict(ALL)
    allacc = (pr==GT).sum()
    nov = sum(1 for s in novel if pr[s]==GT[s])
    print("%-22s %-10s %-14s %-16s %s"%(name, params(m), "%d/21"%tr,
          "%d/256 (%.0f%%)"%(allacc,100*allacc/256), "%d/14"%nov))
print("-"*82)
print("%-22s %-10s %-14s %-16s %s"%("LAW C (no weights)","0","21/21","256/256 (100%)","14/14"))


# --------------------------------------------------------------------------
# The non-circular arm: run each learned model as an ACTUAL CONTROLLER on the
# six adversarial engines. Solve rate is the score, not agreement with the law.
def deployment_arm():
    from sphere.transfer import load_domains, drive, make_brains
    from sphere.law import ACTS
    mods = load_domains()
    def brain_of(m):
        tbl = [ACTS[int(v)] for v in m.predict(ALL)]
        return lambda s: tbl[s]
    def score(brain):
        ok = tot = 0
        for mo in mods:
            for job in mo.jobs():
                tot += 1; ok += drive(mo, job, brain, LC, 20)["ok"]
        return ok, tot
    print("\nSIX ADVERSARIAL ENGINES -- learned controllers vs the weightless law\n")
    print("%-22s %s" % ("controller", "solved"))
    print("-" * 40)
    for name, m in MODELS:
        m.fit(X, y); ok, tot = score(brain_of(m))
        print("%-22s %d/%d" % (name, ok, tot))
    lawc, _ = make_brains()["LAW_C"]; ok, tot = score(lawc)
    print("-" * 40)
    print("%-22s %d/%d   <- 0 parameters" % ("LAW C", ok, tot))


if __name__ == "__main__":
    deployment_arm()
