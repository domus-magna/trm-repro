# %% [code]
# %% [markdown]
"""
# TRM ARC-AGI-2 Inference Notebook

Paper-faithful Tiny Recursive Model (TRM) inference on the ARC Prize 2025 evaluation set.

**What this notebook does**

1. Installs prepackaged dependencies from the attached wheels dataset (no internet).
2. Unpacks a clean snapshot of the TinyRecursiveModels repo.
3. Builds the ARC evaluation dataset in TRM format (no augmentation).
4. Loads `model.ckpt` from `seconds0/trm-offline-wheels-py311`.
5. Runs the ARC evaluator to produce `submission.json` for leaderboard submission.

**Before you run**

- Attach these datasets:
  - `seconds0/trm-offline-wheels-py311`
  - `seconds0/trm-repo-clean`
  - `arc-prize-2025` (competition data)
- Switch the runtime to **GPU** (Settings → Accelerator → GPU).
- Execution order matters: run each cell sequentially.
"""

# %% [markdown]
"""
## 0. Offline bootstrap (install wheels & unpack repo)
"""

# %%
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import hashlib
import importlib

import base64
import numpy as np

LEGACY_BUILDER_BASE64 = (
    'ZnJvbSB0eXBpbmcgaW1wb3J0IExpc3QsIFR1cGxlLCBEaWN0CmZyb20gZGF0YWNsYXNzZXMgaW1w'
    'b3J0IGRhdGFjbGFzcwppbXBvcnQgb3MKaW1wb3J0IGpzb24KaW1wb3J0IGhhc2hsaWIKaW1wb3J0'
    'IG51bXB5IGFzIG5wCgpmcm9tIGFyZ2RhbnRpYyBpbXBvcnQgQXJnUGFyc2VyCmZyb20gcHlkYW50'
    'aWMgaW1wb3J0IEJhc2VNb2RlbAoKZnJvbSBkYXRhc2V0LmNvbW1vbiBpbXBvcnQgUHV6emxlRGF0'
    'YXNldE1ldGFkYXRhLCBkaWhlZHJhbF90cmFuc2Zvcm0sIGludmVyc2VfZGloZWRyYWxfdHJhbnNm'
    'b3JtCgoKY2xpID0gQXJnUGFyc2VyKCkKCgpjbGFzcyBEYXRhUHJvY2Vzc0NvbmZpZyhCYXNlTW9k'
    'ZWwpOgogICAgaW5wdXRfZmlsZV9wcmVmaXg6IHN0cgogICAgb3V0cHV0X2Rpcjogc3RyCiAgICBz'
    'dWJzZXRzOiBMaXN0W3N0cl0KICAgIHRlc3Rfc2V0X25hbWU6IHN0cgogICAgdGVzdF9zZXRfbmFt'
    'ZTI6IHN0ciA9ICJ5b3VyX3Rlc3Rfc2V0IgogICAgc2VlZDogaW50ID0gNDIKICAgIG51bV9hdWc6'
    'IGludCA9IDEwMDAKICAgIHB1enpsZV9pZGVudGlmaWVyc19zdGFydDogaW50ID0gMSAjIHN0YXJ0'
    'ID4gMSB0byBoYW5kbGUgbXVsdGlwbGUgZGF0YXNldHMKICAgIApBUkNNYXhHcmlkU2l6ZSA9IDMw'
    'CkFSQ0F1Z21lbnRSZXRyaWVzRmFjdG9yID0gNQoKUHV6emxlSWRTZXBhcmF0b3IgPSAifHx8Igog'
    'ICAgCgpAZGF0YWNsYXNzCmNsYXNzIEFSQ1B1enpsZToKICAgIGlkOiBzdHIKICAgIGV4YW1wbGVz'
    'OiBMaXN0W1R1cGxlW25wLm5kYXJyYXksIG5wLm5kYXJyYXldXQoKICAgIApkZWYgYXJjX2dyaWRf'
    'dG9fbnAoZ3JpZDogTGlzdFtMaXN0W2ludF1dKToKICAgIGFyciA9IG5wLmFycmF5KGdyaWQpCgog'
    'ICAgIyBTaGFwZSBjaGVjawogICAgYXNzZXJ0IGFyci5uZGltID09IDIKICAgIGFzc2VydCBhcnIu'
    'c2hhcGVbMF0gPD0gQVJDTWF4R3JpZFNpemUgYW5kIGFyci5zaGFwZVsxXSA8PSBBUkNNYXhHcmlk'
    'U2l6ZQogICAgIyBFbGVtZW50IGNoZWNrCiAgICBhc3NlcnQgbnAuYWxsKChhcnIgPj0gMCkgJiAo'
    'YXJyIDw9IDkpKQogICAgcmV0dXJuIGFyci5hc3R5cGUobnAudWludDgpCgoKZGVmIG5wX2dyaWRf'
    'dG9fc2VxX3RyYW5zbGF0aW9uYWxfYXVnbWVudChpbnA6IG5wLm5kYXJyYXksIG91dDogbnAubmRh'
    'cnJheSwgZG9fdHJhbnNsYXRpb246IGJvb2wpOgogICAgIyBQQUQ6IDAsIDxlb3M+OiAxLCBkaWdp'
    'dHM6IDIgLi4uIDExCiAgICAjIENvbXB1dGUgcmFuZG9tIHRvcC1sZWZ0IHBhZAogICAgaWYgZG9f'
    'dHJhbnNsYXRpb246CiAgICAgICAgcGFkX3IgPSBucC5yYW5kb20ucmFuZGludCgwLCBBUkNNYXhH'
    'cmlkU2l6ZSAtIG1heChpbnAuc2hhcGVbMF0sIG91dC5zaGFwZVswXSkgKyAxKQogICAgICAgIHBh'
    'ZF9jID0gbnAucmFuZG9tLnJhbmRpbnQoMCwgQVJDTWF4R3JpZFNpemUgLSBtYXgoaW5wLnNoYXBl'
    'WzFdLCBvdXQuc2hhcGVbMV0pICsgMSkKICAgIGVsc2U6CiAgICAgICAgcGFkX3IgPSBwYWRfYyA9'
    'IDAKCiAgICAjIFBhZCBncmlkCiAgICByZXN1bHQgPSBbXQogICAgZm9yIGdyaWQgaW4gW2lucCwg'
    'b3V0XToKICAgICAgICBucm93LCBuY29sID0gZ3JpZC5zaGFwZQogICAgICAgIGdyaWQgPSBucC5w'
    'YWQoZ3JpZCArIDIsICgocGFkX3IsIEFSQ01heEdyaWRTaXplIC0gcGFkX3IgLSBucm93KSwgKHBh'
    'ZF9jLCBBUkNNYXhHcmlkU2l6ZSAtIHBhZF9jIC0gbmNvbCkpLCBjb25zdGFudF92YWx1ZXM9MCkK'
    'CiAgICAgICAgIyBBZGQgPGVvcz4KICAgICAgICBlb3Nfcm93LCBlb3NfY29sID0gcGFkX3IgKyBu'
    'cm93LCBwYWRfYyArIG5jb2wKICAgICAgICBpZiBlb3Nfcm93IDwgQVJDTWF4R3JpZFNpemU6CiAg'
    'ICAgICAgICAgIGdyaWRbZW9zX3JvdywgcGFkX2M6ZW9zX2NvbF0gPSAxCiAgICAgICAgaWYgZW9z'
    'X2NvbCA8IEFSQ01heEdyaWRTaXplOgogICAgICAgICAgICBncmlkW3BhZF9yOmVvc19yb3csIGVv'
    'c19jb2xdID0gMQoKICAgICAgICByZXN1bHQuYXBwZW5kKGdyaWQuZmxhdHRlbigpKQoKICAgIHJl'
    'dHVybiByZXN1bHQKCgpkZWYgZ3JpZF9oYXNoKGdyaWQ6IG5wLm5kYXJyYXkpOgogICAgYXNzZXJ0'
    'IGdyaWQubmRpbSA9PSAyCiAgICBhc3NlcnQgZ3JpZC5kdHlwZSA9PSBucC51aW50OAoKICAgIGJ1'
    'ZmZlciA9IFt4LnRvX2J5dGVzKDEsIGJ5dGVvcmRlcj0nYmlnJykgZm9yIHggaW4gZ3JpZC5zaGFw'
    'ZV0KICAgIGJ1ZmZlci5hcHBlbmQoZ3JpZC50b2J5dGVzKCkpCiAgICAKICAgIHJldHVybiBoYXNo'
    'bGliLnNoYTI1NihiIiIuam9pbihidWZmZXIpKS5oZXhkaWdlc3QoKQoKCmRlZiBwdXp6bGVfaGFz'
    'aChwdXp6bGU6IGRpY3QpOgogICAgIyBIYXNoIHRoZSBwdXp6bGUgZm9yIGNoZWNraW5nIGVxdWl2'
    'YWxlbmNlCiAgICBoYXNoZXMgPSBbXQogICAgZm9yIGV4YW1wbGVfdHlwZSwgZXhhbXBsZSBpbiBw'
    'dXp6bGUuaXRlbXMoKToKICAgICAgICBmb3IgaW5wdXQsIGxhYmVsIGluIGV4YW1wbGUuZXhhbXBs'
    'ZXM6CiAgICAgICAgICAgIGhhc2hlcy5hcHBlbmQoZiJ7Z3JpZF9oYXNoKGlucHV0KX18e2dyaWRf'
    'aGFzaChsYWJlbCl9IikKICAgICAgICAgICAgCiAgICBoYXNoZXMuc29ydCgpCiAgICByZXR1cm4g'
    'aGFzaGxpYi5zaGEyNTYoInwiLmpvaW4oaGFzaGVzKS5lbmNvZGUoKSkuaGV4ZGlnZXN0KCkKCgpk'
    'ZWYgYXVnKG5hbWU6IHN0cik6CiAgICAjIEF1Z21lbnQgcGxhbgogICAgdHJhbnNfaWQgPSBucC5y'
    'YW5kb20ucmFuZGludCgwLCA4KQogICAgbWFwcGluZyA9IG5wLmNvbmNhdGVuYXRlKFtucC5hcmFu'
    'Z2UoMCwgMSwgZHR5cGU9bnAudWludDgpLCBucC5yYW5kb20ucGVybXV0YXRpb24obnAuYXJhbmdl'
    'KDEsIDEwLCBkdHlwZT1ucC51aW50OCkpXSkgICMgUGVybXV0ZSBjb2xvcnMsIEV4Y2x1ZGluZyAi'
    'MCIgKGJsYWNrKQogICAgCiAgICBuYW1lX3dpdGhfYXVnX3JlcHIgPSBmIntuYW1lfXtQdXp6bGVJ'
    'ZFNlcGFyYXRvcn10e3RyYW5zX2lkfXtQdXp6bGVJZFNlcGFyYXRvcn17Jycuam9pbihzdHIoeCkg'
    'Zm9yIHggaW4gbWFwcGluZyl9IgoKICAgIGRlZiBfbWFwX2dyaWQoZ3JpZDogbnAubmRhcnJheSk6'
    'CiAgICAgICAgcmV0dXJuIGRpaGVkcmFsX3RyYW5zZm9ybShtYXBwaW5nW2dyaWRdLCB0cmFuc19p'
    'ZCkKICAgIAogICAgcmV0dXJuIG5hbWVfd2l0aF9hdWdfcmVwciwgX21hcF9ncmlkCgoKZGVmIGlu'
    'dmVyc2VfYXVnKG5hbWU6IHN0cik6CiAgICAjIEludmVyc2UgdGhlICJhdWciIGZ1bmN0aW9uCiAg'
    'ICBpZiBQdXp6bGVJZFNlcGFyYXRvciBub3QgaW4gbmFtZToKICAgICAgICByZXR1cm4gbmFtZSwg'
    'bGFtYmRhIHg6IHgKCiAgICB0cmFuc19pZCwgcGVybSA9IG5hbWUuc3BsaXQoUHV6emxlSWRTZXBh'
    'cmF0b3IpWy0yOl0KICAgIHRyYW5zX2lkID0gaW50KHRyYW5zX2lkWzE6XSkgICMgUmVtb3ZlICJ0'
    'IiBsZXR0ZXIKICAgIGludl9wZXJtID0gbnAuYXJnc29ydChsaXN0KHBlcm0pKS5hc3R5cGUobnAu'
    'dWludDgpCiAgICAKICAgIGRlZiBfbWFwX2dyaWQoZ3JpZDogbnAubmRhcnJheSk6CiAgICAgICAg'
    'cmV0dXJuIGludl9wZXJtW2ludmVyc2VfZGloZWRyYWxfdHJhbnNmb3JtKGdyaWQsIHRyYW5zX2lk'
    'KV0KICAgIAogICAgcmV0dXJuIG5hbWUuc3BsaXQoUHV6emxlSWRTZXBhcmF0b3IpWzBdLCBfbWFw'
    'X2dyaWQKCgpkZWYgY29udmVydF9zaW5nbGVfYXJjX3B1enpsZShyZXN1bHRzOiBkaWN0LCBuYW1l'
    'OiBzdHIsIHB1enpsZTogZGljdCwgYXVnX2NvdW50OiBpbnQsIGRlc3RfbWFwcGluZzogRGljdFtz'
    'dHIsIFR1cGxlW3N0ciwgc3RyXV0pOgogICAgIyBDb252ZXJ0CiAgICBkZXN0cyA9IHNldChkZXN0'
    'X21hcHBpbmcudmFsdWVzKCkpCiAgICBjb252ZXJ0ZWQgPSB7ZGVzdDogQVJDUHV6emxlKG5hbWUs'
    'IFtdKSBmb3IgZGVzdCBpbiBkZXN0c30KICAgIGZvciBleGFtcGxlX3R5cGUsIGV4YW1wbGVzIGlu'
    'IHB1enpsZS5pdGVtcygpOgogICAgICAgICMgTWFwIHRvIHRhcmdldCBzcGxpdAogICAgICAgIGRl'
    'c3QgPSBkZXN0X21hcHBpbmdbZXhhbXBsZV90eXBlXQogICAgICAgIGNvbnZlcnRlZFtkZXN0XS5l'
    'eGFtcGxlcy5leHRlbmQoWyhhcmNfZ3JpZF90b19ucChleGFtcGxlWyJpbnB1dCJdKSwgYXJjX2dy'
    'aWRfdG9fbnAoZXhhbXBsZVsib3V0cHV0Il0pKSBmb3IgZXhhbXBsZSBpbiBleGFtcGxlc10pCgog'
    'ICAgZ3JvdXAgPSBbY29udmVydGVkXQogICAgCiAgICAjIEF1Z21lbnQKICAgIGlmIGF1Z19jb3Vu'
    'dCA+IDA6CiAgICAgICAgaGFzaGVzID0ge3B1enpsZV9oYXNoKGNvbnZlcnRlZCl9CgogICAgICAg'
    'IGZvciBfdHJpYWwgaW4gcmFuZ2UoQVJDQXVnbWVudFJldHJpZXNGYWN0b3IgKiBhdWdfY291bnQp'
    'OgogICAgICAgICAgICBhdWdfbmFtZSwgX21hcF9ncmlkID0gYXVnKG5hbWUpCgogICAgICAgICAg'
    'ICAjIENoZWNrIGR1cGxpY2F0ZQogICAgICAgICAgICBhdWdtZW50ZWQgPSB7ZGVzdDogQVJDUHV6'
    'emxlKGF1Z19uYW1lLCBbKF9tYXBfZ3JpZChpbnB1dCksIF9tYXBfZ3JpZChsYWJlbCkpIGZvciAo'
    'aW5wdXQsIGxhYmVsKSBpbiBwdXp6bGUuZXhhbXBsZXNdKSBmb3IgZGVzdCwgcHV6emxlIGluIGNv'
    'bnZlcnRlZC5pdGVtcygpfQogICAgICAgICAgICBoID0gcHV6emxlX2hhc2goYXVnbWVudGVkKQog'
    'ICAgICAgICAgICBpZiBoIG5vdCBpbiBoYXNoZXM6CiAgICAgICAgICAgICAgICBoYXNoZXMuYWRk'
    'KGgpCiAgICAgICAgICAgICAgICBncm91cC5hcHBlbmQoYXVnbWVudGVkKQogICAgICAgICAgICAg'
    'ICAgCiAgICAgICAgICAgIGlmIGxlbihncm91cCkgPj0gYXVnX2NvdW50ICsgMToKICAgICAgICAg'
    'ICAgICAgIGJyZWFrCiAgICAgICAgICAgIAogICAgICAgIGlmIGxlbihncm91cCkgPCBhdWdfY291'
    'bnQgKyAxOgogICAgICAgICAgICBwcmludCAoZiJbUHV6emxlIHtuYW1lfV0gYXVnbWVudGF0aW9u'
    'IG5vdCBmdWxsLCBvbmx5IHtsZW4oZ3JvdXApfSIpCgogICAgIyBBcHBlbmQKICAgIGZvciBkZXN0'
    'IGluIGRlc3RzOgogICAgICAgICMgQ29udmVydCB0aGUgZXhhbXBsZXMKICAgICAgICBkZXN0X3Nw'
    'bGl0LCBkZXN0X3NldCA9IGRlc3QKCiAgICAgICAgcmVzdWx0cy5zZXRkZWZhdWx0KGRlc3Rfc3Bs'
    'aXQsIHt9KQogICAgICAgIHJlc3VsdHNbZGVzdF9zcGxpdF0uc2V0ZGVmYXVsdChkZXN0X3NldCwg'
    'W10pCiAgICAgICAgcmVzdWx0c1tkZXN0X3NwbGl0XVtkZXN0X3NldF0uYXBwZW5kKFtjb252ZXJ0'
    'ZWRbZGVzdF0gZm9yIGNvbnZlcnRlZCBpbiBncm91cF0pCgoKZGVmIGxvYWRfcHV6emxlc19hcmNh'
    'Z2koY29uZmlnOiBEYXRhUHJvY2Vzc0NvbmZpZyk6CiAgICB0cmFpbl9leGFtcGxlc19kZXN0ID0g'
    'KCJ0cmFpbiIsICJhbGwiKQogICAgdGVzdF9leGFtcGxlc19tYXAgPSB7CiAgICAgICAgY29uZmln'
    'LnRlc3Rfc2V0X25hbWU6IFsoMS4wLCAoInRlc3QiLCAiYWxsIikpXSwKICAgICAgICBjb25maWcu'
    'dGVzdF9zZXRfbmFtZTI6IFsoMS4wLCAoInRlc3QiLCAiYWxsIikpXSwKICAgICAgICAiX2RlZmF1'
    'bHQiOiBbKDEuMCwgKCJ0cmFpbiIsICJhbGwiKSldCiAgICB9CiAgICAKICAgIHRlc3RfcHV6emxl'
    'cyA9IHt9CiAgICByZXN1bHRzID0ge30KCiAgICB0b3RhbF9wdXp6bGVzID0gMAogICAgZm9yIHN1'
    'YnNldF9uYW1lIGluIGNvbmZpZy5zdWJzZXRzOgogICAgICAgICMgTG9hZCBhbGwgcHV6emxlcyBp'
    'biB0aGlzIHN1YnNldAogICAgICAgIHdpdGggb3BlbihmIntjb25maWcuaW5wdXRfZmlsZV9wcmVm'
    'aXh9X3tzdWJzZXRfbmFtZX1fY2hhbGxlbmdlcy5qc29uIiwgInIiKSBhcyBmOgogICAgICAgICAg'
    'ICBwdXp6bGVzID0ganNvbi5sb2FkKGYpCgogICAgICAgIHNvbHNfZmlsZW5hbWUgPSBmIntjb25m'
    'aWcuaW5wdXRfZmlsZV9wcmVmaXh9X3tzdWJzZXRfbmFtZX1fc29sdXRpb25zLmpzb24iCiAgICAg'
    'ICAgaWYgb3MucGF0aC5pc2ZpbGUoc29sc19maWxlbmFtZSk6CiAgICAgICAgICAgIHdpdGggb3Bl'
    'bihzb2xzX2ZpbGVuYW1lLCAiciIpIGFzIGY6CiAgICAgICAgICAgICAgICBzb2xzID0ganNvbi5s'
    'b2FkKGYpCiAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgIGZvciBwdXp6bGVfaWQgaW4g'
    'cHV6emxlcy5rZXlzKCk6CiAgICAgICAgICAgICAgICAgICAgZm9yIGlkeCwgc29sX2dyaWQgaW4g'
    'ZW51bWVyYXRlKHNvbHNbcHV6emxlX2lkXSk6CiAgICAgICAgICAgICAgICAgICAgICAgIHB1enps'
    'ZXNbcHV6emxlX2lkXVsidGVzdCJdW2lkeF1bIm91dHB1dCJdID0gc29sX2dyaWQKICAgICAgICBl'
    'bHNlOgogICAgICAgICAgICAjIEZpbGwgd2l0aCBkdW1teQogICAgICAgICAgICBwcmludCAoZiJ7'
    'c3Vic2V0X25hbWV9IHNvbHV0aW9ucyBub3QgZm91bmQsIGZpbGxpbmcgd2l0aCBkdW1teSIpCgog'
    'ICAgICAgICAgICBmb3IgcHV6emxlX2lkLCBwdXp6bGUgaW4gcHV6emxlcy5pdGVtcygpOgogICAg'
    'ICAgICAgICAgICAgZm9yIGV4YW1wbGUgaW4gcHV6emxlWyJ0ZXN0Il06CiAgICAgICAgICAgICAg'
    'ICAgICAgZXhhbXBsZS5zZXRkZWZhdWx0KCJvdXRwdXQiLCBbWzBdXSkKCiAgICAgICAgIyBTaHVm'
    'ZmxlIHB1enpsZXMKICAgICAgICBwdXp6bGVzID0gbGlzdChwdXp6bGVzLml0ZW1zKCkpCiAgICAg'
    'ICAgbnAucmFuZG9tLnNodWZmbGUocHV6emxlcykKICAgICAgICAKICAgICAgICAjIEFzc2lnbiBi'
    'eSBmcmFjdGlvbgogICAgICAgIGZvciBpZHgsIChuYW1lLCBwdXp6bGUpIGluIGVudW1lcmF0ZShw'
    'dXp6bGVzKToKICAgICAgICAgICAgZnJhY3Rpb24gPSBpZHggLyBsZW4ocHV6emxlcykKICAgICAg'
    'ICAgICAgdGVzdF9leGFtcGxlc19kZXN0ID0gTm9uZQogICAgICAgICAgICBmb3IgZiwgZGVzdCBp'
    'biB0ZXN0X2V4YW1wbGVzX21hcC5nZXQoc3Vic2V0X25hbWUsIHRlc3RfZXhhbXBsZXNfbWFwWyJf'
    'ZGVmYXVsdCJdKToKICAgICAgICAgICAgICAgIGlmIGZyYWN0aW9uIDwgZjoKICAgICAgICAgICAg'
    'ICAgICAgICB0ZXN0X2V4YW1wbGVzX2Rlc3QgPSBkZXN0CiAgICAgICAgICAgICAgICAgICAgYnJl'
    'YWsKICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgYXNzZXJ0IHRlc3RfZXhhbXBsZXNf'
    'ZGVzdCBpcyBub3QgTm9uZQogICAgICAgICAgICAKICAgICAgICAgICAgaWYgdGVzdF9leGFtcGxl'
    'c19kZXN0WzBdID09ICJ0ZXN0IjoKICAgICAgICAgICAgICAgIHRlc3RfcHV6emxlc1tuYW1lXSA9'
    'IHB1enpsZQogICAgICAgICAgICAgICAgCiAgICAgICAgICAgIGNvbnZlcnRfc2luZ2xlX2FyY19w'
    'dXp6bGUocmVzdWx0cywgbmFtZSwgcHV6emxlLCBjb25maWcubnVtX2F1ZywgeyJ0cmFpbiI6IHRy'
    'YWluX2V4YW1wbGVzX2Rlc3QsICJ0ZXN0IjogdGVzdF9leGFtcGxlc19kZXN0fSkKICAgICAgICAg'
    'ICAgdG90YWxfcHV6emxlcyArPSAxCgogICAgcHJpbnQgKGYiVG90YWwgcHV6emxlczoge3RvdGFs'
    'X3B1enpsZXN9IikKICAgIHJldHVybiByZXN1bHRzLCB0ZXN0X3B1enpsZXMKCgpkZWYgY29udmVy'
    'dF9kYXRhc2V0KGNvbmZpZzogRGF0YVByb2Nlc3NDb25maWcpOgogICAgbnAucmFuZG9tLnNlZWQo'
    'Y29uZmlnLnNlZWQpCiAgICAKICAgICMgUmVhZCBkYXRhc2V0CiAgICBkYXRhLCB0ZXN0X3B1enps'
    'ZXMgPSBsb2FkX3B1enpsZXNfYXJjYWdpKGNvbmZpZykKICAgIAogICAgIyBNYXAgZ2xvYmFsIHB1'
    'enpsZSBpZGVudGlmaWVycwogICAgbnVtX2lkZW50aWZpZXJzID0gY29uZmlnLnB1enpsZV9pZGVu'
    'dGlmaWVyc19zdGFydCAgIyAwIGlzIGJsYW5rLCBzdGFydCBhdCAxCiAgICBpZGVudGlmaWVyX21h'
    'cCA9IHt9CiAgICBmb3Igc3BsaXRfbmFtZSwgc3BsaXQgaW4gZGF0YS5pdGVtcygpOgogICAgICAg'
    'IGZvciBzdWJzZXRfbmFtZSwgc3Vic2V0IGluIHNwbGl0Lml0ZW1zKCk6CiAgICAgICAgICAgIGZv'
    'ciBncm91cCBpbiBzdWJzZXQ6CiAgICAgICAgICAgICAgICBmb3IgcHV6emxlIGluIGdyb3VwOgog'
    'ICAgICAgICAgICAgICAgICAgIGlmIHB1enpsZS5pZCBub3QgaW4gaWRlbnRpZmllcl9tYXA6CiAg'
    'ICAgICAgICAgICAgICAgICAgICAgIGlkZW50aWZpZXJfbWFwW3B1enpsZS5pZF0gPSBudW1faWRl'
    'bnRpZmllcnMKICAgICAgICAgICAgICAgICAgICAgICAgbnVtX2lkZW50aWZpZXJzICs9IDEKICAg'
    'IHByaW50IChmIlRvdGFsIHB1enpsZSBJRHMgKGluY2x1ZGluZyA8Ymxhbms+KToge251bV9pZGVu'
    'dGlmaWVyc30iKQoKICAgICMgU2F2ZQogICAgZm9yIHNwbGl0X25hbWUsIHNwbGl0IGluIGRhdGEu'
    'aXRlbXMoKToKICAgICAgICBvcy5tYWtlZGlycyhvcy5wYXRoLmpvaW4oY29uZmlnLm91dHB1dF9k'
    'aXIsIHNwbGl0X25hbWUpLCBleGlzdF9vaz1UcnVlKQogICAgICAgIAogICAgICAgICMgVHJhbnNs'
    'YXRpb25hbCBhdWdtZW50YXRpb25zCiAgICAgICAgZW5hYmxlX3RyYW5zbGF0aW9uYWxfYXVnbWVu'
    'dCA9IHNwbGl0X25hbWUgPT0gInRyYWluIgoKICAgICAgICAjIFN0YXRpc3RpY3MKICAgICAgICB0'
    'b3RhbF9leGFtcGxlcyA9IDAKICAgICAgICB0b3RhbF9wdXp6bGVzID0gMAogICAgICAgIHRvdGFs'
    'X2dyb3VwcyA9IDAKICAgICAgICAKICAgICAgICBmb3Igc3Vic2V0X25hbWUsIHN1YnNldCBpbiBz'
    'cGxpdC5pdGVtcygpOiAjICJhbGwiIGlzIHRoZSBvbmx5IHN1YnNldAogICAgICAgICAgICAjIENv'
    'bnN0cnVjdCBzdWJzZXQKICAgICAgICAgICAgcmVzdWx0cyA9IHtrOiBbXSBmb3IgayBpbiBbImlu'
    'cHV0cyIsICJsYWJlbHMiLCAicHV6emxlX2lkZW50aWZpZXJzIiwgInB1enpsZV9pbmRpY2VzIiwg'
    'Imdyb3VwX2luZGljZXMiXX0KICAgICAgICAgICAgcmVzdWx0c1sicHV6emxlX2luZGljZXMiXS5h'
    'cHBlbmQoMCkKICAgICAgICAgICAgcmVzdWx0c1siZ3JvdXBfaW5kaWNlcyJdLmFwcGVuZCgwKQog'
    'ICAgICAgICAgICAKICAgICAgICAgICAgZXhhbXBsZV9pZCA9IDAKICAgICAgICAgICAgcHV6emxl'
    'X2lkID0gMAogICAgICAgICAgICAKICAgICAgICAgICAgZm9yIGdyb3VwIGluIHN1YnNldDoKICAg'
    'ICAgICAgICAgICAgIGZvciBwdXp6bGUgaW4gZ3JvdXA6CiAgICAgICAgICAgICAgICAgICAgIyBQ'
    'dXNoIHB1enpsZQogICAgICAgICAgICAgICAgICAgIG5vX2F1Z19pZCA9IG5wLnJhbmRvbS5yYW5k'
    'aW50KDAsIGxlbihwdXp6bGUuZXhhbXBsZXMpKQogICAgICAgICAgICAgICAgICAgIGZvciBfaWR4'
    'X2V4LCAoaW5wLCBvdXQpIGluIGVudW1lcmF0ZShwdXp6bGUuZXhhbXBsZXMpOgogICAgICAgICAg'
    'ICAgICAgICAgICAgICBpbnAsIG91dCA9IG5wX2dyaWRfdG9fc2VxX3RyYW5zbGF0aW9uYWxfYXVn'
    'bWVudChpbnAsIG91dCwgZG9fdHJhbnNsYXRpb249ZW5hYmxlX3RyYW5zbGF0aW9uYWxfYXVnbWVu'
    'dCBhbmQgX2lkeF9leCAhPSBub19hdWdfaWQpCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAK'
    'ICAgICAgICAgICAgICAgICAgICAgICAgcmVzdWx0c1siaW5wdXRzIl0uYXBwZW5kKGlucCkKICAg'
    'ICAgICAgICAgICAgICAgICAgICAgcmVzdWx0c1sibGFiZWxzIl0uYXBwZW5kKG91dCkKICAgICAg'
    'ICAgICAgICAgICAgICAgICAgZXhhbXBsZV9pZCArPSAxCiAgICAgICAgICAgICAgICAgICAgICAg'
    'IAogICAgICAgICAgICAgICAgICAgICAgICB0b3RhbF9leGFtcGxlcyArPSAxCgogICAgICAgICAg'
    'ICAgICAgICAgIHJlc3VsdHNbInB1enpsZV9pbmRpY2VzIl0uYXBwZW5kKGV4YW1wbGVfaWQpCiAg'
    'ICAgICAgICAgICAgICAgICAgcmVzdWx0c1sicHV6emxlX2lkZW50aWZpZXJzIl0uYXBwZW5kKGlk'
    'ZW50aWZpZXJfbWFwW3B1enpsZS5pZF0pCiAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAg'
    'ICAgICAgICAgcHV6emxlX2lkICs9IDEKICAgICAgICAgICAgICAgICAgICB0b3RhbF9wdXp6bGVz'
    'ICs9IDEKICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICMgUHVzaCBncm91cAog'
    'ICAgICAgICAgICAgICAgcmVzdWx0c1siZ3JvdXBfaW5kaWNlcyJdLmFwcGVuZChwdXp6bGVfaWQp'
    'CiAgICAgICAgICAgICAgICB0b3RhbF9ncm91cHMgKz0gMQogICAgICAgICAgICAKICAgICAgICAg'
    'ICAgZm9yIGssIHYgaW4gcmVzdWx0cy5pdGVtcygpOgogICAgICAgICAgICAgICAgaWYgayBpbiB7'
    'ImlucHV0cyIsICJsYWJlbHMifToKICAgICAgICAgICAgICAgICAgICB2ID0gbnAuc3RhY2sodiwg'
    'MCkKICAgICAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICAgICAgdiA9IG5wLmFycmF5'
    'KHYsIGR0eXBlPW5wLmludDMyKQogICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICBucC5z'
    'YXZlKG9zLnBhdGguam9pbihjb25maWcub3V0cHV0X2Rpciwgc3BsaXRfbmFtZSwgZiJ7c3Vic2V0'
    'X25hbWV9X197a30ubnB5IiksIHYpCiAgICAgICAgCiAgICAgICAgIyBNZXRhZGF0YQogICAgICAg'
    'IG1ldGFkYXRhID0gUHV6emxlRGF0YXNldE1ldGFkYXRhKAogICAgICAgICAgICBzZXFfbGVuPUFS'
    'Q01heEdyaWRTaXplICogQVJDTWF4R3JpZFNpemUsCiAgICAgICAgICAgIHZvY2FiX3NpemU9MTAg'
    'KyAyLCAgIyBQQUQgKyBFT1MgKyAiMCIgLi4uICI5IgogICAgICAgICAgICBwYWRfaWQ9MCwKICAg'
    'ICAgICAgICAgaWdub3JlX2xhYmVsX2lkPTAsCiAgICAgICAgICAgIGJsYW5rX2lkZW50aWZpZXJf'
    'aWQ9MCwKICAgICAgICAgICAgbnVtX3B1enpsZV9pZGVudGlmaWVycz1udW1faWRlbnRpZmllcnMs'
    'CiAgICAgICAgICAgIHRvdGFsX2dyb3Vwcz10b3RhbF9ncm91cHMsCiAgICAgICAgICAgIG1lYW5f'
    'cHV6emxlX2V4YW1wbGVzPXRvdGFsX2V4YW1wbGVzIC8gdG90YWxfcHV6emxlcywKICAgICAgICAg'
    'ICAgdG90YWxfcHV6emxlcz10b3RhbF9wdXp6bGVzLAogICAgICAgICAgICBzZXRzPWxpc3Qoc3Bs'
    'aXQua2V5cygpKQogICAgICAgICkKCiAgICAgICAgIyBTYXZlIG1ldGFkYXRhIGFzIEpTT04uCiAg'
    'ICAgICAgd2l0aCBvcGVuKG9zLnBhdGguam9pbihjb25maWcub3V0cHV0X2Rpciwgc3BsaXRfbmFt'
    'ZSwgImRhdGFzZXQuanNvbiIpLCAidyIpIGFzIGY6CiAgICAgICAgICAgIGpzb24uZHVtcChtZXRh'
    'ZGF0YS5tb2RlbF9kdW1wKCksIGYpCiAgICAgICAgICAgIAogICAgIyBTYXZlIElEcyBtYXBwaW5n'
    'CiAgICB3aXRoIG9wZW4ob3MucGF0aC5qb2luKGNvbmZpZy5vdXRwdXRfZGlyLCAiaWRlbnRpZmll'
    'cnMuanNvbiIpLCAidyIpIGFzIGY6CiAgICAgICAgaWRzX21hcHBpbmcgPSB7djogayBmb3Igaywg'
    'diBpbiBpZGVudGlmaWVyX21hcC5pdGVtcygpfQogICAgICAgIGpzb24uZHVtcChbaWRzX21hcHBp'
    'bmcuZ2V0KGksICI8Ymxhbms+IikgZm9yIGkgaW4gcmFuZ2UobnVtX2lkZW50aWZpZXJzKV0sIGYp'
    'CiAgICAKICAgICMgU2F2ZSBUZXN0IFB1enpsZXMKICAgIHdpdGggb3Blbihvcy5wYXRoLmpvaW4o'
    'Y29uZmlnLm91dHB1dF9kaXIsICJ0ZXN0X3B1enpsZXMuanNvbiIpLCAidyIpIGFzIGY6CiAgICAg'
    'ICAganNvbi5kdW1wKHRlc3RfcHV6emxlcywgZikKCgpAY2xpLmNvbW1hbmQoc2luZ2xldG9uPVRy'
    'dWUpCmRlZiBtYWluKGNvbmZpZzogRGF0YVByb2Nlc3NDb25maWcpOgogICAgY29udmVydF9kYXRh'
    'c2V0KGNvbmZpZykKCgppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOgogICAgY2xpKCkKCgoKCgoK'
    'CgoKCgoK'
)


def write_legacy_builder(target: Path) -> None:
    target.write_bytes(base64.b64decode(LEGACY_BUILDER_BASE64))

try:
    import torch.distributed as dist
except ImportError:
    dist = None

from packaging.version import Version, InvalidVersion

INPUT_ROOT = Path("/kaggle/input")


def resolve_dataset(
    primary_slug: str,
    filename: str | None = None,
    aliases: tuple[str, ...] = (),
    display_slug: str | None = None,
) -> Path:
    """Return path to an attached dataset (optionally a file within it)."""
    candidate_slugs = (primary_slug, *aliases)
    for slug in candidate_slugs:
        base = INPUT_ROOT / slug
        target = base / filename if filename is not None else base
        if target.exists():
            return target
    attached = ", ".join(sorted(p.name for p in INPUT_ROOT.iterdir()))
    missing = display_slug or primary_slug
    msg = f"Attach dataset {missing}. Currently mounted: {attached or 'none'}"
    raise FileNotFoundError(msg)


WHEELS_DATASET = resolve_dataset(
    "trm-offline-wheels-py311",
    display_slug="seconds0/trm-offline-wheels-py311",
    aliases=("seconds0-trm-offline-wheels-py311", "trm-offline-wheels"),
)
if WHEELS_DATASET.is_file():
    wheels_extract_dir = Path("/kaggle/working/trm_offline_wheels")
    if wheels_extract_dir.exists():
        shutil.rmtree(wheels_extract_dir)
    wheels_extract_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(WHEELS_DATASET, wheels_extract_dir)
    WHEELS = wheels_extract_dir
else:
    WHEELS = WHEELS_DATASET
try:
    WEIGHTS_DATASET = resolve_dataset(
        "trm-arc2-weights-trm-arc2-8gpu-step249575",
        display_slug="seconds0/trm-arc2-weights-trm-arc2-8gpu-step249575",
        aliases=(
            "seconds0-trm-arc2-weights-trm-arc2-8gpu-step249575",
            "trm-arc2-weights-trm_arc2_8gpu_step249575",
            "seconds0-trm-arc2-weights-trm-arc2-8gpu-step119432",
            "trm-arc2-weights-trm_arc2_8gpu_step119432",
            "seconds0-trm-arc2-weights-trm-arc2-8gpu-eval100",
            "trm-arc2-weights-trm_arc2_8gpu_eval100",
            "trm-arc2-weights-trm-arc2-8gpu-resume",
            "trm-arc2-weights",
        ),
    )
except FileNotFoundError:
    print("[WARN] Checkpoint dataset not attached; falling back to wheels dataset for weights.")
    WEIGHTS_DATASET = WHEELS

CHECKPOINT_MANIFEST = WEIGHTS_DATASET / "MANIFEST.txt"
CHECKPOINT_STEP = 249575
if CHECKPOINT_MANIFEST.exists():
    manifest: dict[str, str] = {}
    for line in CHECKPOINT_MANIFEST.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        manifest[key.strip()] = value.strip()
    if "CHECKPOINT_STEP" in manifest:
        try:
            CHECKPOINT_STEP = int(manifest["CHECKPOINT_STEP"])
        except ValueError:
            print(f"[WARN] Invalid CHECKPOINT_STEP in {CHECKPOINT_MANIFEST}: {manifest['CHECKPOINT_STEP']}")
    print(f"Checkpoint manifest: step={CHECKPOINT_STEP}")

REPO_DATASET = resolve_dataset(
    "trm-repo-clean",
    display_slug="seconds0/trm-repo-clean",
    aliases=("seconds0-trm-repo-clean",),
)
REPO_ZIP = REPO_DATASET / "TinyRecursiveModels_clean.zip"
REPO_DIR_CANDIDATE = REPO_DATASET / "TinyRecursiveModels_clean"
REPO_FROM_DIR = False
if not REPO_ZIP.exists():
    if REPO_DIR_CANDIDATE.exists():
        REPO_FROM_DIR = True
    else:
        zip_candidates = sorted(REPO_DATASET.glob("**/*.zip"))
        if len(zip_candidates) == 1:
            REPO_ZIP = zip_candidates[0]
        elif not zip_candidates:
            listing = "\n".join(sorted(p.name for p in REPO_DATASET.iterdir()))
            raise FileNotFoundError(
                f"No zip archive found in {REPO_DATASET}. Expected TinyRecursiveModels_clean.zip.\n"
                f"Contents: {listing or '[empty]'}"
            )
        else:
            names = ", ".join(str(z.relative_to(REPO_DATASET)) for z in zip_candidates)
            raise FileNotFoundError(
                f"Multiple zip archives found in {REPO_DATASET}: {names}. "
                "Set REPO_ZIP manually in the first cell."
            )
REPO_DIR = Path("/kaggle/working/TinyRecursiveModels")

try:
    import importlib.metadata as metadata
except ImportError:  # Python <3.8 fallback (not expected on Kaggle)
    import importlib_metadata as metadata  # type: ignore


def is_requirement_satisfied(requirement: str) -> bool:
    """Return True if the exact requirement is already installed."""
    name, _, version = requirement.partition("==")
    try:
        installed_version = metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    if not version:
        return True
    try:
        return Version(installed_version) >= Version(version)
    except InvalidVersion:
        return installed_version == version


def find_distribution(dist_name: str, version: str) -> Path | None:
    """Best-effort locate an offline wheel or sdist for dist_name==version."""
    normalized = dist_name.replace("-", "_")
    search_patterns = [
        f"{dist_name}-{version}-*.whl",
        f"{normalized}-{version}-*.whl",
        f"{dist_name}-{version}.whl",
        f"{normalized}-{version}.whl",
        f"{dist_name}-{version}.tar.gz",
        f"{normalized}-{version}.tar.gz",
        f"{dist_name}-{version}.zip",
        f"{normalized}-{version}.zip",
    ]
    for pattern in search_patterns:
        matches = list(WHEELS.rglob(pattern))
        if matches:
            return matches[0]
    dir_patterns = [
        f"{dist_name}-{version}",
        f"{normalized}-{version}",
    ]
    for pattern in dir_patterns:
        matches = list(WHEELS.rglob(pattern))
        dirs = [m for m in matches if m.is_dir()]
        if dirs:
            setup_dirs = [
                d for d in dirs
                if (d / "setup.py").exists()
                or (d / "pyproject.toml").exists()
                or (d / "setup.cfg").exists()
            ]
            if setup_dirs:
                setup_dirs.sort(key=lambda p: (len(p.parts), str(p)))
                return setup_dirs[0]
            dirs.sort(key=lambda p: (len(p.parts), str(p)))
            return dirs[0]
    return None


def pip_install_requirement(requirement: str, *, mandatory: bool = False, env: dict | None = None, extra_args: list[str] | None = None) -> None:
    """Install requirement from offline wheels, optionally skipping if missing."""
    if is_requirement_satisfied(requirement):
        print(f"{requirement} already satisfied; skipping install.")
        return
    name, _, version = requirement.partition("==")
    location = find_distribution(name, version)
    if location is None:
        message = f"[WARN] {requirement} not found in {WHEELS}; skipping install."
        if mandatory:
            available = []
            if WHEELS.exists():
                try:
                    # Show a handful of available artifacts to aid debugging
                    available = sorted(
                        str(p.relative_to(WHEELS))
                        for p in WHEELS.rglob("*.*")
                        if p.is_file()
                    )[:20]
                except Exception:
                    available = []
            raise FileNotFoundError(
                f"Missing offline artifact for {requirement} in {WHEELS}.\n"
                f"Sample contents: {available}"
            )
        print(message)
        return

    cmd: list[str]
    if location.is_dir():
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            str(location),
        ]
    elif location.suffix == ".whl":
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(WHEELS),
            requirement,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            str(location),
        ]
    if extra_args:
        cmd.extend(extra_args)
    print("Installing", requirement, "from", location.name)
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        if mandatory:
            raise
        print(f"[WARN] Installation failed for {requirement}: {exc}. Continuing with existing environment.")


pip_install_requirement("hydra-core==1.3.2")
pip_install_requirement("omegaconf==2.3.0")
pip_install_requirement("annotated-types==0.7.0")
pip_install_requirement("pydantic-core==2.20.1")
pip_install_requirement("pydantic==2.8.2")
pip_install_requirement("typing-extensions==4.15.0")
pip_install_requirement("typing-inspection==0.4.0")
pip_install_requirement("pydantic-settings==2.4.0")
pip_install_requirement("python-dotenv==1.0.1")
pip_install_requirement("argdantic==1.3.3")
pip_install_requirement("coolname==2.2.0")
pip_install_requirement("einops==0.8.0")
pip_install_requirement("numba==0.60.0")
pip_install_requirement("llvmlite==0.43.0")
pip_install_requirement("antlr4-python3-runtime==4.9.3")
try:
    pip_install_requirement(
        "adam-atan2==0.0.3",
        mandatory=True,
    )
except Exception as exc:
    # Fallback: provide a stub optimizer so evaluation can proceed without the extension
    import types
    import torch.optim as optim

    print("[WARN] adam-atan2 installation failed:", exc)
    print("[WARN] Falling back to torch.optim.Adam as AdamATan2 stub (evaluation only).")
    adam_stub = types.ModuleType("adam_atan2")

    class AdamATan2(optim.Adam):
        pass

    adam_stub.AdamATan2 = AdamATan2
    sys.modules["adam_atan2"] = adam_stub

if REPO_FROM_DIR:
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    shutil.copytree(REPO_DIR_CANDIDATE, REPO_DIR)
    print("Repository copied from dataset directory to", REPO_DIR)
else:
    REPO_DIR.mkdir(exist_ok=True)
    shutil.unpack_archive(REPO_ZIP, REPO_DIR, format="zip")
    print("Repository unpacked from", REPO_ZIP.name, "to", REPO_DIR)

if not (REPO_DIR / "dataset").exists():
    nested_candidates = [
        d for d in REPO_DIR.iterdir()
        if d.is_dir() and d.name.lower().startswith("tinyrecursivemodels")
    ]
    for nested in nested_candidates:
        print("Flattening nested repo directory", nested.name)
        for item in nested.iterdir():
            destination = REPO_DIR / item.name
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(item), REPO_DIR)
        shutil.rmtree(nested)
        if (REPO_DIR / "dataset").exists():
            break

if not (REPO_DIR / "dataset").exists():
    raise FileNotFoundError(f"dataset/ not found in {REPO_DIR}. Check the repo dataset attachment.")

print("Repo root entries:", sorted(p.name for p in REPO_DIR.iterdir())[:12])

identifier_mode = os.environ.get("ARC_IDENTIFIER_MODE", "legacy").lower()
use_legacy_identifiers = identifier_mode != "sorted"
if use_legacy_identifiers:
    target_builder = REPO_DIR / "dataset" / "build_arc_dataset.py"
    legacy_builder = REPO_DIR / "dataset" / "build_arc_dataset_legacy.py"
    if legacy_builder.exists():
        shutil.copy(legacy_builder, target_builder)
        print("Applying legacy identifier mapping (shuffled) to match legacy checkpoints.")
    else:
        write_legacy_builder(target_builder)
        print("Applying embedded legacy builder fallback to match legacy checkpoints.")
else:
    print("Using sorted identifier mapping for dataset build.")

# %% [markdown]
"""
## 1. Build ARC evaluation dataset
Converts competition JSON into TRM's expected format (no augmentation).
"""

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["ARC_EVAL_CANDIDATE_DEBUG"] = os.environ.get("ARC_EVAL_CANDIDATE_DEBUG", "10")
os.environ["ARC_EVAL_CANDIDATE_DEBUG_TOP"] = os.environ.get("ARC_EVAL_CANDIDATE_DEBUG_TOP", "5")
sys.path.append(str(REPO_DIR))
sys.path.append(str(REPO_DIR / "TinyRecursiveModels"))


def _apply_arc_sort_patch() -> None:
    arc_module = None
    import_errors = []
    for module_name in ("TinyRecursiveModels.evaluators.arc", "evaluators.arc"):
        try:
            arc_module = importlib.import_module(module_name)
            break
        except Exception as exc:  # pragma: no cover - diagnostic aid
            import_errors.append(f"{module_name}: {exc}")
    if arc_module is None:
        print(f"[WARN] Unable to patch ARC evaluator: {'; '.join(import_errors)}")
        return

    if getattr(arc_module, "_score_candidates_patch_applied", False):
        return

    def _score_candidates(candidate_stats):
        scored = []
        for h, stats in candidate_stats.items():
            count = stats[0]
            total_q = stats[1]
            avg_q = total_q / count if count else 0.0
            scored.append((h, avg_q, count))
        scored.sort(key=lambda item: (item[1], item[2]), reverse=True)
        return scored

    arc_module._score_candidates = _score_candidates  # type: ignore[attr-defined]
    arc_module._score_candidates_patch_applied = True  # type: ignore[attr-defined]
    print("[ARC PATCH] Updated candidate ranking to weight halt probability first.")

_apply_arc_sort_patch()


DATA_DIR = Path("/kaggle/working/arc_dataset")
DATA_DIR.mkdir(exist_ok=True)
ARC_DATA_ROOT = Path("/kaggle/input/arc-prize-2025")
ARC_PREFIX = str(ARC_DATA_ROOT / "arc-agi")
ARC_SUBSET = "evaluation"

print(f"Building {ARC_SUBSET} dataset …")
subprocess.run(
    [
        "python3",
        str(REPO_DIR / "dataset/build_arc_dataset.py"),
        "--input-file-prefix",
        ARC_PREFIX,
        "--output-dir",
        str(DATA_DIR),
        "--subsets",
        ARC_SUBSET,
        "--test-set-name",
        ARC_SUBSET,
        "--num-aug",
        "0",
    ],
    check=True,
    env={**os.environ, "PYTHONPATH": str(REPO_DIR)},
)

# %% [markdown]
"""
### 1.1 Dataset validators
Confirm the build targets the ARC competition evaluation split (120 puzzles).
"""

# %%
with open(DATA_DIR / "test_puzzles.json") as f:
    test_puzzles = json.load(f)

if isinstance(test_puzzles, dict):
    puzzle_ids = sorted(test_puzzles.keys())
else:
    puzzle_ids = sorted(str(pid) for pid in test_puzzles)

source_file = ARC_DATA_ROOT / f"arc-agi_{ARC_SUBSET}_challenges.json"
if not source_file.exists():
    raise FileNotFoundError(
        f"{source_file.name} missing from competition data. "
        "Confirm that the official ARC Prize dataset is attached."
    )

with open(source_file) as f:
    evaluation_source = json.load(f)

expected_eval_puzzles = len(evaluation_source)

with open(DATA_DIR / "test" / "dataset.json") as f:
    dataset_meta = json.load(f)

if dataset_meta["total_puzzles"] != expected_eval_puzzles:
    raise RuntimeError(
        f"Unexpected evaluation puzzle count: {dataset_meta['total_puzzles']} (expected {expected_eval_puzzles}). "
        "Check that arc-prize-2025/arc-agi is attached instead of the sample dataset."
    )

source_ids = sorted(evaluation_source.keys())
if puzzle_ids != source_ids:
    missing = sorted(set(source_ids) - set(puzzle_ids))
    extra = sorted(set(puzzle_ids) - set(source_ids))
    raise RuntimeError(
        "Evaluation puzzle IDs do not match the competition file.\n"
        f"Missing IDs: {missing[:5]}\nExtra IDs: {extra[:5]}"
    )

id_hash = hashlib.sha256("\n".join(source_ids).encode("utf-8")).hexdigest()
print(
    f"Validators: {ARC_SUBSET} split confirmed "
    f"({expected_eval_puzzles} puzzles, SHA256={id_hash})."
)

identifiers_candidates = [
    DATA_DIR / ARC_SUBSET / "identifiers.json",
    DATA_DIR / "identifiers.json",
    DATA_DIR / "test" / "identifiers.json",
]
identifiers_path = next((path for path in identifiers_candidates if path.exists()), None)
if identifiers_path is not None:
    identifiers_sha256 = hashlib.sha256(identifiers_path.read_bytes()).hexdigest()
    EXPECTED_IDENTIFIER_HASHES = {
        "legacy": {
            "evaluation": "c364837393c2428e40c6116692fb1b66bf011108ec9930475df306cd779bbfd1",
            "test": "9af3f07ab5c05320e2da99c85ad76086f7cbabe2159b5cf694da01aa7e33546f",
            "default": "c364837393c2428e40c6116692fb1b66bf011108ec9930475df306cd779bbfd1",
        },
        "sorted": {
            "default": "f3fe1a1f0b27b36fd53166ac17faf980e6c7ff9e73ee16d884095a6c860637a5",
        },
    }
    subset_hashes = EXPECTED_IDENTIFIER_HASHES["legacy" if use_legacy_identifiers else "sorted"]
    expected_hash = subset_hashes.get(ARC_SUBSET, subset_hashes.get("default"))
    print("ARC identifiers sha256:", identifiers_sha256, "(expected", expected_hash, ")")
    if identifiers_sha256 != expected_hash:
        raise RuntimeError(
            "Identifier mapping hash mismatch. Expected "
            f"{expected_hash}, got {identifiers_sha256}. "
            "Ensure the dataset builder matches the checkpoint you are evaluating."
        )
else:
    print("[WARN] identifiers.json missing; skipping identifier hash validation.")

# %% [markdown]
"""
## 2. Load TRM components and checkpoint
"""

# %%
import torch
from torch.utils.data import DataLoader

try:
    from TinyRecursiveModels.dataset.common import PuzzleDatasetMetadata
    from TinyRecursiveModels.pretrain import (
        ArchConfig,
        EvaluatorConfig,
        LossConfig,
        PretrainConfig,
        TrainState,
        create_evaluators,
        create_model,
        evaluate,
    )
    from TinyRecursiveModels.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
except ModuleNotFoundError:
    from dataset.common import PuzzleDatasetMetadata
    from pretrain import (
        ArchConfig,
        EvaluatorConfig,
        LossConfig,
        PretrainConfig,
        TrainState,
        create_evaluators,
        create_model,
        evaluate,
    )
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

if not torch.cuda.is_available():
    raise RuntimeError("GPU runtime required. Enable GPU for this notebook.")

if dist is not None and dist.is_available() and not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29400")
    dist.init_process_group("gloo", rank=0, world_size=1)

CHECKPOINT_PATH = WEIGHTS_DATASET / "model.ckpt"
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"model.ckpt missing from {WEIGHTS_DATASET}. "
        "Attach seconds0/trm-arc2-weights-trm-arc2-8gpu-step249575 (or matching alias) "
        "alongside the offline wheels dataset."
    )

checkpoint_state = torch.load(CHECKPOINT_PATH, map_location="cpu")
def _resolve_puzzle_key(state_dict: dict[str, "torch.Tensor"]) -> str:
    primary = "model.inner.puzzle_emb.weights"
    alternate = "_orig_mod.model.inner.puzzle_emb.weights"
    if primary in state_dict:
        return primary
    if alternate in state_dict:
        return alternate
    raise KeyError(f"Puzzle embedding weights not found in checkpoint keys: {list(state_dict)[:5]}")

puzzle_emb_key = _resolve_puzzle_key(checkpoint_state)
puzzle_vocab_size = checkpoint_state[puzzle_emb_key].shape[0]
ckpt_has_attention_keys = any("self_attn.qkv_proj.weight" in k for k in checkpoint_state)

def _strip_prefix(state_dict: dict[str, "torch.Tensor"], prefix: str = "_orig_mod.") -> dict[str, "torch.Tensor"]:
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items() }

normalized_checkpoint_state = _strip_prefix(checkpoint_state)
EVAL_SAVE_DIR = Path("/kaggle/working/trm_eval_outputs")
EVAL_SAVE_DIR.mkdir(exist_ok=True)

loss_cfg = LossConfig(
    name="losses@ACTLossHead",
    loss_type="stablemax_cross_entropy",
)
os.environ.setdefault("ARC_HALTING_MAX_STEPS", "448")
os.environ.setdefault("ARC_SAMPLING_COUNT", "8")
os.environ.setdefault("ARC_SAMPLING_TEMPERATURE", "2.0")
HALT_MAX_STEPS = int(os.environ["ARC_HALTING_MAX_STEPS"])
print(f"[ARC CONFIG] halt_max_steps={HALT_MAX_STEPS}")
ARC_SAMPLING_COUNT = int(os.environ["ARC_SAMPLING_COUNT"])
if ARC_SAMPLING_COUNT > 1:
    os.environ.setdefault("ARC_SAMPLING_MODE", "sample")
    print(f"[ARC CONFIG] sampling_count={ARC_SAMPLING_COUNT}")
else:
    print("[ARC CONFIG] sampling_count=1 (argmax)")
print(f"[ARC CONFIG] sampling_temperature={os.environ['ARC_SAMPLING_TEMPERATURE']}")

arch_cfg = ArchConfig(
    name="recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1",
    loss=loss_cfg,
    mlp_t=False,
    H_cycles=3,
    L_cycles=4,
    H_layers=0,
    L_layers=2,
    hidden_size=512,
    num_heads=8,
    expansion=4,
    puzzle_emb_ndim=512,
    puzzle_emb_len=16,
    forward_dtype="float32",
    pos_encodings="rope",
    halt_max_steps=HALT_MAX_STEPS,
    halt_exploration_prob=0.1,
    no_ACT_continue=True,
)
eval_cfg = EvaluatorConfig(
    name="arc@ARC",
    submission_K=2,
    pass_Ks=[1, 2, 5, 10, 100, 1000],
    aggregated_voting=True,
)

cfg = PretrainConfig(
    arch=arch_cfg,
    data_paths=[str(DATA_DIR)],
    data_paths_test=[str(DATA_DIR)],
    evaluators=[eval_cfg],
    global_batch_size=64,
    epochs=1,
    lr=1e-4,
    lr_min_ratio=1.0,
    lr_warmup_steps=1,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    puzzle_emb_lr=0.01,
    puzzle_emb_weight_decay=0.1,
    project_name="Arc2concept-aug-1000-ACT-torch",
    run_name="trm_arc2_8gpu_eval100",
    load_checkpoint=str(CHECKPOINT_PATH),
    checkpoint_path=str(EVAL_SAVE_DIR),
    checkpoint_every_eval=False,
    eval_interval=1,
    min_eval_interval=0,
    eval_save_outputs=[],
    ema=True,
    ema_rate=0.999,
    freeze_weights=False,
    seed=0,
)
cfg.load_checkpoint = None  # manual load to support remapped checkpoint keys

arch_summary = {
    "name": arch_cfg.name,
    "mlp_t": arch_cfg.mlp_t,
    "num_heads": arch_cfg.num_heads,
    "pos_encodings": arch_cfg.pos_encodings,
    "forward_dtype": arch_cfg.forward_dtype,
    "halt_max_steps": arch_cfg.halt_max_steps,
}
print("ARCH_CFG:", arch_summary)
print("CHECKPOINT_PATH:", CHECKPOINT_PATH)
print("CHECKPOINT_STEP:", CHECKPOINT_STEP)
print("CKPT_HAS_ATTENTION_KEYS:", ckpt_has_attention_keys)

train_dataset = PuzzleDataset(
    PuzzleDatasetConfig(
        seed=cfg.seed,
        dataset_paths=cfg.data_paths,
        global_batch_size=cfg.global_batch_size,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    ),
    split="train",
)
train_metadata: PuzzleDatasetMetadata = train_dataset.metadata
train_metadata.num_puzzle_identifiers = puzzle_vocab_size
del train_dataset

eval_dataset = PuzzleDataset(
    PuzzleDatasetConfig(
        seed=cfg.seed,
        dataset_paths=cfg.data_paths_test or cfg.data_paths,
        global_batch_size=cfg.global_batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    ),
    split="test",
)
eval_loader = DataLoader(eval_dataset, batch_size=None)

model, optimizers, optimizer_lrs = create_model(cfg, train_metadata, rank=0, world_size=1)
model.eval()

incompatible = model.load_state_dict(normalized_checkpoint_state, strict=False)
if incompatible.missing_keys:
    print(f"[WARN] Missing keys when loading checkpoint: {sorted(incompatible.missing_keys)[:5]}")
if incompatible.unexpected_keys:
    print(f"[WARN] Unexpected keys when loading checkpoint: {sorted(incompatible.unexpected_keys)[:5]}")
del checkpoint_state
del normalized_checkpoint_state

inner_model = getattr(model, "model", None)
loss_head = getattr(model, "loss", None)
model_classes = {
    "model": type(model).__name__,
    "inner": type(inner_model).__name__ if inner_model is not None else "None",
    "loss_head": type(loss_head).__name__ if loss_head is not None else "None",
}
print("MODEL_CLASSES:", model_classes)

train_state = TrainState(
    model=model,
    optimizers=optimizers,
    optimizer_lrs=optimizer_lrs,
    carry=None,
    step=CHECKPOINT_STEP,
    total_steps=0,
)

evaluators = create_evaluators(cfg, eval_dataset.metadata)
metrics = evaluate(
    cfg,
    train_state,
    eval_loader,
    eval_dataset.metadata,
    evaluators,
    rank=0,
    world_size=1,
    cpu_group=None,
)

submission_dirs = sorted(EVAL_SAVE_DIR.glob("evaluator_ARC_step_*"))
if not submission_dirs:
    raise FileNotFoundError("Expected evaluator output not found.")
submission_path = submission_dirs[-1] / "submission.json"
if not submission_path.exists():
    raise FileNotFoundError(f"{submission_path} missing.")

shutil.copy(submission_path, "/kaggle/working/submission.json")

print("Saved submission:", submission_path)
print("Copied to /kaggle/working/submission.json")
print("Evaluator metrics:")
def _json_default(obj):
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    return obj

print(json.dumps(metrics, indent=2, default=_json_default))

# %% [markdown]
"""
## 3. Housekeeping (keep only submission.json)
"""

# %%
cleanup_targets = [
    EVAL_SAVE_DIR,
    REPO_DIR,
    DATA_DIR,
    Path("/kaggle/working/trm_offline_wheels"),
]

for target in cleanup_targets:
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            try:
                target.unlink()
            except FileNotFoundError:
                pass

for leftover in Path("/kaggle/working").iterdir():
    if leftover.name in {"submission.json"}:
        continue
    if leftover.is_dir():
        shutil.rmtree(leftover, ignore_errors=True)
    else:
        try:
            leftover.unlink()
        except FileNotFoundError:
            pass

# %% [markdown]
