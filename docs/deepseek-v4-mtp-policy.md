# DeepSeek V4 MTP tensor policy

Current policy: fail-closed.

DeepSeek V4 Flash checkpoint indexes include `mtp.*` tensors for multi-token prediction (MTP). The current MLX port does not have proof that those tensors are correctly implemented, safely ignored, or safely stripped. Therefore `mtp.*` tensors remain review-required blockers in the tensor mapping scanner.

## Explicit status

- MTP is not currently supported.
- MTP is not currently ignored.
- MTP is not currently stripped.
- `mtp.*` tensors are classified as `mtp.intentional-review-required`.
- Any report containing `mtp.intentional-review-required` must be treated as not OK.

Do not make mapping pass for `mtp.*` tensors unless one of the following is proven and documented with tests:

1. **Support MTP**: implement the required MTP modules and prove load/forward parity on tiny fixtures.
2. **Ignore MTP**: prove that the target MLX conversion/runtime path never needs these tensors and that ignoring them does not change required generation semantics.
3. **Strip MTP**: prove a deliberate strip path that rewrites/removes MTP tensors safely, updates metadata, and keeps downstream conversion/runtime behavior correct.

Until one of those proofs exists, `scan_tensor_names()` must include `mtp.*` names in `review_required`, attach the explicit MTP policy in `review_required_policy`, and keep `TensorMappingReport.ok == False`.

## Rationale

MTP tensors can affect model architecture or auxiliary prediction behavior. Silently accepting, dropping, or remapping them would make conversion appear successful without proving semantic compatibility. The safe default is to block conversion gates and force an explicit decision.

## Current reporting contract

For any `mtp.*` tensor, the mapping layer reports:

```text
family: mtp.intentional-review-required
status: review-required
action: fail-closed
supported: false
ignored: false
stripped: false
```

Future agents should update this document and the tests before changing that behavior.
