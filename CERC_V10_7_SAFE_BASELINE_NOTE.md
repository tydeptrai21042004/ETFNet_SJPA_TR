# CERC v10.7 — Safe residual consensus and baseline audit

## Unified initialization
For evidence fields `X_m`, CERC uses one score equation for every `M >= 1`:

`ell_{m,g}(p) = gamma q_{m,g} + s_global(X_m) + s_local(X_{m,g})(p)`.

The evidence weights are `softmax_m(ell)`.  `gamma`, the local output kernel,
and the global output layer are zero initialized.  Therefore the weights are
uniform for `M > 1` and exactly one for `M = 1`.  The relational transport
kernel is also zero initialized.  Thus, after common tensors are copied, CERC
is exactly the trained mean-fusion model at adaptation step 0, and exactly the
single-stream model when only one evidence field is present.

All zero-initialized CERC convolutions use explicit `nn.Parameter(torch.zeros)`
plus `F.conv2d`, so constructing CERC consumes no random numbers and cannot
shift downstream initialization.

## Added variable-cardinality baselines
The same backbone/head can now be evaluated with:

1. uniform mean evidence fusion;
2. elementwise max evidence fusion;
3. parameter-free RMS-energy softmax fusion;
4. learned shared global gate fusion;
5. DeepSets-style shared residual transform + mean pooling.

All five baselines accept one or more evidence fields and contain no dataset or
task names.

## Safe residual protocol
A strong baseline may be trained first.  CERC then copies every common tensor,
freezes the backbone/head including BatchNorm running statistics, and trains
only `backbone.relations.*`.  Step 0 remains a valid checkpoint.  The screening
script accepts an adapted checkpoint only when validation score improves by at
least `0.015`; otherwise it keeps exact baseline parity.

This is a non-degradation *screening protocol*, not a theorem that test accuracy
must improve on every future dataset.
