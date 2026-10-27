"""Correctness of the fused RMSNorm + INT8 Triton kernel against the fp32 PyTorch reference.
Needs a CUDA GPU and Triton (skipped otherwise, e.g. on the Windows side)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from servekernels import rmsnorm_int8 as K

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
# (rows, hidden): the model's hidden size, odd sizes, tiny sizes, one row, a size above a power of two, the largest supported
SHAPES = [(1, 1536), (7, 1536), (300, 1000), (5, 33), (3, 1), (4, 2), (17, 4096), (13, 2049), (2, 16384), (1025, 1536)]


def make(shape, dtype, seed=0, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(shape, device="cuda", generator=g) * scale).to(dtype)
    w = (torch.rand(shape[-1], device="cuda", generator=g) + 0.5).to(dtype)
    return x, w


def run(x, w, eps=1e-6):
    q, s = K.rmsnorm_int8(x, w, eps)
    qr, sr = K.reference(x, w, eps)
    return q, s, qr, sr


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_reference(shape, dtype):
    x, w = make(shape, dtype)
    q, s, qr, sr = run(x, w)
    assert q.dtype == torch.int8 and s.dtype == torch.float32
    assert q.shape == x.shape and s.shape == x.shape[:-1]
    torch.testing.assert_close(s, sr, rtol=1e-5, atol=0)
    diff = (q.int() - qr.int()).abs()
    assert diff.max().item() <= 1, "a code may differ by one only when fp32 rounding flips a .5 boundary"
    assert (diff > 0).float().mean().item() <= 1e-3


@pytest.mark.parametrize("dtype", DTYPES)
def test_dequantised_error_is_within_half_a_step(dtype):
    x, w = make((64, 1536), dtype, seed=3, scale=3.0)
    q, s = K.rmsnorm_int8(x, w)
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    err = (q.float() * s[:, None] - y).abs()
    assert (err <= s[:, None] * 0.5001 + 1e-7).all()


def test_batched_leading_dims_and_noncontiguous_inputs():
    x, w = make((2, 3, 1536), torch.float16)
    q, s = K.rmsnorm_int8(x, w)
    assert q.shape == (2, 3, 1536) and s.shape == (2, 3)
    q2, s2 = K.rmsnorm_int8(x.reshape(6, 1536), w)
    assert torch.equal(q.reshape(6, 1536), q2) and torch.equal(s.reshape(6), s2)
    wide, _ = make((10, 2048), torch.float16, seed=5)
    view = wide[:, :1536]                                   # row stride 2048, not N
    qv, sv = K.rmsnorm_int8(view, w)
    qc, sc = K.rmsnorm_int8(view.contiguous(), w)
    assert torch.equal(qv, qc) and torch.equal(sv, sc)
    strided = wide[:, ::2]                                  # last-dim stride 2 -> made contiguous
    qs, ss = K.rmsnorm_int8(strided, w[:1024])
    qs2, ss2 = K.rmsnorm_int8(strided.contiguous(), w[:1024])
    assert torch.equal(qs, qs2) and torch.equal(ss, ss2)


def test_empty_input():
    x, w = make((0, 1536), torch.float16)
    q, s = K.rmsnorm_int8(x, w)
    assert q.shape == (0, 1536) and s.shape == (0,)


class TestEdgeValues:
    def test_zero_row_gives_zero_codes_and_zero_scale(self):
        x, w = make((3, 1536), torch.float16)
        x[1].zero_()
        q, s = K.rmsnorm_int8(x, w)
        assert (q[1] == 0).all() and s[1].item() == 0.0
        qr, sr = K.reference(x, w)
        assert torch.equal(q[1], qr[1]) and s[0].item() > 0

    def test_constant_row_saturates_to_plus_or_minus_127(self):
        x = torch.full((2, 64), 3.0, device="cuda", dtype=torch.float16)
        x[1] = -3.0
        q, _ = K.rmsnorm_int8(x, torch.ones(64, device="cuda", dtype=torch.float16))
        assert (q[0] == 127).all() and (q[1] == -127).all()

    def test_codes_never_reach_minus_128(self):
        x, w = make((256, 1536), torch.float16, seed=9, scale=50.0)
        q, _ = K.rmsnorm_int8(x, w)
        assert q.min().item() >= -127 and q.max().item() <= 127

    def test_values_that_overflow_fp16_squares_are_fine_because_the_kernel_computes_in_fp32(self):
        x, w = make((8, 1536), torch.float16, seed=1, scale=3000.0)     # x^2 ~ 9e6 > fp16 max 65504
        q, s, qr, sr = run(x, w)
        assert torch.isfinite(s).all()
        torch.testing.assert_close(s, sr, rtol=1e-5, atol=0)
        assert (q.int() - qr.int()).abs().max().item() <= 1

    def test_subnormal_and_tiny_inputs(self):
        x, w = make((8, 1536), torch.float16, seed=2, scale=1e-6)     # mostly fp16 subnormals
        q, s, qr, sr = run(x, w, eps=1e-12)
        torch.testing.assert_close(s, sr, rtol=1e-4, atol=0)
        assert (q.int() - qr.int()).abs().max().item() <= 1

    def test_single_outlier_sets_the_scale_and_crushes_the_rest(self):
        x, w = make((1, 1536), torch.float16, seed=4)
        x[0, 7] = 4000.0
        q, s, qr, sr = run(x, w)
        torch.testing.assert_close(s, sr, rtol=1e-5, atol=0)
        assert q[0, 7].item() == 127 and (q[0].int() - qr[0].int()).abs().max().item() <= 1

    def test_eps_matters_for_a_zero_row_of_a_nonzero_weight(self):
        x, w = make((2, 128), torch.float16)
        x[0].zero_()
        q, s = K.rmsnorm_int8(x, w, 1e-6)
        assert s[0].item() == 0.0 and s[1].item() > 0

    def test_nan_in_one_row_does_not_leak_into_other_rows(self):
        x, w = make((3, 1536), torch.float16)
        x[1, 5] = float("nan")
        q, s = K.rmsnorm_int8(x, w)
        qr, sr = K.reference(x, w)
        assert torch.isnan(s[1])
        for r in (0, 2):
            assert torch.equal(q[r], qr[r]) or (q[r].int() - qr[r].int()).abs().max().item() <= 1
            torch.testing.assert_close(s[r], sr[r], rtol=1e-5, atol=0)


def test_rejects_bad_inputs():
    x, w = make((4, 64), torch.float16)
    with pytest.raises(ValueError):
        K.rmsnorm_int8(x.cpu(), w.cpu())
    with pytest.raises(ValueError):
        K.rmsnorm_int8(x, w[:63])
    with pytest.raises(TypeError):
        K.rmsnorm_int8(x.int(), w)
    big, wb = make((1, K.MAX_N + 1), torch.float16)
    with pytest.raises(ValueError):
        K.rmsnorm_int8(big, wb)


@pytest.mark.parametrize("shape,dtype", [((4, 1536), torch.float16), ((2, 3, 33), torch.bfloat16), ((1, 4096), torch.float32)])
def test_opcheck(shape, dtype):
    """torch.library.opcheck: schema, fake-tensor/meta agreement, aot_autograd and faketensor-cache checks, and no input mutation."""
    x, w = make(shape, dtype)
    torch.library.opcheck(torch.ops.serving.rmsnorm_int8, (x, w, 1e-6))
    torch.library.opcheck(torch.ops.serving.rmsnorm_int8, (x, w))


def test_works_under_torch_compile_via_the_registered_fake():
    x, w = make((16, 1536), torch.float16)

    def f(a, b):
        q, s = torch.ops.serving.rmsnorm_int8(a, b, 1e-6)
        return q.float() * s[:, None]

    compiled = torch.compile(f, fullgraph=True)
    torch.testing.assert_close(compiled(x, w), f(x, w))
