using Microsoft.Extensions.Logging;
using Ssz.AI.Helpers;
using Ssz.Utils.Logging;
using System;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public sealed class BilingualLinearMapper
{
    public MatrixFloat A; // d x d
    public MatrixFloat B; // d x d
    
    public float[] MuRu;  // d
    public float[] MuEn;  // d

    public int Dim { get; }

    private ILoggersSet _loggersSet;

    public BilingualLinearMapper(int dim, ILoggersSet loggersSet)
    {
        Dim = dim;
        _loggersSet = loggersSet;
        A = new MatrixFloat(dim, dim);
        B = new MatrixFloat(dim, dim);
        MuRu = new float[dim];
        MuEn = new float[dim];
        SetIdentity(A);
        SetIdentity(B);        
    }

    // F12(x) = A * (x - MuRu) + MuEn
    public void F12(ReadOnlySpan<float> x, Span<float> y)
    {
        if (x.Length != Dim || y.Length != Dim)
            throw new ArgumentException("F12: vector length mismatch.");

        Span<float> xc = stackalloc float[Dim];
        Subtract(x, MuRu, xc);
        MatVec(A, xc, y);
        AddInPlace(y, MuEn);
    }

    // F21(y) = B * (y - MuEn) + MuRu
    public void F21(ReadOnlySpan<float> y, Span<float> x)
    {
        if (y.Length != Dim || x.Length != Dim)
            throw new ArgumentException("F21: vector length mismatch.");

        Span<float> yc = stackalloc float[Dim];
        Subtract(y, MuEn, yc);
        MatVec(B, yc, x);
        AddInPlace(x, MuRu);
    }

    public void ApplyF12(MatrixFloat src, MatrixFloat dst)
    {
        if (src.Rows() != Dim || dst.Rows() != Dim || dst.Cols() != src.Cols())
            throw new ArgumentException("ApplyF12: shape mismatch.");
        int n = src.Cols();
        for (int j = 0; j < n; j++)
        {
            var s = src.GetColumn(j);
            var d = dst.GetColumn(j);
            F12(s, d);
        }
    }

    public void ApplyF21(MatrixFloat src, MatrixFloat dst)
    {
        if (src.Rows() != Dim || dst.Rows() != Dim || dst.Cols() != src.Cols())
            throw new ArgumentException("ApplyF21: shape mismatch.");
        int n = src.Cols();
        for (int j = 0; j < n; j++)
        {
            var s = src.GetColumn(j);
            var d = dst.GetColumn(j);
            F21(s, d);
        }
    }

    public sealed class TrainOptions
    {
        public int Epochs = 30;
        public int BatchSize = 512;
        public float LearningRate = 0.05f;

        public float WCycle = 1.0f;
        public float WOrth = 0.1f;
        public float WMmd = 1.0f;

        public float MmdSigma = 1.0f;
        public int Seed = 42;
    }

    public void Fit(MatrixFloat ruEmb, MatrixFloat enEmb, TrainOptions? opt = null)
    {
        opt ??= new TrainOptions();
        if (ruEmb.Rows() != Dim || enEmb.Rows() != Dim)
            throw new ArgumentException("Embeddings must have shape (dim x n).");

        ComputeMean(ruEmb, MuRu);
        ComputeMean(enEmb, MuEn);

        var Xc = ruEmb.Clone();
        var Yc = enEmb.Clone();
        CenterColumnsInPlace(Xc, MuRu);
        CenterColumnsInPlace(Yc, MuEn);

        SetIdentity(A);
        SetIdentity(B);

        var rng = new Random(opt.Seed);
        int nRu = Xc.Cols();
        int nEn = Yc.Cols();
        int b = Math.Min(opt.BatchSize, Math.Min(nRu, nEn));

        var Xb = new MatrixFloat(Dim, b);
        var Yb = new MatrixFloat(Dim, b);
        var AX = new MatrixFloat(Dim, b);
        var BY = new MatrixFloat(Dim, b);
        var BAX = new MatrixFloat(Dim, b);
        var ABY = new MatrixFloat(Dim, b);

        var gA = new MatrixFloat(Dim, Dim);
        var gB = new MatrixFloat(Dim, Dim);
        var tmpDxN = new MatrixFloat(Dim, b);
        var tmpDxD = new MatrixFloat(Dim, Dim);

        var AtA = new MatrixFloat(Dim, Dim);
        var BtB = new MatrixFloat(Dim, Dim);
        var I = new MatrixFloat(Dim, Dim);
        SetIdentity(I);

        var gZ_A = new MatrixFloat(Dim, b);
        var gZ_B = new MatrixFloat(Dim, b);

        int stepsPerEpoch = Math.Max(1, Math.Min(nRu, nEn) / b);

        for (int epoch = 0; epoch < opt.Epochs; epoch++)
        {
            for (int step = 0; step < stepsPerEpoch; step++)
            {
                SampleColumns(rng, Xc, Xb);
                SampleColumns(rng, Yc, Yb);

                MatMat(A, Xb, AX);   // (d x d) * (d x b) = (d x b)
                MatMat(B, Yb, BY);   // (d x d) * (d x b) = (d x b)
                MatMat(B, AX, BAX);  // (d x d) * (d x b) = (d x b)
                MatMat(A, BY, ABY);  // (d x d) * (d x b) = (d x b)

                gA.Clear();
                gB.Clear();

                // Циклическая часть
                SubtractInPlace(BAX, Xb);                       // E_x
                MatMatT_LeftT(B, BAX, tmpDxN);                  // tmp = B^T * E_x (d x b)
                AccumulateOuter(tmpDxN, Xb, gA, scale: 2.0f);   // gA += 2 * tmp * Xb^T

                AccumulateOuter(BAX, AX, gB, scale: 2.0f);      // gB += 2 * E_x * (AX)^T

                SubtractInPlace(ABY, Yb);                       // E_y
                AccumulateOuter(ABY, BY, gA, scale: 2.0f);      // gA += 2 * E_y * (BY)^T

                MatMatT_LeftT(A, ABY, tmpDxN);                  // tmp = A^T * E_y
                AccumulateOuter(tmpDxN, Yb, gB, scale: 2.0f);   // gB += 2 * tmp * Yb^T

                // MMD градиенты
                gZ_A.Clear();
                gZ_B.Clear();
                ComputeMmdGradZ(AX, Yb, opt.MmdSigma, gZ_A);    // dL/d(AX)
                ComputeMmdGradZ(BY, Xb, opt.MmdSigma, gZ_B);    // dL/d(BY)
                AccumulateOuter(gZ_A, Xb, gA, scale: opt.WMmd);
                AccumulateOuter(gZ_B, Yb, gB, scale: opt.WMmd);

                // Ортогональность: 4 * A (A^T A - I), 4 * B (B^T B - I)
                MatMatT_RightT(A, A, AtA);   // AtA = A^T * A
                SubtractInPlace(AtA, I);
                MatMat(A, AtA, tmpDxD);
                AddScaledInPlace(gA, tmpDxD, 4.0f * opt.WOrth);

                MatMatT_RightT(B, B, BtB);   // BtB = B^T * B
                SubtractInPlace(BtB, I);
                MatMat(B, BtB, tmpDxD);
                AddScaledInPlace(gB, tmpDxD, 4.0f * opt.WOrth);

                // Обновление
                ScaleInPlace(gA, opt.LearningRate / b);
                ScaleInPlace(gB, opt.LearningRate / b);
                SubtractInPlace(A, gA);
                SubtractInPlace(B, gB);
            }
            _loggersSet.UserFriendlyLogger.LogInformation($"Epoch done. {epoch}");
        }
    }

    // Утилиты

    static void SetIdentity(MatrixFloat M)
    {
        M.Clear();
        int d = Math.Min(M.Rows(), M.Cols());
        for (int i = 0; i < d; i++) M[i, i] = 1.0f;
    }

    static void ComputeMean(MatrixFloat X, Span<float> mean)
    {
        if (mean.Length != X.Rows()) throw new ArgumentException("mean length != rows");
        mean.Clear();
        int d = X.Rows();
        int n = X.Cols();
        for (int j = 0; j < n; j++)
        {
            var col = X.GetColumn(j);
            // mean += col
            for (int i = 0; i < d; i++) mean[i] += col[i];
        }
        float invN = 1.0f / Math.Max(1, n);
        for (int i = 0; i < d; i++) mean[i] *= invN;
    }

    static void CenterColumnsInPlace(MatrixFloat X, ReadOnlySpan<float> mean)
    {
        if (mean.Length != X.Rows()) throw new ArgumentException("mean length != rows");
        int n = X.Cols();
        for (int j = 0; j < n; j++)
        {
            var col = X.GetColumn(j);
            TensorPrimitives.Subtract(col, mean, col);
        }
    }

    // y = M * x, где M(rows x cols), x(cols), y(rows)
    static void MatVec(MatrixFloat M, ReadOnlySpan<float> x, Span<float> y)
    {
        if (x.Length != M.Cols() || y.Length != M.Rows())
            throw new ArgumentException("MatVec: shape mismatch.");
        y.Clear();
        int rows = M.Rows();
        int cols = M.Cols();
        for (int k = 0; k < cols; k++)
        {
            float alpha = x[k];
            if (alpha == 0) continue;
            var mcol = M.GetColumn(k); // длина rows
            for (int i = 0; i < rows; i++)
                y[i] += mcol[i] * alpha;
        }
    }

    // Z = M * X, где M(rows x k), X(k x n), Z(rows x n)
    static void MatMat(MatrixFloat M, MatrixFloat X, MatrixFloat Z)
    {
        int rows = M.Rows();
        int k = M.Cols();
        if (X.Rows() != k) throw new ArgumentException("MatMat: inner dim mismatch (M.Cols() != X.Rows()).");
        int n = X.Cols();
        if (Z.Rows() != rows || Z.Cols() != n) throw new ArgumentException("MatMat: output shape mismatch.");

        for (int j = 0; j < n; j++)
        {
            var x = X.GetColumn(j); // len k
            var z = Z.GetColumn(j); // len rows
            z.Clear();
            for (int c = 0; c < k; c++)
            {
                float alpha = x[c];
                if (alpha == 0) continue;
                var mcol = M.GetColumn(c); // len rows
                for (int i = 0; i < rows; i++)
                    z[i] += mcol[i] * alpha;
            }
        }
    }

    // tmp = L^T * R, L(r x k) -> L^T(k x r), R(r x n) => tmp(k x n)
    static void MatMatT_LeftT(MatrixFloat L, MatrixFloat R, MatrixFloat tmp)
    {
        int r = L.Rows();
        int k = L.Cols();
        if (R.Rows() != r) throw new ArgumentException("MatMatT_LeftT: inner dim mismatch (L.Rows() != R.Rows()).");
        int n = R.Cols();
        if (tmp.Rows() != k || tmp.Cols() != n) throw new ArgumentException("MatMatT_LeftT: output shape mismatch.");

        for (int j = 0; j < n; j++)
        {
            var rcol = R.GetColumn(j); // len r
            var tcol = tmp.GetColumn(j); // len k
            tcol.Clear();
            for (int i = 0; i < k; i++)
            {
                var lcoli = L.GetColumn(i); // len r == L[:,i]
                float s = 0f;
                for (int p = 0; p < r; p++)
                    s += lcoli[p] * rcol[p];
                tcol[i] = s;
            }
        }
    }

    // C = A^T * B, A(r x k) -> A^T(k x r), B(r x m) => C(k x m)
    static void MatMatT_RightT(MatrixFloat A, MatrixFloat B, MatrixFloat C)
    {
        int r = A.Rows();
        int k = A.Cols();
        if (B.Rows() != r) throw new ArgumentException("MatMatT_RightT: inner dim mismatch (A.Rows() != B.Rows()).");
        int m = B.Cols();
        if (C.Rows() != k || C.Cols() != m) throw new ArgumentException("MatMatT_RightT: output shape mismatch.");

        C.Clear();
        for (int j = 0; j < m; j++)
        {
            var bcol = B.GetColumn(j); // len r
            for (int i = 0; i < k; i++)
            {
                var acol_i = A.GetColumn(i); // len r
                float s = 0f;
                for (int p = 0; p < r; p++)
                    s += acol_i[p] * bcol[p];
                C[i, j] = s;
            }
        }
    }

    // G += scale * (U * V^T), U(ru x n), V(rv x n) => G(ru x rv)
    static void AccumulateOuter(MatrixFloat U, MatrixFloat V, MatrixFloat G, float scale)
    {
        int ru = U.Rows();
        int n = U.Cols();
        if (V.Cols() != n) throw new ArgumentException("AccumulateOuter: batch mismatch (U.Cols() != V.Cols()).");
        int rv = V.Rows();
        if (G.Rows() != ru || G.Cols() != rv) throw new ArgumentException("AccumulateOuter: output shape mismatch.");

        for (int j = 0; j < n; j++)
        {
            var u = U.GetColumn(j); // len ru
            var v = V.GetColumn(j); // len rv
            for (int i = 0; i < ru; i++)
            {
                float ui = u[i] * scale;
                if (ui == 0) continue;
                for (int k = 0; k < rv; k++)
                    G[i, k] += ui * v[k];
            }
        }
    }

    static void AddInPlace(Span<float> a, ReadOnlySpan<float> b)
    {
        TensorPrimitives.Add(a, b, a);
    }

    static void Subtract(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> dst)
    {
        TensorPrimitives.Subtract(a, b, dst);
    }

    static void SubtractInPlace(MatrixFloat A, MatrixFloat B)
    {
        if (A.Rows() != B.Rows() || A.Cols() != B.Cols())
            throw new ArgumentException("SubtractInPlace: shape mismatch.");
        TensorPrimitives.Subtract(A.Data, B.Data, A.Data);
    }

    static void AddScaledInPlace(MatrixFloat A, MatrixFloat B, float scale)
    {
        if (A.Rows() != B.Rows() || A.Cols() != B.Cols())
            throw new ArgumentException("AddScaledInPlace: shape mismatch.");
        int len = A.Data.Length;
        for (int i = 0; i < len; i++)
            A.Data[i] += scale * B.Data[i];
    }

    static void ScaleInPlace(MatrixFloat A, float scale)
    {
        TensorPrimitives.Multiply(A.Data, scale, A.Data);
    }

    static void SubtractInPlace(MatrixFloat A, MatrixFloat grad, bool checkShape = true)
    {
        if (checkShape && (A.Rows() != grad.Rows() || A.Cols() != grad.Cols()))
            throw new ArgumentException("SubtractInPlace(A, grad): shape mismatch.");
        TensorPrimitives.Subtract(A.Data, grad.Data, A.Data);
    }

    static void SampleColumns(Random rng, MatrixFloat src, MatrixFloat dst)
    {
        if (src.Rows() != dst.Rows()) throw new ArgumentException("SampleColumns: rows mismatch.");
        int n = src.Cols();
        int b = dst.Cols();
        for (int j = 0; j < b; j++)
        {
            int idx = rng.Next(n);
            var s = src.GetColumn(idx);
            var t = dst.GetColumn(j);
            s.CopyTo(t);
        }
    }

    static void ComputeMmdGradZ(MatrixFloat Z, MatrixFloat Y, float sigma, MatrixFloat gZ)
    {
        if (Z.Rows() != Y.Rows() || Z.Rows() != gZ.Rows() || Z.Cols() != gZ.Cols())
            throw new ArgumentException("ComputeMmdGradZ: shape mismatch.");

        int d = Z.Rows();
        int n = Z.Cols();
        int m = Y.Cols();

        float s2 = sigma * sigma;
        float inv2s2 = 1.0f / (2f * s2);
        float cZZ = (n > 1) ? 2.0f / (n * (n - 1)) : 0f;
        float cZY = (n > 0 && m > 0) ? -2.0f / (n * m) : 0f;

        for (int i = 0; i < n; i++)
        {
            var zi = Z.GetColumn(i);
            var g = gZ.GetColumn(i);

            if (cZZ != 0f)
            {
                for (int j = 0; j < n; j++)
                {
                    if (j == i) continue;
                    var zj = Z.GetColumn(j);
                    float dist2 = SquaredDistance(zi, zj);
                    float k = MathF.Exp(-dist2 * inv2s2);
                    float coef = cZZ * k * (2f / s2);
                    AccumulateDiffScaled(zi, zj, g, coef);
                }
            }

            if (cZY != 0f)
            {
                for (int j = 0; j < m; j++)
                {
                    var yj = Y.GetColumn(j);
                    float dist2 = SquaredDistance(zi, yj);
                    float k = MathF.Exp(-dist2 * inv2s2);
                    float coef = cZY * k * (2f / s2);
                    AccumulateDiffScaled(zi, yj, g, coef);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static float SquaredDistance(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float s = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            float d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void AccumulateDiffScaled(ReadOnlySpan<float> zi, ReadOnlySpan<float> zj, Span<float> g, float scale)
    {
        for (int k = 0; k < zi.Length; k++)
            g[k] += scale * (zi[k] - zj[k]);
    }

    static float MSE(MatrixFloat A, MatrixFloat B)
    {
        if (A.Rows() != B.Rows() || A.Cols() != B.Cols())
            throw new ArgumentException("MSE: shape mismatch.");
        float s = 0f;
        int len = A.Data.Length;
        for (int i = 0; i < len; i++)
        {
            float d = A.Data[i] - B.Data[i];
            s += d * d;
        }
        // среднее на вектор (по колонкам)
        return s / Math.Max(1, A.Cols());
    }
}