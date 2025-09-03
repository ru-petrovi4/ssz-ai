using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using MathNet.Numerics.LinearAlgebra; // NuGet: MathNet.Numerics
using MathNet.Numerics.LinearAlgebra.Single;
using Ssz.Utils.Logging;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public class BilingualAlignment
{
    private ILoggersSet _loggersSet;

    // Dimensions: d=300, N,M <= 20000
    private readonly int d;
    private readonly int kCsls;
    private readonly float betaOrtho;

    private readonly int advVocabCap;     // напр. 50000 частотных слов
    private readonly int advBatchSize;     // размер батча
    private readonly float lrDisc;         // шаг Дискриминатора
    private readonly float lrGen;          // шаг Генератора (W)
    private readonly float labelSmoothing; // s=0.2
    private readonly int advEpochs;


    public BilingualAlignment(
        ILoggersSet loggersSet,
        int d = 300, int kCsls = 10, float betaOrtho = 0.01f,
        int advVocabCap = 50000, int advBatchSize = 512,
        float lrDisc = 0.1f, float lrGen = 0.1f,
        float labelSmoothing = 0.2f, int advEpochs = 5)
    {
        _loggersSet = loggersSet;
        this.d = d;
        this.kCsls = kCsls;
        this.betaOrtho = betaOrtho;
        this.advVocabCap = advVocabCap;
        this.advBatchSize = advBatchSize;
        this.lrDisc = lrDisc;
        this.lrGen = lrGen;
        this.labelSmoothing = labelSmoothing;
        this.advEpochs = advEpochs;
    }

    // Normalize columns to unit L2
    public void NormalizeColumns(MatrixFloat A)
    {
        int rows = A.Dimensions[0];
        int cols = A.Dimensions[1];
        for (int j = 0; j < cols; j++)
        {
            var col = A.GetColumn(j);
            float norm = TensorPrimitives.Norm(col);
            if (norm > 0)
            {
                TensorPrimitives.Divide(col, norm, col);
            }
        }
    }

    // W: dxd, A: dxn => C = W*A
    public MatrixFloat Multiply(MatrixFloat W, MatrixFloat A)
    {
        int r = W.Dimensions[0]; // d
        int k = W.Dimensions[1]; // d[3]
        int n = A.Dimensions[0];
        var C = new MatrixFloat(new[] { r, n });
        // Column-major multiplication by columns
        var Wdata = W.Data.AsSpan();
        for (int j = 0; j < n; j++)
        {
            var a = A.GetColumn(j);
            var c = C.GetColumn(j);
            c.Clear();
            // c = W * a
            for (int t = 0; t < k; t++)
            {
                float a_t = a[t];
                if (a_t == 0) continue;
                var Wcol = new Span<float>(W.Data, t * r, r);
                // c += a_t * Wcol
                TensorPrimitives.AddMultiply(Wcol, a_t, c, c); // c = c + a_t * Wcol
            }
        }
        return C;
    }

    // C = A * B^T (A: dxn, B: dxm) => C: n x m (сходимся к CSLS, нужны косинусы)
    // Здесь вместо полной матрицы будем рассчитывать локально в kNN, чтобы экономить память.

    // Cosine similarity of two columns
    public static float Cosine(Span<float> x, Span<float> y)
    {
        float dot = TensorPrimitives.Dot(x, y);
        // x и y предполагаются нормализованы
        return dot;
    }

    // Orthonormalization step: W <- (1+β)W - β (W W^T) W
    public void Orthonormalize(MatrixFloat W)
    {
        int d = W.Dimensions[0];
        // Compute WWt = W * W^T => d x d
        var WWt = new float[d * d];
        // WWt[i, j] = sum_t W[i,t]*W[j,t]
        for (int t = 0; t < d; t++)
        {
            var wcol = new Span<float>(W.Data, t * d, d);
            for (int i = 0; i < d; i++)
            {
                float wi = wcol[i];
                if (wi == 0) continue;
                for (int j = 0; j < d; j++)
                {
                    WWt[i + j * d] += wi * wcol[j];
                }
            }
        }
        // T = (WWt) * W
        var T = new float[d * d];
        for (int col = 0; col < d; col++)
        {
            var Wc = new Span<float>(W.Data, col * d, d);
            var Tc = new Span<float>(T, col * d, d);
            for (int i = 0; i < d; i++)
            {
                float s = 0f;
                // Tc[i] = sum_j WWt[i,j] * W[j,col]
                for (int j = 0; j < d; j++)
                {
                    s += WWt[i + j * d] * Wc[j];
                }
                Tc[i] = s;
            }
        }
        // W = (1+β)W - β T
        for (int idx = 0; idx < W.Data.Length; idx++)
        {
            W.Data[idx] = (1f + betaOrtho) * W.Data[idx] - betaOrtho * T[idx];
        }
    }

    // Build kNN neighborhoods and mean similarities r_T(Wx) and r_S(y)
    // Returns arrays rT (for source mapped) and rS (for target)
    public (float[] rT, float[] rS, int[][] knnT, int[][] knnS) BuildCslsNeighborhoods(MatrixFloat WX, MatrixFloat Y)
    {
        int n = WX.Dimensions[1];
        int m = Y.Dimensions[1];
        var rT = new float[n];
        var rS = new float[m];
        var knnT = new int[n][];
        var knnS = new int[m][];

        // For each source col, find top-k cosine sims in Y
        for (int i = 0; i < n; i++)
        {
            var x = WX.GetColumn(i);
            var top = new (int j, float s)[kCsls];
            int filled = 0;
            for (int j = 0; j < m; j++)
            {
                var y = Y.GetColumn(j);
                float s = TensorPrimitives.Dot(x, y); // normed => cosine
                if (filled < kCsls)
                {
                    top[filled++] = (j, s);
                    if (filled == kCsls) Array.Sort(top, (a, b) => a.s.CompareTo(b.s));
                }
                //TODO
                //else if (s > top.s)
                //{
                //    top = (j, s);
                //    Array.Sort(top, (a, b) => a.s.CompareTo(b.s));
                //}
            }
            float mean = 0f;
            var ids = new int[kCsls];
            for (int t = 0; t < kCsls; t++)
            {
                mean += top[t].s;
                ids[t] = top[t].j;
            }
            rT[i] = mean / kCsls;
            knnT[i] = ids;
        }

        // For each target col, find top-k in WX
        for (int j = 0; j < m; j++)
        {
            var y = Y.GetColumn(j);
            var top = new (int i, float s)[kCsls];
            int filled = 0;
            for (int i = 0; i < n; i++)
            {
                var x = WX.GetColumn(i);
                float s = TensorPrimitives.Dot(x, y);
                if (filled < kCsls)
                {
                    top[filled++] = (i, s);
                    if (filled == kCsls) Array.Sort(top, (a, b) => a.s.CompareTo(b.s));
                }
                // TODO
                //else if (s > top.s)
                //{
                //    top = (i, s);
                //    Array.Sort(top, (a, b) => a.s.CompareTo(b.s));
                //}
            }
            float mean = 0f;
            var ids = new int[kCsls];
            for (int t = 0; t < kCsls; t++)
            {
                mean += top[t].s;
                ids[t] = top[t].i;
            }
            rS[j] = mean / kCsls;
            knnS[j] = ids;
        }

        return (rT, rS, knnT, knnS);
    }

    // CSLS score between mapped source i and target j given precomputed rT, rS
    private float CslsScore(Span<float> xi, Span<float> yj, float rTi, float rSj)
    {
        float cos = TensorPrimitives.Dot(xi, yj);
        return 2f * cos - rTi - rSj;
    }

    // Mutual NN dictionary by CSLS
    public List<(int iSrc, int jTgt)> BuildMutualCslsDictionary(MatrixFloat WX, MatrixFloat Y, float[] rT, float[] rS)
    {
        int n = WX.Dimensions[1];
        int m = Y.Dimensions[1];
        var bestTgtForSrc = new int[n];
        var bestSrcForTgt = new int[m];
        Array.Fill(bestTgtForSrc, -1);
        Array.Fill(bestSrcForTgt, -1);

        // Best target for each source
        for (int i = 0; i < n; i++)
        {
            var xi = WX.GetColumn(i);
            float best = float.NegativeInfinity;
            int bestJ = -1;
            for (int j = 0; j < m; j++)
            {
                var yj = Y.GetColumn(j);
                float s = CslsScore(xi, yj, rT[i], rS[j]);
                if (s > best)
                {
                    best = s;
                    bestJ = j;
                }
            }
            bestTgtForSrc[i] = bestJ;
        }

        // Best source for each target
        for (int j = 0; j < m; j++)
        {
            var yj = Y.GetColumn(j);
            float best = float.NegativeInfinity;
            int bestI = -1;
            for (int i = 0; i < n; i++)
            {
                var xi = WX.GetColumn(i);
                float s = CslsScore(xi, yj, rT[i], rS[j]);
                if (s > best)
                {
                    best = s;
                    bestI = i;
                }
            }
            bestSrcForTgt[j] = bestI;
        }

        // Mutual pairs
        var pairs = new List<(int, int)>();
        for (int i = 0; i < n; i++)
        {
            int j = bestTgtForSrc[i];
            if (j >= 0 && bestSrcForTgt[j] == i)
            {
                pairs.Add((i, j));
            }
        }
        return pairs;
    }

    // Procrustes: given pairs (i,j), compute W = argmin ||W X - Y|| with orthogonальным W => W = U V^T for SVD(Y X^T)
    public MatrixFloat Procrustes(MatrixFloat X, MatrixFloat Y, List<(int iSrc, int jTgt)> pairs)
    {
        // Build Xp, Yp as d x p (p = pairs.Count)
        int p = pairs.Count;
        var Xp = new float[d * p];
        var Yp = new float[d * p];
        for (int idx = 0; idx < p; idx++)
        {
            var (i, j) = pairs[idx];
            X.GetColumn(i).CopyTo(new Span<float>(Xp, idx * d, d));
            Y.GetColumn(j).CopyTo(new Span<float>(Yp, idx * d, d));
        }
        // Compute M = Yp * Xp^T => d x d
        var M = new float[d * d];
        for (int col = 0; col < d; col++)
        {
            // column 'col' of M
            for (int row = 0; row < d; row++)
            {
                float sum = 0f;
                for (int t = 0; t < p; t++)
                {
                    float y_rt = Yp[row + t * d];
                    float x_ct = Xp[col + t * d];
                    sum += y_rt * x_ct;
                }
                M[row + col * d] = sum;
            }
        }
        // SVD(M) = U * S * V^T
        var Md = DenseMatrix.OfColumnMajor(d, d, M);
        var svd = Md.Svd(computeVectors: true);
        var U = svd.U;
        var VT = svd.VT;
        var W = new MatrixFloat(new[] { d, d });
        // W = U * V^T
        var UVT = U * VT;
        var wcolmaj = UVT.ToColumnMajorArray();
        Array.Copy(wcolmaj, W.Data, W.Data.Length);
        return W;
    }

    // One refinement iteration: build WX, CSLS neighborhoods, mutual dict, Procrustes
    public MatrixFloat Refine(MatrixFloat X, MatrixFloat Y, MatrixFloat W)
    {
        var WX = Multiply(W, X);
        // Ensure cols normalized (after multiplication, norms may drift)
        NormalizeColumns(WX);
        var (rT, rS, _, _) = BuildCslsNeighborhoods(WX, Y);
        var dict = BuildMutualCslsDictionary(WX, Y, rT, rS);
        var Wnew = Procrustes(X, Y, dict);
        return Wnew;
    }

    // Translate: for source column i, find best target j by CSLS
    public int Translate(MatrixFloat X, MatrixFloat Y, MatrixFloat W, int iSrc, float[]? rT = null, float[]? rS = null)
    {
        var WX = Multiply(W, X);
        NormalizeColumns(WX);
        if (rT == null || rS == null)
        {
            var tmp = BuildCslsNeighborhoods(WX, Y);
            rT = tmp.rT; rS = tmp.rS;
        }
        var xi = WX.GetColumn(iSrc);
        int m = Y.Dimensions[1];
        int bestJ = -1;
        float best = float.NegativeInfinity;
        for (int j = 0; j < m; j++)
        {
            var yj = Y.GetColumn(j);
            float s = 2f * TensorPrimitives.Dot(xi, yj) - rT[iSrc] - rS[j];
            if (s > best) { best = s; bestJ = j; }
        }
        return bestJ;
    }
}
