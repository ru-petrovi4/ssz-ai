using Microsoft.Extensions.Logging;
using Ssz.AI.Helpers;
using Ssz.Utils.Logging;
using System;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public class BilingualMapper
{
    public readonly int D;
    public MatrixFloat W12; // ru -> en
    public MatrixFloat W21; // en -> ru


    private ILoggersSet _loggersSet;
    private readonly Adam opt12;
    private readonly Adam opt21;
    private readonly Random rng;

    public BilingualMapper(int dim, ILoggersSet loggersSet, int seed = 42)
    {
        _loggersSet = loggersSet;
        D = dim;
        W12 = new MatrixFloat(D, D);
        W21 = new MatrixFloat(D, D);
        InitWeights();
        opt12 = new Adam(W12.Data.Length);
        opt21 = new Adam(W21.Data.Length);
        rng = new Random(seed);
    }

    private void InitWeights()
    {
        // Инициализация: почти тождественные + слабый шум, W21 = W12^T для старта
        LinAlg.SetIdentity(W12);
        LinAlg.SetIdentity(W21);
        var tmp = new float[D];
        for (int j = 0; j < D; j++)
        {
            var col = W12.GetColumn(j);
            for (int i = 0; i < D; i++)
            {
                float noise = (float)(0.01 * (new Random(j * 7919 + i * 104729).NextDouble() - 0.5));
                col[i] += noise;
            }
        }
        // Симметризация на старте
        for (int j = 0; j < D; j++)
        {
            for (int i = 0; i < D; i++)
            {
                W21[i, j] = W12[j, i];
            }
        }
        // Легкая орто-проекция шага
        OrthoRetractionInPlace(W12, 0.01f);
        OrthoRetractionInPlace(W21, 0.01f);
    }

    public void NormalizeAndCenter(MatrixFloat X)
    {
        int n = X.Dimensions[1];
        var mean = new float[D];
        // L2-нормировка столбцов и накопление среднего
        for (int j = 0; j < n; j++)
        {
            var col = X.GetColumn(j);
            float nrm2 = (float)Math.Sqrt(LinAlg.Dot(col, col));
            if (nrm2 > 0) LinAlg.ScaleInPlace(col, 1f / nrm2);
            for (int i = 0; i < D; i++) mean[i] += col[i];
        }
        for (int i = 0; i < D; i++) mean[i] /= n;
        // Центрирование
        for (int j = 0; j < n; j++)
        {
            var col = X.GetColumn(j);
            for (int i = 0; i < D; i++) col[i] -= mean[i];
        }
    }

    public struct TrainOptions
    {
        public int Epochs;
        public int BatchSize;
        public float Lr;
        public float CycleWeight;     // вес L_cycle
        public float CoralWeight;     // вес выравнивания ковариаций
        public float MeanWeight;      // вес выравнивания средних
        public float OrthoWeight;     // вес ||W^T W - I||^2
        public float CoupleWeight;    // вес ||W21 - W12^T||^2
        public float OrthoRetraction; // шаг мягкой проекции на ортогональную группу
        public int RetractionEvery;   // каждые N шагов
    }

    public void Fit(MatrixFloat Xru, MatrixFloat Xen, TrainOptions opt)
    {
        int nru = Xru.Dimensions[1];
        int nen = Xen.Dimensions[1];

        // Буферы
        var x = new float[D];
        var y = new float[D];
        var z = new float[D];
        var t = new float[D];
        var muRu = new float[D];
        var muEn = new float[D];

        // Градиенты
        var g12 = new MatrixFloat(D, D);
        var g21 = new MatrixFloat(D, D);

        // Временные матрицы для регуляризаторов/CoRAL
        var M12 = new MatrixFloat(D, D);
        var M21 = new MatrixFloat(D, D);
        var T12 = new MatrixFloat(D, D);
        var T21 = new MatrixFloat(D, D);

        var covY = new MatrixFloat(D, D);
        var covTgt = new MatrixFloat(D, D);
        var Sry = new MatrixFloat(D, D);
        var Set = new MatrixFloat(D, D);

        var Cdiff = new MatrixFloat(D, D);
        var tmpGrad = new MatrixFloat(D, D);

        int steps = 0;

        for (int epoch = 0; epoch < opt.Epochs; epoch++)
        {
            // Перемешанные индексы
            var idxRu = ShuffledIndices(nru);
            var idxEn = ShuffledIndices(nen);

            int batches = Math.Max((nru + opt.BatchSize - 1) / opt.BatchSize,
                                   (nen + opt.BatchSize - 1) / opt.BatchSize);

            for (int b = 0; b < batches; b++)
            {
                g12.Clear();
                g21.Clear();

                // RU batch
                int startRu = b * opt.BatchSize;
                int endRu = Math.Min(nru, startRu + opt.BatchSize);
                int br = Math.Max(1, endRu - startRu);

                // EN batch
                int startEn = b * opt.BatchSize;
                int endEn = Math.Min(nen, startEn + opt.BatchSize);
                int be = Math.Max(1, endEn - startEn);

                // Средние по батчу
                Array.Clear(muRu, 0, D);
                Array.Clear(muEn, 0, D);
                for (int j = startRu; j < endRu; j++)
                {
                    var col = Xru.GetColumn(idxRu[j]);
                    LinAlg.AddInPlace(muRu.AsSpan(), col);
                }
                for (int j = 0; j < D; j++) muRu[j] /= br;

                for (int j = startEn; j < endEn; j++)
                {
                    var col = Xen.GetColumn(idxEn[j]);
                    LinAlg.AddInPlace(muEn.AsSpan(), col);
                }
                for (int j = 0; j < D; j++) muEn[j] /= be;

                // 1) Cycle-consistency градиенты
                // RU side: e = W21*(W12*x) - x
                for (int j = startRu; j < endRu; j++)
                {
                    var col = Xru.GetColumn(idxRu[j]);
                    LinAlg.Copy(col, x);

                    LinAlg.MatVec(W12, x, y);      // y = W12 x
                    LinAlg.MatVec(W21, y, z);      // z = W21 y
                    for (int i = 0; i < D; i++) z[i] -= x[i]; // e = z - x

                    // dW21 += 2/br * e y^T
                    LinAlg.OuterAdd(z.AsSpan(), y, g21, 2f / br * opt.CycleWeight);

                    // t = W21^T e
                    LinAlg.MatTVec(W21, z, t);
                    // dW12 += 2/br * t x^T
                    LinAlg.OuterAdd(t.AsSpan(), x, g12, 2f / br * opt.CycleWeight);
                }

                // EN side: e2 = W12*(W21*x) - x
                for (int j = startEn; j < endEn; j++)
                {
                    var col = Xen.GetColumn(idxEn[j]);
                    LinAlg.Copy(col, x);

                    LinAlg.MatVec(W21, x, y);      // y = W21 x
                    LinAlg.MatVec(W12, y, z);      // z = W12 y
                    for (int i = 0; i < D; i++) z[i] -= x[i];

                    // dW12 += 2/be * e2 y^T
                    LinAlg.OuterAdd(z.AsSpan(), y, g12, 2f / be * opt.CycleWeight);

                    // t = W12^T e2
                    LinAlg.MatTVec(W12, z, t);
                    // dW21 += 2/be * t x^T
                    LinAlg.OuterAdd(t.AsSpan(), x, g21, 2f / be * opt.CycleWeight);
                }

                // 2) Mean alignment: ||W12*muRu - muEn||^2 + ||W21*muEn - muRu||^2
                LinAlg.MatVec(W12, muRu, y);
                for (int i = 0; i < D; i++) y[i] -= muEn[i]; // e_mu12
                LinAlg.OuterAdd(y.AsSpan(), muRu, g12, 2f * opt.MeanWeight);

                LinAlg.MatVec(W21, muEn, y);
                for (int i = 0; i < D; i++) y[i] -= muRu[i]; // e_mu21
                LinAlg.OuterAdd(y.AsSpan(), muEn, g21, 2f * opt.MeanWeight);

                // 3) CORAL (ковариации) для RU->EN
                covY.Clear(); covTgt.Clear(); Sry.Clear();
                // Сначала накопим cov(Xen_c)
                for (int j = startEn; j < endEn; j++)
                {
                    var xe = Xen.GetColumn(idxEn[j]);
                    for (int i = 0; i < D; i++) t[i] = xe[i] - muEn[i]; // t = xe_c
                    LinAlg.OuterAdd(t.AsSpan(), t, covTgt, 1f);
                }
                if (be > 1) LinAlg.ScaleMatrixInPlace(covTgt, 1f / (be - 1));

                // Теперь Y = W12 * (x_ru - muRu), covY и S = sum(y * xr^T)
                for (int j = startRu; j < endRu; j++)
                {
                    var xr = Xru.GetColumn(idxRu[j]);
                    for (int i = 0; i < D; i++) x[i] = xr[i] - muRu[i]; // x = xr_c
                    LinAlg.MatVec(W12, x, y);
                    LinAlg.OuterAdd(y.AsSpan(), y, covY, 1f);
                    LinAlg.OuterAdd(y.AsSpan(), x, Sry, 1f);
                }
                if (br > 1) LinAlg.ScaleMatrixInPlace(covY, 1f / (br - 1));

                // Cdiff = covY - covTgt
                LinAlg.SubtractMatrices(covY, covTgt, Cdiff);

                // tmpGrad = (Cdiff) * (Sry) ; g12 += 4/(br-1) * CoralWeight * tmpGrad
                tmpGrad.Clear();
                LinAlg.MatMulAdd(Cdiff, Sry, tmpGrad, 1f);
                float coralScaleRu = (br > 1) ? (4f * opt.CoralWeight / (br - 1)) : 0f;
                LinAlg.ScaleMatrixInPlace(tmpGrad, coralScaleRu);
                LinAlg.AddMatricesInPlace(g12, tmpGrad);

                // 4) CORAL для EN->RU
                covY.Clear(); covTgt.Clear(); Set.Clear();
                // cov(Xru_c)
                for (int j = startRu; j < endRu; j++)
                {
                    var xr = Xru.GetColumn(idxRu[j]);
                    for (int i = 0; i < D; i++) t[i] = xr[i] - muRu[i]; // t = xr_c
                    LinAlg.OuterAdd(t.AsSpan(), t, covTgt, 1f);
                }
                if (br > 1) LinAlg.ScaleMatrixInPlace(covTgt, 1f / (br - 1));

                // Y = W21 * (x_en - muEn), covY и S = sum(y * xe_c^T)
                for (int j = startEn; j < endEn; j++)
                {
                    var xe = Xen.GetColumn(idxEn[j]);
                    for (int i = 0; i < D; i++) x[i] = xe[i] - muEn[i]; // x = xe_c
                    LinAlg.MatVec(W21, x, y);
                    LinAlg.OuterAdd(y.AsSpan(), y, covY, 1f);
                    LinAlg.OuterAdd(y.AsSpan(), x, Set, 1f);
                }
                if (be > 1) LinAlg.ScaleMatrixInPlace(covY, 1f / (be - 1));

                LinAlg.SubtractMatrices(covY, covTgt, Cdiff);
                tmpGrad.Clear();
                LinAlg.MatMulAdd(Cdiff, Set, tmpGrad, 1f);
                float coralScaleEn = (be > 1) ? (4f * opt.CoralWeight / (be - 1)) : 0f;
                LinAlg.ScaleMatrixInPlace(tmpGrad, coralScaleEn);
                LinAlg.AddMatricesInPlace(g21, tmpGrad);

                // 5) Ортогональность: grad += 4*lambda * W (W^T W - I)
                M12.Clear(); M21.Clear();
                LinAlg.Gram(W12, M12);
                LinAlg.SubIdentityInPlace(M12, 1f);
                // T12 = W12 * M12
                T12.Clear();
                LinAlg.MatMulAdd(W12, M12, T12, 1f);
                LinAlg.ScaleMatrixInPlace(T12, 4f * opt.OrthoWeight);
                LinAlg.AddMatricesInPlace(g12, T12);

                LinAlg.Gram(W21, M21);
                LinAlg.SubIdentityInPlace(M21, 1f);
                T21.Clear();
                LinAlg.MatMulAdd(W21, M21, T21, 1f);
                LinAlg.ScaleMatrixInPlace(T21, 4f * opt.OrthoWeight);
                LinAlg.AddMatricesInPlace(g21, T21);

                // 6) Связка W21 ≈ W12^T: ||W21 - W12^T||^2
                for (int j = 0; j < D; j++)
                {
                    for (int i = 0; i < D; i++)
                    {
                        float diff = W21[i, j] - W12[j, i];
                        g21[i, j] += 2f * opt.CoupleWeight * diff;
                        g12[j, i] += -2f * opt.CoupleWeight * diff; // эквивалентно 2*(W12 - W21^T)
                    }
                }

                // Adam шаг
                opt12.Step(W12.Data, g12.Data, opt.Lr);
                opt21.Step(W21.Data, g21.Data, opt.Lr);

                steps++;
                if (opt.RetractionEvery > 0 && (steps % opt.RetractionEvery) == 0)
                {
                    OrthoRetractionInPlace(W12, opt.OrthoRetraction);
                    OrthoRetractionInPlace(W21, opt.OrthoRetraction);
                }
            }
            
            if ((epoch % 10) == 0)
                _loggersSet.UserFriendlyLogger.LogInformation($"Epoch {epoch}");
        }
    }

    // Мягкая проекция на множество ортогональных матриц:
    // W <- (1+beta)W - beta * W (W^T W)
    private static void OrthoRetractionInPlace(MatrixFloat W, float beta)
    {
        if (beta <= 0f) return;
        int D = W.Dimensions[0];
        var G = new MatrixFloat(D, D);
        LinAlg.Gram(W, G);           // G = W^T W
        var WGt = new MatrixFloat(D, D);
        LinAlg.MatMulAdd(W, G, WGt, 1f); // WGt = W * G
                                         // W = (1+beta)W - beta*WGt
        for (int i = 0; i < W.Data.Length; i++)
            W.Data[i] = (1f + beta) * W.Data[i] - beta * WGt.Data[i];
    }

    public void ApplyF12(ReadOnlySpan<float> src, Span<float> dst) => LinAlg.MatVec(W12, src, dst);
    public void ApplyF21(ReadOnlySpan<float> src, Span<float> dst) => LinAlg.MatVec(W21, src, dst);

    public MatrixFloat MapAllF12(MatrixFloat X)
    {
        int n = X.Dimensions[1];
        var Y = new MatrixFloat(D, n);
        var x = new float[D];
        var y = new float[D];
        for (int j = 0; j < n; j++)
        {
            LinAlg.Copy(X.GetColumn(j), x);
            LinAlg.MatVec(W12, x, y);
            Y.GetColumn(j).CopyTo(y);
            for (int i = 0; i < D; i++) Y[i, j] = y[i];
        }
        return Y;
    }

    private static int[] ShuffledIndices(int n)
    {
        var rnd = new Random(123456);
        var idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        for (int i = n - 1; i > 0; i--)
        {
            int j = rnd.Next(i + 1);
            (idx[i], idx[j]) = (idx[j], idx[i]);
        }
        return idx;
    }
}
