using Microsoft.Extensions.Logging;
using Ssz.AI.Helpers;
using Ssz.Utils.Logging;
using System;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

// == BilingualMapper: основной класс для поиска преобразований между пространствами ==
// Содержит всю логику обучения и применения преобразований
public class BilingualMapper
{
    public readonly int D;           // размерность эмбеддингов
    public MatrixFloat W12;          // матрица преобразования: ru -> en
    public MatrixFloat W21;          // матрица преобразования: en -> ru

    private readonly Adam opt12, opt21;  // оптимизаторы для каждого направления
    private ILoggersSet _loggersSet;

    // Конструктор: инициализация матриц, оптимизаторов, начальная проекция
    public BilingualMapper(int dim, ILoggersSet loggersSet)
    {
        D = dim;
        _loggersSet = loggersSet;
        W12 = new MatrixFloat(D, D);
        W21 = new MatrixFloat(D, D);
        InitWeights();
        opt12 = new Adam(W12.Data.Length);
        opt21 = new Adam(W21.Data.Length);
    }

    // Начальная инициализация весов:
    // почти единичные матрицы с шумом; W21 сразу делаем транспонированной W12
    private void InitWeights()
    {
        LinAlg.SetIdentity(W12);
        LinAlg.SetIdentity(W21);

        // Слабый случайный шум на элементы W12
        for (int j = 0; j < D; j++)
        {
            var col = W12.GetColumn(j);
            for (int i = 0; i < D; i++)
                col[i] += (float)(0.01 * (new Random(j * 7919 + i * 104729).NextDouble() - 0.5));
        }
        // W21 = W12^T
        for (int j = 0; j < D; j++)
            for (int i = 0; i < D; i++)
                W21[i, j] = W12[j, i];

        // Первая мягкая проекция на ортогональную группу (ускоряет сходимость)
        OrthoRetractionInPlace(W12, 0.01f);
        OrthoRetractionInPlace(W21, 0.01f);
    }    

    // Структура с параметрами обучения
    public struct TrainOptions
    {
        public int Epochs;           // количество эпох обучения
        public int BatchSize;        // сколько слов в одном батче
        public float Lr;             // шаг (learning rate) для Adam
        public float CycleWeight;    // вес цикла L_cycle
        public float CoralWeight;    // вес CORAL (совпадение ковариаций)
        public float MeanWeight;     // вес выравнивания средних
        public float OrthoWeight;    // вес ортогональности
        public float CoupleWeight;   // вес ||W21 - W12^T||^2
        public float OrthoRetraction;// шаг битого ортогонального проектора
        public int RetractionEvery;  // раз в N батчей делать ортогонализацию
    }

    // == ОБУЧЕНИЕ ФУНКЦИЙ ПРЕОБРАЗОВАНИЯ ==
    // Все регуляризаторы реализованы как производные для градиентов
    public void Fit(MatrixFloat Xru, MatrixFloat Xen, TrainOptions opt)
    {
        int nru = Xru.Dimensions[1], nen = Xen.Dimensions[1];
        int steps = 0;

        // Рабочие буферы для промежуточных вычислений
        float[] x = new float[D], y = new float[D], z = new float[D], t = new float[D];
        float[] muRu = new float[D], muEn = new float[D];

        var g12 = new MatrixFloat(D, D); // градиент W12
        var g21 = new MatrixFloat(D, D); // градиент W21

        // Матричные буферы для лоссов и регуляризаторов
        MatrixFloat M12 = new MatrixFloat(D, D), M21 = new MatrixFloat(D, D);
        MatrixFloat T12 = new MatrixFloat(D, D), T21 = new MatrixFloat(D, D);
        MatrixFloat covY = new MatrixFloat(D, D), covTgt = new MatrixFloat(D, D);
        MatrixFloat Sry = new MatrixFloat(D, D), Set = new MatrixFloat(D, D);
        MatrixFloat Cdiff = new MatrixFloat(D, D), tmpGrad = new MatrixFloat(D, D);

        // Основной цикл по эпохам
        for (int epoch = 0; epoch < opt.Epochs; epoch++)
        {
            // Перемешиваем индексы: это гарантирует случайное разбиение батчей для SGD
            int[] idxRu = ShuffledIndices(nru), idxEn = ShuffledIndices(nen);
            int batches = Math.Max((nru + opt.BatchSize - 1) / opt.BatchSize,
                                   (nen + opt.BatchSize - 1) / opt.BatchSize);
            
            // Цикл по батчам
            for (int b = 0; b < batches; b++)
            {
                g12.Clear();
                g21.Clear();

                // Определяем границы текущего батча в русских и английских эмбеддингах
                int startRu = b * opt.BatchSize, endRu = Math.Min(nru, startRu + opt.BatchSize);
                int startEn = b * opt.BatchSize, endEn = Math.Min(nen, startEn + opt.BatchSize);
                int br = Math.Max(1, endRu - startRu), be = Math.Max(1, endEn - startEn);

                // == 1. CYCLE-CONSISTENCY ==
                // Для русских эмбеддингов: применяем сначала W12, затем W21, сравниваем с исходным                
                for (int j = startRu; j < endRu; j++)
                {
                    var col = Xru.GetColumn(idxRu[j]);
                    LinAlg.Copy(col, x);

                    LinAlg.MatVec(W12, x, y);   // y = W12 x
                    LinAlg.MatVec(W21, y, z);   // z = W21 y -- результат обратного преобразования
                    TensorPrimitives.Subtract(z, x, z); // ошибка восстановления
                    
                    LinAlg.OuterAdd(z.AsSpan(), y, g21, 2f / br * opt.CycleWeight); // градиент W21
                    LinAlg.MatTVec(W21, z, t); // производная W12 через обратное восстановление
                    LinAlg.OuterAdd(t.AsSpan(), x, g12, 2f / br * opt.CycleWeight);
                }
                // Аналогично для английских
                for (int j = startEn; j < endEn; j++)
                {
                    var col = Xen.GetColumn(idxEn[j]);
                    LinAlg.Copy(col, x);

                    LinAlg.MatVec(W21, x, y);
                    LinAlg.MatVec(W12, y, z);
                    for (int i = 0; i < D; i++) z[i] -= x[i];

                    LinAlg.OuterAdd(z.AsSpan(), y, g12, 2f / be * opt.CycleWeight);
                    LinAlg.MatTVec(W12, z, t);
                    LinAlg.OuterAdd(t.AsSpan(), x, g21, 2f / be * opt.CycleWeight);
                }

                // == 2. ВЫРАВНИВАНИЕ СРЕДНИХ ==
                // muRu: средний эмбеддинг по батчу русских; muEn: по английским
                ComputeBatchMean(Xru, idxRu, startRu, endRu, muRu);
                ComputeBatchMean(Xen, idxEn, startEn, endEn, muEn);

                LinAlg.MatVec(W12, muRu, y);
                for (int i = 0; i < D; i++) y[i] -= muEn[i];    // ошибка по среднему
                LinAlg.OuterAdd(y.AsSpan(), muRu, g12, 2f * opt.MeanWeight);

                LinAlg.MatVec(W21, muEn, y);
                for (int i = 0; i < D; i++) y[i] -= muRu[i];
                LinAlg.OuterAdd(y.AsSpan(), muEn, g21, 2f * opt.MeanWeight);

                // == 3. CORAL: совпадение ковариаций исходных и преобразованных батчей ==
                // Считаем ковариации преобразований и целевых батчей
                BatchedCovarianceAndS(Xru, idxRu, startRu, endRu, muRu, W12, covY, Sry);
                BatchedTargetCovariance(Xen, idxEn, startEn, endEn, muEn, covTgt);
                LinAlg.SubtractMatrices(covY, covTgt, Cdiff);
                tmpGrad.Clear();
                LinAlg.MatMulAdd(Cdiff, Sry, tmpGrad, 1f);
                LinAlg.ScaleMatrixInPlace(tmpGrad, (br > 1) ? (4f * opt.CoralWeight / (br - 1)) : 0f);
                LinAlg.AddMatricesInPlace(g12, tmpGrad);

                BatchedCovarianceAndS(Xen, idxEn, startEn, endEn, muEn, W21, covY, Set);
                BatchedTargetCovariance(Xru, idxRu, startRu, endRu, muRu, covTgt);
                LinAlg.SubtractMatrices(covY, covTgt, Cdiff);
                tmpGrad.Clear();
                LinAlg.MatMulAdd(Cdiff, Set, tmpGrad, 1f);
                LinAlg.ScaleMatrixInPlace(tmpGrad, (be > 1) ? (4f * opt.CoralWeight / (be - 1)) : 0f);
                LinAlg.AddMatricesInPlace(g21, tmpGrad);

                // == 4. Ортогональность: W^T W ~ I, чтобы преобразование почти изометрично ==
                OrthoPenaltyGrad(W12, g12, opt.OrthoWeight);
                OrthoPenaltyGrad(W21, g21, opt.OrthoWeight);

                // == 5. Связка: W21 ~ W12^T (чтобы преобразования были обратно связаны) ==
                CouplePenaltyGrad(W12, W21, g12, g21, opt.CoupleWeight);

                // == 6. Adam — основное обновление весов ==
                opt12.Step(W12.Data, g12.Data, opt.Lr);
                opt21.Step(W21.Data, g21.Data, opt.Lr);

                // == 7. Периодическая ортогонализация (ускоряет сходимость и качество) ==
                steps++;
                if (opt.RetractionEvery > 0 && (steps % opt.RetractionEvery) == 0)
                {
                    OrthoRetractionInPlace(W12, opt.OrthoRetraction);
                    OrthoRetractionInPlace(W21, opt.OrthoRetraction);
                }
            }

            float errorRu = 0.0f;
            for (int j = 0; j < nru; j++)
            {
                var col = Xru.GetColumn(j);
                LinAlg.Copy(col, x);

                LinAlg.MatVec(W12, x, y);   // y = W12 x
                LinAlg.MatVec(W21, y, z);   // z = W21 y -- результат обратного преобразования
                TensorPrimitives.Subtract(z, x, z); // ошибка восстановления
                var e = TensorPrimitives.Norm(z);
                if (e > errorRu)
                    errorRu = e;
            }

            float errorEn = 0.0f;
            for (int j = 0; j < nen; j++)
            {
                var col = Xen.GetColumn(j);
                LinAlg.Copy(col, x);

                LinAlg.MatVec(W21, x, y);   // y = W12 x
                LinAlg.MatVec(W12, y, z);   // z = W21 y -- результат обратного преобразования
                TensorPrimitives.Subtract(z, x, z); // ошибка восстановления
                var e = TensorPrimitives.Norm(z);
                if (e > errorEn)
                    errorEn = e;
            }

            _loggersSet.UserFriendlyLogger.LogInformation($"Epoch {epoch} done. ErrorRu: {errorRu}; ErrorEn: {errorEn}");
        }
    }

    // == ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==
    // Мягкая ортогонализация матрицы: W <- (1+beta)W - beta W(W^T W)
    private static void OrthoRetractionInPlace(MatrixFloat W, float beta)
    {
        if (beta <= 0f) return;
        int D = W.Dimensions[0];
        var G = new MatrixFloat(D, D);
        LinAlg.Gram(W, G); // вычисляем матрицу W^T W
        var WG = new MatrixFloat(D, D);
        LinAlg.MatMulAdd(W, G, WG, 1f); // W * Gram
        for (int i = 0; i < W.Data.Length; i++)
            W.Data[i] = (1f + beta) * W.Data[i] - beta * WG.Data[i];
    }

    // Средний вектор по батчу индексов
    private static void ComputeBatchMean(MatrixFloat X, int[] idx, int start, int end, float[] outMean)
    {
        Array.Clear(outMean, 0, outMean.Length);
        int n = Math.Max(1, end - start);
        for (int j = start; j < end; j++)
            LinAlg.AddInPlace(outMean.AsSpan(), X.GetColumn(idx[j]));
        for (int i = 0; i < outMean.Length; i++)
            outMean[i] /= n;
    }

    // CORAL: вычислить ковариацию преобразований и также вспомогательную матрицу для градиентов
    private static void BatchedCovarianceAndS(MatrixFloat X, int[] idx, int start, int end, float[] mu, MatrixFloat W, MatrixFloat cov, MatrixFloat S)
    {
        int D = X.Dimensions[0];
        cov.Clear(); S.Clear();
        var x = new float[D]; var y = new float[D];
        int n = Math.Max(1, end - start);
        for (int j = start; j < end; j++)
        {
            var col = X.GetColumn(idx[j]);
            for (int i = 0; i < D; i++) x[i] = col[i] - mu[i];
            LinAlg.MatVec(W, x, y);
            LinAlg.OuterAdd(y.AsSpan(), y, cov, 1f); // ковариация выводов
            LinAlg.OuterAdd(y.AsSpan(), x, S, 1f);   // cross-term для градиента
        }
        if (n > 1) LinAlg.ScaleMatrixInPlace(cov, 1f / (n - 1));
    }

    // CORAL: ковариация батча целевой стороны
    private static void BatchedTargetCovariance(MatrixFloat X, int[] idx, int start, int end, float[] mu, MatrixFloat cov)
    {
        int D = X.Dimensions[0]; cov.Clear();
        int n = Math.Max(1, end - start);
        var temp = new float[D];
        for (int j = start; j < end; j++)
        {
            var col = X.GetColumn(idx[j]);
            for (int i = 0; i < D; i++) temp[i] = col[i] - mu[i];
            LinAlg.OuterAdd(temp.AsSpan(), temp, cov, 1f);
        }
        if (n > 1) LinAlg.ScaleMatrixInPlace(cov, 1f / (n - 1));
    }

    // Добавить к градиенту регуляризатор ортогональности
    private static void OrthoPenaltyGrad(MatrixFloat W, MatrixFloat grad, float lambda)
    {
        var M = new MatrixFloat(W.Dimensions[0], W.Dimensions[1]);
        LinAlg.Gram(W, M);
        LinAlg.SubIdentityInPlace(M, 1f);
        var T = new MatrixFloat(W.Dimensions[0], W.Dimensions[1]);
        LinAlg.MatMulAdd(W, M, T, 1f);
        LinAlg.ScaleMatrixInPlace(T, 4f * lambda);
        LinAlg.AddMatricesInPlace(grad, T);
    }

    // Регуляризатор: W21 ≈ W12^T
    private static void CouplePenaltyGrad(MatrixFloat W12, MatrixFloat W21, MatrixFloat g12, MatrixFloat g21, float lambda)
    {
        int D = W12.Dimensions[0];
        for (int j = 0; j < D; j++)
            for (int i = 0; i < D; i++)
            {
                float diff = W21[i, j] - W12[j, i];
                g21[i, j] += 2f * lambda * diff;
                g12[j, i] -= 2f * lambda * diff;
            }
    }

    // Перемешать индексы (Fisher-Yates)
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

    // == Применить преобразование F12 к одному вектору
    public void ApplyF12(ReadOnlySpan<float> src, Span<float> dst)
        => LinAlg.MatVec(W12, src, dst);

    // == Применить преобразование F21 к одному вектору
    public void ApplyF21(ReadOnlySpan<float> src, Span<float> dst)
        => LinAlg.MatVec(W21, src, dst);
}