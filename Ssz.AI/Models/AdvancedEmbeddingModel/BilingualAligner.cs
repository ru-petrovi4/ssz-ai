using Microsoft.Extensions.Logging;
using Ssz.AI.Helpers;
using Ssz.Utils.Logging;
using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public class BilingualAligner
{
    private ILoggersSet _loggersSet;

    public BilingualAligner(ILoggersSet loggersSet)
    {
        _loggersSet = loggersSet;
    }

    public MatrixFloat W12 = new MatrixFloat();
    public MatrixFloat W21 = new MatrixFloat();

    // == Применить преобразование F12 к одному вектору
    public void ApplyF12(ReadOnlySpan<float> src, Span<float> dst)
        => LinAlg.MatVec(W12, src, dst);

    // == Применить преобразование F21 к одному вектору
    public void ApplyF21(ReadOnlySpan<float> src, Span<float> dst)
        => LinAlg.MatVec(W21, src, dst);

    // Главная тренировка: возвращает (W12, W21)
    public void Train(
        MatrixFloat ru,           // 300 x N_ru
        MatrixFloat en,           // 300 x N_en
        int iters = 10,           // число итераций самосогласования
        int maxPairs = 5000,      // максимум пар в псевдословаре на итерацию
        int invSqrtIters = 10     // итерации Ньютона–Шульца для (A^T A)^(-1/2)
    )
    {
        if (ru.Dimensions[0] != 300 || en.Dimensions[0] != 300)
            throw new ArgumentException("Ожидаются матрицы эмбеддингов размера 300 x VocabSize");

        // 1) Нормализация столбцов -> косинусная метрика = скалярное произведение
        //NormalizeColumns(ru);
        //NormalizeColumns(en);

        // 2) Инициализация W12 как случайная ортогональная (через QR-подобную ортогонализацию)
        var rnd = new Random(42);
        W12 = RandomOrthonormal(300, rnd);
        W21 = Transpose(W12); // строгая циклическая согласованность на старте

        // Рабочие буферы для отображений
        var ru_mapped = new MatrixFloat(300, ru.Dimensions[1]);
        var en_mapped = new MatrixFloat(300, en.Dimensions[1]);

        for (int it = 0; it < iters; it++)
        {
            // 3) Прямое и обратное отображения
            Apply(W12, ru, ru_mapped); // ru -> en space
            Apply(W21, en, en_mapped); // en -> ru space

            // 4) Поиск взаимных ближайших соседей по косинусной близости
            var pairs12 = MutualNearestNeighbors(ru_mapped, en, maxPairs);
            var pairs21 = MutualNearestNeighbors(en_mapped, ru, maxPairs);

            // 5) Обновление W12 по Procrustes (полярное разложение через (M^T M)^(-1/2))
            //    M12 = Y X^T, где X — исходные ru для pairs12, Y — цели en для pairs12
            var (X12, Y12) = GatherPairs(ru, en, pairs12);
            var M12 = MatMul_AT(Y12, X12); // 300xNp * (300xNp)^T == 300x300
            var W12_new = PolarOrthogonal(M12, invSqrtIters);

            // 6) Симметричное обновление: W21 := W12^T для строгой циклической согласованности
            W12 = W12_new;
            W21 = Transpose(W12);




            //float errorRu = 0.0f;
            //for (int j = 0; j < nru; j++)
            //{
            //    var col = Xru.GetColumn(j);
            //    LinAlg.Copy(col, x);

            //    LinAlg.MatVec(W12, x, y);   // y = W12 x
            //    LinAlg.MatVec(W21, y, z);   // z = W21 y -- результат обратного преобразования
            //    TensorPrimitives.Subtract(z, x, z); // ошибка восстановления
            //    var e = TensorPrimitives.Norm(z);
            //    if (e > errorRu)
            //        errorRu = e;
            //}

            //float errorEn = 0.0f;
            //for (int j = 0; j < nen; j++)
            //{
            //    var col = Xen.GetColumn(j);
            //    LinAlg.Copy(col, x);

            //    LinAlg.MatVec(W21, x, y);   // y = W12 x
            //    LinAlg.MatVec(W12, y, z);   // z = W21 y -- результат обратного преобразования
            //    TensorPrimitives.Subtract(z, x, z); // ошибка восстановления
            //    var e = TensorPrimitives.Norm(z);
            //    if (e > errorEn)
            //        errorEn = e;
            //}

            //_loggersSet.UserFriendlyLogger.LogInformation($"Epoch {it} done. ErrorRu: {errorRu}; ErrorEn: {errorEn}");
            _loggersSet.UserFriendlyLogger.LogInformation($"Epoch {it} done.");
        }
    }

    // Применение линейного отображения Y = W * X
    public void Apply(MatrixFloat W, MatrixFloat X, MatrixFloat Y)
    {
        int d = W.Dimensions[0];
        if (W.Dimensions[1] != d || X.Dimensions[0] != d)
            throw new ArgumentException("Размерности не согласованы (ожидается W: dxd, X: dxN).");
        if (Y.Dimensions[0] != d || Y.Dimensions[1] != X.Dimensions[1])
            throw new ArgumentException("Буфер Y неправильного размера.");

        int n = X.Dimensions[1];
        for (int j = 0; j < n; j++)
        {
            var x = X.GetColumn(j);
            var y = Y.GetColumn(j);
            // y = W * x
            for (int col = 0; col < d; col++)
            {
                float sum = 0f;
                // W column-major: W[i, col] = Data[i + col*d]
                // y[col] = sum_i W[col, i] * x[i]  (осторожно с индексами из-за хранения)
                // Наш индексер: this[i,j] = Data[i + j * d], значит строка = i, столбец = j
                // Хотим y[r] = sum_c W[r, c] * x[c]
                for (int c = 0; c < d; c++)
                    sum += W[col, c] * x[c]; // используем индексатор ниже
                y[col] = sum;
            }
            // Нормализуем для стабильной косинусной метрики
            NormalizeSpanInPlace(y);
        }
    }

    // Поиск взаимных ближайших соседей по косинусной близости (возвращает до maxPairs пар)
    // srcMapped: d x Ns (уже в целевом пространстве), tgt: d x Nt
    private List<(int src, int tgt, float sim)> MutualNearestNeighbors(
        MatrixFloat srcMapped,
        MatrixFloat tgt,
        int maxPairs)
    {
        int d = srcMapped.Dimensions[0];
        int ns = srcMapped.Dimensions[1];
        int nt = tgt.Dimensions[1];

        var fBestIdx = new int[ns];
        var fBestSim = new float[ns];
        Array.Fill(fBestIdx, -1);
        for (int i = 0; i < ns; i++) fBestSim[i] = float.NegativeInfinity;

        var rBestIdx = new int[nt];
        var rBestSim = new float[nt];
        Array.Fill(rBestIdx, -1);
        for (int j = 0; j < nt; j++) rBestSim[j] = float.NegativeInfinity;

        // Вперёд: для каждого src ищем лучший tgt
        for (int i = 0; i < ns; i++)
        {
            var xi = srcMapped.GetColumn(i);
            for (int j = 0; j < nt; j++)
            {
                var yj = tgt.GetColumn(j);
                float sim = TensorPrimitives.CosineSimilarity(xi, yj);
                if (sim > fBestSim[i])
                {
                    fBestSim[i] = sim;
                    fBestIdx[i] = j;
                }
            }
        }

        // Назад: для каждого tgt ищем лучший src
        for (int j = 0; j < nt; j++)
        {
            var yj = tgt.GetColumn(j);
            for (int i = 0; i < ns; i++)
            {
                var xi = srcMapped.GetColumn(i);
                float sim = TensorPrimitives.CosineSimilarity(yj, xi);
                if (sim > rBestSim[j])
                {
                    rBestSim[j] = sim;
                    rBestIdx[j] = i;
                }
            }
        }

        // Взаимность
        var pairs = new List<(int src, int tgt, float sim)>();
        for (int i = 0; i < ns; i++)
        {
            int j = fBestIdx[i];
            if (j >= 0 && rBestIdx[j] == i)
                pairs.Add((i, j, fBestSim[i]));
        }

        // Топ по схожести
        pairs.Sort((a, b) => b.sim.CompareTo(a.sim));
        if (pairs.Count > maxPairs) pairs.RemoveRange(maxPairs, pairs.Count - maxPairs);
        return pairs;
    }

    // Собрать подматрицы X (source) и Y (target) по списку пар индексов
    private (MatrixFloat X, MatrixFloat Y) GatherPairs(
        MatrixFloat sourceAll, MatrixFloat targetAll,
        List<(int src, int tgt, float sim)> pairs)
    {
        int d = sourceAll.Dimensions[0];
        int n = pairs.Count;
        var X = new MatrixFloat(d, n);
        var Y = new MatrixFloat(d, n);
        for (int k = 0; k < n; k++)
        {
            var (isrc, jtgt, _) = pairs[k];
            sourceAll.GetColumn(isrc).CopyTo(X.GetColumn(k));
            targetAll.GetColumn(jtgt).CopyTo(Y.GetColumn(k));
        }
        return (X, Y);
    }

    // M = A * B^T (A: d x n, B: d x n) => d x d
    private MatrixFloat MatMul_AT(MatrixFloat A, MatrixFloat B)
    {
        int d = A.Dimensions[0];
        int n = A.Dimensions[1];
        if (B.Dimensions[0] != d || B.Dimensions[1] != n)
            throw new ArgumentException("Для A*B^T требуется A: d x n и B: d x n");

        var M = new MatrixFloat(d, d);
        // M[r,c] = sum_p A[r,p] * B[c,p]
        for (int p = 0; p < n; p++)
        {
            var ap = A.GetColumn(p);
            var bp = B.GetColumn(p);
            for (int r = 0; r < d; r++)
            {
                float ar = ap[r];
                for (int c = 0; c < d; c++)
                    M[r, c] += ar * bp[c];
            }
        }
        return M;
    }

    // Ортогональный фактор полярного разложения через (M^T M)^(-1/2) по Ньютон–Шульцу
    private MatrixFloat PolarOrthogonal(MatrixFloat M, int iters = 10)
    {
        int d = M.Dimensions[0];
        if (M.Dimensions[1] != d)
            throw new ArgumentException("Ожидается квадратная матрица M");

        var Mt = Transpose(M);
        var A = MatMul(Mt, M); // A = M^T M (SPD)
        var invSqrt = InverseSquareRootSPD(A, iters);
        var W = MatMul(M, invSqrt); // W = M * (M^T M)^(-1/2)
        return W;
    }

    // (SPD)^(-1/2) через Ньютон–Шульца с предварительной нормировкой по ||A||_F
    private MatrixFloat InverseSquareRootSPD(MatrixFloat A, int iters)
    {
        int d = A.Dimensions[0];
        if (A.Dimensions[1] != d)
            throw new ArgumentException("Ожидается квадратная SPD-матрица");

        // Y0 = A / alpha, Z0 = I, где alpha = ||A||_F
        float alpha = TensorPrimitives.Norm(A.Data);
        if (alpha <= 0f) throw new ArgumentException("Матрица вырождена либо нулевая");

        var Y = A.Clone();
        ScaleInPlace(Y, 1.0f / alpha);
        var Z = Identity(d);

        var I = Identity(d);
        for (int k = 0; k < iters; k++)
        {
            // T = 0.5 * (3I - Z*Y)
            var ZY = MatMul(Z, Y);
            var threeI_minus_ZY = Subtract(Scale(I, 3.0f), ZY);
            var T = Scale(threeI_minus_ZY, 0.5f);

            // Y = Y * T
            Y = MatMul(Y, T);
            // Z = T * Z
            Z = MatMul(T, Z);
        }

        // A^{-1/2} ≈ Z / sqrt(alpha)
        ScaleInPlace(Z, 1.0f / MathF.Sqrt(alpha));
        return Z;
    }

    // Базовая линалг
    private MatrixFloat MatMul(MatrixFloat A, MatrixFloat B)
    {
        int m = A.Dimensions[0];
        int k = A.Dimensions[1];
        if (B.Dimensions[0] != k) throw new ArgumentException("Несогласованные размерности в MatMul");
        int n = B.Dimensions[1];

        var C = new MatrixFloat(m, n);
        for (int j = 0; j < n; j++)
        {
            var bj = B.GetColumn(j);
            var cj = C.GetColumn(j);
            // cj = A * bj
            for (int r = 0; r < m; r++)
            {
                float sum = 0f;
                for (int t = 0; t < k; t++)
                    sum += A[r, t] * bj[t];
                cj[r] = sum;
            }
        }
        return C;
    }

    private MatrixFloat Transpose(MatrixFloat A)
    {
        int m = A.Dimensions[0];
        int n = A.Dimensions[1];
        var T = new MatrixFloat(n, m);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                T[j, i] = A[i, j];
        return T;
    }

    private MatrixFloat Identity(int d)
    {
        var I = new MatrixFloat(d, d);
        for (int i = 0; i < d; i++) I[i, i] = 1f;
        return I;
    }

    private MatrixFloat Scale(MatrixFloat A, float s)
    {
        var B = A.Clone();
        ScaleInPlace(B, s);
        return B;
    }

    private void ScaleInPlace(MatrixFloat A, float s)
    {
        var data = A.Data.AsSpan();
        for (int i = 0; i < data.Length; i++) data[i] *= s;
    }

    private MatrixFloat Subtract(MatrixFloat A, MatrixFloat B)
    {
        if (A.Dimensions[0] != B.Dimensions[0] || A.Dimensions[1] != B.Dimensions[1])
            throw new ArgumentException("Размерности не совпадают в Subtract");
        var C = A.Clone();
        var a = C.Data.AsSpan();
        var b = B.Data.AsSpan();
        TensorPrimitives.Subtract(a, b, a); // C = A - B
        return C;
    }

    private MatrixFloat RandomOrthonormal(int d, Random rnd)
    {
        // Простая ортогонализация столбцов методом Грама–Шмидта
        var M = new MatrixFloat(d, d);
        for (int j = 0; j < d; j++)
        {
            var col = M.GetColumn(j);
            for (int i = 0; i < d; i++)
                col[i] = (float)(rnd.NextDouble() * 2.0 - 1.0);

            // Вычитание проекций
            for (int k = 0; k < j; k++)
            {
                var prev = M.GetColumn(k);
                float dot = TensorPrimitives.Dot(col, prev);
                for (int i = 0; i < d; i++)
                    col[i] -= dot * prev[i];
            }
            // Нормировка
            NormalizeSpanInPlace(col);
        }
        return M;
    }

    private void NormalizeColumns(MatrixFloat X)
    {
        int d = X.Dimensions[0];
        int n = X.Dimensions[1];
        for (int j = 0; j < n; j++)
            NormalizeSpanInPlace(X.GetColumn(j));
    }

    private void NormalizeSpanInPlace(Span<float> v)
    {
        float norm = MathF.Sqrt(TensorPrimitives.SumOfSquares(v));
        if (norm > 0f)
        {
            float inv = 1.0f / norm;
            for (int i = 0; i < v.Length; i++) v[i] *= inv;
        }
    }    
}