using Ssz.AI.Models;
using System;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Helpers;

public static class LinAlg
{
    // Скопировать массив src в dst, предполагая одинаковую длину
    public static void Copy(ReadOnlySpan<float> src, Span<float> dst)
        => src.CopyTo(dst);

    // Нормализация и центрирование эмбеддингов:
    // - L2-нормировка каждого столбца
    // - вычитание среднего вектора по всему батчу
    public static void NormalizeAndCenter(MatrixFloat X)
    {
        int n = X.Dimensions[1];
        int D = X.Dimensions[0];
        var mean = new float[D];

        // Считаем среднее
        for (int j = 0; j < n; j++)
        {
            var col = X.GetColumn(j);
            float norm = (float)Math.Sqrt(LinAlg.Dot(col, col));
            if (norm > 0) TensorPrimitives.Divide(col, norm, col); // L2-norm
            TensorPrimitives.Add(mean, col, mean);
        }
        TensorPrimitives.Divide(mean, n, mean);

        // Вычитаем среднее из каждого столбца
        for (int j = 0; j < n; j++)
        {
            var col = X.GetColumn(j);
            TensorPrimitives.Subtract(col, mean, col);
        }
    }

    // Заполнить весь dst одним значением
    public static void Fill(Span<float> dst, float value)
    {
        for (int i = 0; i < dst.Length; i++)
            dst[i] = value;
    }

    // Элементное сложение: dst_i += add_i
    public static void AddInPlace(Span<float> dst, ReadOnlySpan<float> add)
    {
        TensorPrimitives.Add(dst, add, dst);
    }

    // Элементное вычитание: dst_i -= sub_i
    public static void SubInPlace(Span<float> dst, ReadOnlySpan<float> sub)
    {
        TensorPrimitives.Subtract(dst, sub, dst);
    }

    // Умножение всех элементов dst на скаляр
    public static void ScaleInPlace(Span<float> dst, float s)
        => TensorPrimitives.Multiply(dst, s, dst);

    // Скалярное произведение двух векторов (dot product)
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        => TensorPrimitives.Dot(a, b);

    // y = W * x: прямое умножение матрицы W[d x d] на вектор x[d]
    // Результат помещается в y[d]
    public static void MatVec(MatrixFloat W, ReadOnlySpan<float> x, Span<float> y)
    {
        // Обнуляем результат
        Fill(y, 0f);
        for (int col = 0; col < W.Dimensions[1]; col++)
        {
            float coef = x[col];
            if (coef == 0f) continue; // ускоряет вычисление при разреженных/нормированных векторах
            var wCol = W.GetColumn(col); // берем столбец
            MultiplyAddScalar(wCol, coef, y); // y += coef * W[:,col]
        }
    }

    // y = W^T * x: умножение транспонированной матрицы на вектор
    public static void MatTVec(MatrixFloat W, ReadOnlySpan<float> x, Span<float> y)
    {
        // Проходим по столбцам матрицы, для каждого: скалярное произведение с x
        for (int col = 0; col < W.Dimensions[1]; col++)
            y[col] = TensorPrimitives.Dot(W.GetColumn(col), x);
    }

    // C += alpha * A * B, где все матрицы квадратные
    // Не создаёт новых матриц, результат записывается поверх C
    public static void MatMulAdd(MatrixFloat A, MatrixFloat B, MatrixFloat C, float alpha = 1f)
    {
        int d = A.Dimensions[0], k = A.Dimensions[1], n = B.Dimensions[1];
        // Классическое тройное умножение
        for (int colB = 0; colB < n; colB++)
        {
            var cCol = C.GetColumn(colB);
            for (int i = 0; i < k; i++)
            {
                float mult = alpha * B[i, colB];
                if (mult == 0f) continue;
                var aCol = A.GetColumn(i);
                MultiplyAddScalar(aCol, mult, cCol);
            }
        }
    }

    // Out += alpha * u v^T
    // u: длина d, v: длина k (это Out[d x k])
    public static void OuterAdd(Span<float> u, ReadOnlySpan<float> v, MatrixFloat Out, float alpha = 1f)
    {
        int d = Out.Dimensions[0], k = Out.Dimensions[1];
        for (int col = 0; col < k; col++)
        {
            float val = alpha * v[col];
            if (val == 0f) continue;
            var outCol = Out.GetColumn(col);
            MultiplyAddScalar(u, val, outCol);
        }
    }

    // Простое прибавление к каждому элементу dest[i] += a[i] * scalar
    public static void MultiplyAddScalar(ReadOnlySpan<float> a, float scalar, Span<float> dest)
    {
        TensorPrimitives.MultiplyAdd(a, scalar, dest, dest);        
    }

    // C = A - B (помещает результат в C)
    public static void SubtractMatrices(MatrixFloat A, MatrixFloat B, MatrixFloat C)
    {
        TensorPrimitives.Subtract(A.Data, B.Data, C.Data);
    }

    // A += B (поэлементное сложение двух матриц одной размерности)
    public static void AddMatricesInPlace(MatrixFloat A, MatrixFloat B)
    {
        TensorPrimitives.Add(A.Data, B.Data, A.Data);
    }

    // Поэлементное умножение матрицы на скаляр
    public static void ScaleMatrixInPlace(MatrixFloat A, float s)
        => TensorPrimitives.Multiply(A.Data, s, A.Data);

    // Сделать матрицу единичной (diagonal=1, off-diagonal=0)
    public static void SetIdentity(MatrixFloat A)
    {
        A.Clear();
        int minDim = Math.Min(A.Dimensions[0], A.Dimensions[1]);
        for (int i = 0; i < minDim; i++)
            A[i, i] = 1f;
    }

    // Грам-матрица: G = W^T * W
    // Это (скалярные) произведения всех столбцов W друг с другом
    public static void Gram(MatrixFloat W, MatrixFloat G)
    {
        int d = W.Dimensions[0], k = W.Dimensions[1];
        G.Clear();
        for (int j = 0; j < k; j++)
        {
            var wj = W.GetColumn(j);
            for (int t = j; t < k; t++)
            {
                var wt = W.GetColumn(t);
                float val = TensorPrimitives.Dot(wj, wt);
                G[j, t] += val;
                if (t != j) G[t, j] += val; // симметричное заполнение
            }
        }
    }

    // Прибавить alpha к диагонали матрицы
    public static void AddIdentityInPlace(MatrixFloat A, float alpha = 1f)
    {
        int minDim = Math.Min(A.Dimensions[0], A.Dimensions[1]);
        for (int i = 0; i < minDim; i++)
            A[i, i] += alpha;
    }

    // Вычесть alpha с диагонали матрицы (удобно для Gram-един. регулир.)
    public static void SubIdentityInPlace(MatrixFloat A, float alpha = 1f)
        => AddIdentityInPlace(A, -alpha);

    /// <summary>
    /// Возвращает индекс столбца матрицы X (d x n), ближайшего к вектору query (длина d).
    /// По умолчанию предполагается, что столбцы X L2-нормированы (||col||=1), и ближайший
    /// определяется по косинусной близости как максимум скалярного произведения col·query.
    /// Если assumeUnitNormalized = false, ближайший определяется по евклидовой метрике:
    /// argmin_j ||X[:,j] - query||^2 = ||X[:,j]||^2 + ||query||^2 - 2·(X[:,j]·query).
    /// </summary>
    /// <param name="X">Матрица эмбеддингов размером d x n (столбцы — слова)</param>
    /// <param name="query">Запрос-вектор длины d</param>
    /// <param name="assumeUnitNormalized">
    /// true — использовать косинус/скалярный продукт (быстрее, если столбцы нормированы);
    /// false — использовать евклидову дистанцию (нормы не требуются единичные)
    /// </param>
    /// <param name="colNorm2">
    /// (Необязательно) Предвычисленные нормы ||X[:,j]||^2 длины n для ускорения в евклидовом режиме.
    /// Если не задано или размер не совпадает, нормы столбцов будут вычисляться на лету.
    /// </param>
    /// <returns>Индекс столбца [0..n-1], ближайшего к query</returns>
    public static int NearestColumnIndex(
        MatrixFloat X,
        ReadOnlySpan<float> query,
        bool assumeUnitNormalized = true,
        ReadOnlySpan<float> colNorm2 = default)
    {
        int d = X.Dimensions[0];
        int n = X.Dimensions[1];

        if (n <= 0) return -1; // Пустая матрица — возвращаем -1 как признак отсутствия

        int bestIdx = 0;

        if (assumeUnitNormalized)
        {
            // Косинусная близость эквивалентна максимуму dot при ||col||=1 (норма запроса не влияет на ранжирование)
            float bestDot = float.NegativeInfinity;

            for (int j = 0; j < n; j++)
            {
                var col = X.GetColumn(j);
                float dot = TensorPrimitives.Dot(col, query); // SIMD-ускоренный dot без аллокаций
                if (dot > bestDot)
                {
                    bestDot = dot;
                    bestIdx = j;
                }
            }
        }
        else
        {
            // Евклидова метрика: минимизируем ||col - query||^2 = ||col||^2 + ||query||^2 - 2 * dot
            float qNorm2 = TensorPrimitives.Dot(query, query);
            float bestDist2 = float.PositiveInfinity;
            bool hasPrecomputed = !colNorm2.IsEmpty && colNorm2.Length == n;

            for (int j = 0; j < n; j++)
            {
                var col = X.GetColumn(j);
                float dot = TensorPrimitives.Dot(col, query);
                float cNorm2 = hasPrecomputed ? colNorm2[j] : TensorPrimitives.Dot(col, col);
                float dist2 = cNorm2 + qNorm2 - 2f * dot;
                if (dist2 < bestDist2)
                {
                    bestDist2 = dist2;
                    bestIdx = j;
                }
            }
        }

        return bestIdx;
    }

    /// <summary>
    /// Вспомогательный метод: заполняет массив outNorm2 длины n квадратами L2-норм столбцов матрицы X.
    /// Полезно для ускорения многократных вызовов NearestColumnIndex в евклидовом режиме.
    /// </summary>
    public static void ComputeColumnNorm2(MatrixFloat X, Span<float> outNorm2)
    {
        int n = X.Dimensions[1];
        if (outNorm2.Length < n)
            throw new ArgumentException("outNorm2.Length < number of columns in X");

        for (int j = 0; j < n; j++)
        {
            var col = X.GetColumn(j);
            outNorm2[j] = TensorPrimitives.Dot(col, col);
        }
    }
}
