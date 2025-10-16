using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using MathNet.Numerics.LinearAlgebra; // NuGet: MathNet.Numerics
using MathNet.Numerics.LinearAlgebra.Single;
using Microsoft.Extensions.Logging;
using Ssz.Utils.Logging;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public class VectorSetAlignment
{
    private ILoggersSet _loggersSet;

    public VectorSetAlignment(ILoggersSet loggersSet)
    {
        _loggersSet = loggersSet;
    }

    /// <summary>
    /// Находит оптимальное ортогональное преобразование (поворот) для сопоставления двух множеств нормированных векторов.
    /// Учитывает наличие кластеров в множествах: алгоритм ICP адаптирован для оптимального уникального matching с помощью Hungarian algorithm на каждой итерации.
    /// Это обеспечивает биективное (один-к-одному) сопоставление, уважая кластерные структуры (Hungarian максимизирует общую сумму сходств, что естественно группирует похожие кластеры).
    /// - Итеративно находит оптимальные уникальные пары с Hungarian по косинусному сходству (только положительные для избежания антикорреляций).
    /// - Вычисляет поворот с помощью Orthogonal Procrustes на основе текущих пар.
    /// - Применяет поворот и повторяет до сходимости.
    /// Вход: A и B - матрицы [vectorDim, nVectors] с нормированными векторами (норма=1).
    /// Выход: Ортогональная матрица поворота R [vectorDim, vectorDim].
    /// Параметры: maxIterations - максимум итераций; epsilon - порог сходимости по изменению суммарного сходства.
    /// Производительность: Hungarian O(n^3) для n=300 (~27M операций per iter) feasible; TensorPrimitives для dot.
    /// Предполагается наличие реализации HungarianMatching(float[,] costMatrix), где cost = -similarity для максимизации.
    /// </summary>
    public MatrixFloat_ColumnMajor FindOptimalRotation(MatrixFloat_ColumnMajor A, MatrixFloat_ColumnMajor B, int maxIterations = 50, float epsilon = 1e-5f)
    {
        int vectorDim = A.Dimensions[0];
        int nVectors = A.Dimensions[1];
        if (vectorDim != B.Dimensions[0] || nVectors != B.Dimensions[1])
            throw new ArgumentException("Множества должны иметь одинаковые размеры.");

        // Копируем A для итеративного поворота
        MatrixFloat_ColumnMajor rotatedA = A.Clone();

        // Аккумулятор полной матрицы поворота (инициализируем как единичную)
        MatrixFloat_ColumnMajor R_total = CreateIdentityMatrix(vectorDim);

        // Начальное вычисление для сходимости (iter=0)
        float prevTotalSimilarity = ComputeTotalSimilarity(rotatedA, B);

        for (int iterationN = 0; iterationN < maxIterations; ++iterationN)
        {
            // Шаг 1: Вычисляем матрицу косинусного сходства
            float[,] similarityMatrix = ComputeCosineSimilarityMatrix(rotatedA, B);

            // Шаг 2: Оптимальное уникальное matching с Hungarian (cost = -max(0, similarity) для максимизации положительных сходств)
            long[,] costMatrix = new long[nVectors, nVectors];
            for (int i = 0; i < nVectors; ++i)
                for (int j = 0; j < nVectors; ++j)
                    costMatrix[i, j] = (long)(similarityMatrix[i, j] * -10000.0f);
            //costMatrix[i, j] = (int)(-MathF.Max(0f, similarityMatrix[i, j]) * 100.0f); // Только положительные, минус для максимизации

            int[] matching = HungarianAlgorithm.FindAssignments(costMatrix); // Предполагаемая реализация: matching[i] = j для A_i -> B_j

            // Проверяем на полное matching
            int matchedCount = 0;
            for (int i = 0; i < nVectors; ++i)
                if (matching[i] != -1) matchedCount++;
            if (matchedCount < nVectors)
                throw new InvalidOperationException("Hungarian не дал полное сопоставление. Проверьте входные данные.");

            // Вычисляем totalSimilarity для сходимости (сумма сходств по matched парам)
            float totalSimilarity = 0f;
            for (int i = 0; i < nVectors; ++i)
            {
                int j = matching[i];
                totalSimilarity += similarityMatrix[i, j];
            }

            // Проверяем сходимость (дополнено: если изменение мало, проверяем структуру по Frobenius норме расстояний)
            var delta = totalSimilarity - prevTotalSimilarity;
            _loggersSet.UserFriendlyLogger.LogInformation($"Iteration iterationN: {iterationN}; totalSimilarity: {totalSimilarity}; delta: {delta}");
            if (Math.Abs(delta) < epsilon)
            {
                // Дополнительная проверка по структуре
                float structureDiff = ComputeStructureDifference(rotatedA, B);
                if (structureDiff < epsilon) // Можно настроить порог
                    break;
            }
            prevTotalSimilarity = totalSimilarity;

            // Шаг 3: Строим кросс-ковариационную матрицу M на основе matched пар
            MatrixFloat_ColumnMajor M = new MatrixFloat_ColumnMajor(new[] { vectorDim, vectorDim });
            for (int k = 0; k < nVectors; ++k)
            {
                int j = matching[k];
                Span<float> aCol = rotatedA.GetColumn(k);
                Span<float> bCol = B.GetColumn(j);
                for (int i = 0; i < vectorDim; ++i)
                {
                    for (int j2 = 0; j2 < vectorDim; ++j2)
                    {
                        M[i, j2] += aCol[i] * bCol[j2];
                    }
                }
            }
            // Нормализуем M по числу пар (для consistency)
            for (int i = 0; i < vectorDim; ++i)
                for (int j = 0; j < vectorDim; ++j)
                    M[i, j] /= matchedCount;

            // Шаг 4: SVD для Procrustes: M = U S V^T
            float[,] mData = new float[vectorDim, vectorDim];
            for (int i = 0; i < vectorDim; ++i)
                for (int j = 0; j < vectorDim; ++j)
                    mData[i, j] = M[i, j];

            var matrixM = Matrix.Build.DenseOfArray(mData);
            var svd = matrixM.Svd(true);
            var U = svd.U;
            var VT = svd.VT;

            // Вычисляем R = U * VT
            var R_temp = U * VT;

            // Корректировка для чистого поворота (det(R) = 1, без отражения)
            if (R_temp.Determinant() < 0)
            {
                var lastColumn = U.Column(U.ColumnCount - 1);
                U.SetColumn(U.ColumnCount - 1, -lastColumn);
                R_temp = U * VT;
            }

            // Конвертируем в MatrixFloat
            MatrixFloat_ColumnMajor R_iter = new MatrixFloat_ColumnMajor(new[] { vectorDim, vectorDim });
            for (int i = 0; i < vectorDim; ++i)
                for (int j = 0; j < vectorDim; ++j)
                    R_iter[i, j] = (float)R_temp[i, j];

            // Аккумулируем R_total = R_iter * R_total (матричное умножение)
            R_total = MatrixMultiply(R_iter, R_total);

            // Шаг 5: Применяем итеративный поворот к rotatedA
            rotatedA = ApplyRotation(R_iter, rotatedA);

            // Нормализуем векторы после поворота
            NormalizeColumns(rotatedA);
        }

        // Финальная проверка ортогональности R_total (для отладки, можно убрать)
        // MatrixFloat check = MatrixMultiply(R_total, Transpose(R_total));
        // (проверить, близко ли к единичной)

        return R_total; // Возвращаем аккумулированный поворот
    }

    /// <summary>
    /// Вычисляет суммарное сходство для начальной оценки (без matching).
    /// </summary>
    private float ComputeTotalSimilarity(MatrixFloat_ColumnMajor X, MatrixFloat_ColumnMajor Y)
    {
        float[,] sim = ComputeCosineSimilarityMatrix(X, Y);
        float total = 0f;
        for (int i = 0; i < sim.GetLength(0); ++i)
        {
            float maxSim = float.MinValue;
            for (int j = 0; j < sim.GetLength(1); ++j)
                maxSim = Math.Max(maxSim, sim[i, j]);
            total += maxSim;
        }
        return total;
    }

    /// <summary>
    /// Вычисляет разницу структур по Frobenius норме матриц косинусных расстояний.
    /// </summary>
    private float ComputeStructureDifference(MatrixFloat_ColumnMajor X, MatrixFloat_ColumnMajor Y)
    {
        MatrixFloat_ColumnMajor distX = ComputeCosineDistanceMatrix(X);
        MatrixFloat_ColumnMajor distY = ComputeCosineDistanceMatrix(Y);
        float diff = 0f;
        for (int i = 0; i < distX.Data.Length; ++i)
            diff += (distX.Data[i] - distY.Data[i]) * (distX.Data[i] - distY.Data[i]);
        return (float)Math.Sqrt(diff);
    }

    /// <summary>
    /// Транспонирует квадратную матрицу.
    /// </summary>
    private MatrixFloat_ColumnMajor Transpose(MatrixFloat_ColumnMajor M)
    {
        int dim = M.Dimensions[0];
        MatrixFloat_ColumnMajor T = new MatrixFloat_ColumnMajor(new[] { dim, dim });
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j)
                T[i, j] = M[j, i];
        return T;
    }

    /// <summary>
    /// Создает единичную матрицу размера dim x dim.
    /// </summary>
    private MatrixFloat_ColumnMajor CreateIdentityMatrix(int dim)
    {
        MatrixFloat_ColumnMajor I = new MatrixFloat_ColumnMajor(new[] { dim, dim });
        for (int i = 0; i < dim; ++i)
            I[i, i] = 1f;
        return I;
    }

    /// <summary>
    /// Матричное умножение A * B (A и B - [dim, dim]).
    /// Оптимизировано с TensorPrimitives для inner loops.
    /// </summary>
    private MatrixFloat_ColumnMajor MatrixMultiply(MatrixFloat_ColumnMajor A, MatrixFloat_ColumnMajor B)
    {
        int dim = A.Dimensions[0];
        MatrixFloat_ColumnMajor result = new MatrixFloat_ColumnMajor(new[] { dim, dim });
        for (int i = 0; i < dim; ++i)
        {
            for (int j = 0; j < dim; ++j)
            {
                float sum = 0f;
                // Для производительности: используем цикл, но можно Span если contiguous (здесь не)
                for (int k = 0; k < dim; ++k)
                {
                    sum += A[i, k] * B[k, j];
                }
                result[i, j] = sum;
            }
        }
        return result;
    }

    /// <summary>
    /// Вычисляет матрицу косинусного сходства (dot products) между всеми векторами A и B.
    /// Результат: [nA, nB], значения от -1 до 1 (косинус угла).
    /// Для производительности: Полное матричное умножение с TensorPrimitives.
    /// </summary>
    private float[,] ComputeCosineSimilarityMatrix(MatrixFloat_ColumnMajor A, MatrixFloat_ColumnMajor B)
    {
        int nA = A.Dimensions[1];
        int nB = B.Dimensions[1];
        int dim = A.Dimensions[0];
        float[,] result = new float[nA, nB];

        for (int i = 0; i < nA; ++i)
        {
            Span<float> vi = A.GetColumn(i);
            for (int j = 0; j < nB; ++j)
            {
                Span<float> vj = B.GetColumn(j);
                result[i, j] = TensorPrimitives.Dot(vi, vj);
            }
        }
        return result;
    }

    /// <summary>
    /// Применяет ортогональную матрицу поворота R к набору векторов A.
    /// Возвращает новую MatrixFloat [vectorDim, nVectors].
    /// Учитывает column-major хранение: Извлекает элементы без contiguous Span для строк.
    /// </summary>
    public MatrixFloat_ColumnMajor ApplyRotation(MatrixFloat_ColumnMajor R, MatrixFloat_ColumnMajor A)
    {
        int vectorDim = A.Dimensions[0];
        int nVectors = A.Dimensions[1];
        var result = new MatrixFloat_ColumnMajor(new[] { vectorDim, nVectors });

        for (int k = 0; k < nVectors; ++k)
        {
            Span<float> aCol = A.GetColumn(k);
            Span<float> resCol = result.GetColumn(k);
            for (int i = 0; i < vectorDim; ++i)
            {
                float sum = 0f;
                for (int m = 0; m < vectorDim; ++m)
                {
                    sum += R[i, m] * aCol[m]; // Правильный доступ: R[i,m] - элемент строки i
                }
                resCol[i] = sum;
            }
        }
        return result;
    }

    /// <summary>
    /// Нормализует все столбцы (векторы) в матрице к единичной норме.
    /// Использует TensorPrimitives для вычисления норм и деления.
    /// Обработка нулевых векторов: бросаем исключение (или можно оставить как есть).
    /// </summary>
    public void NormalizeColumns(MatrixFloat_ColumnMajor X)
    {
        int vectorDim = X.Dimensions[0];
        int nVectors = X.Dimensions[1];
        for (int k = 0; k < nVectors; ++k)
        {
            Span<float> col = X.GetColumn(k);
            float norm = TensorPrimitives.Norm(col);
            if (norm == 0)
                throw new InvalidOperationException($"Нулевой вектор в столбце {k}. Нормализация невозможна.");
            TensorPrimitives.Divide(col, norm, col);
        }
    }

    /// <summary>
    /// Вычисляет матрицу косинусных расстояний для набора векторов (для проверки структуры).
    /// Результат: [nVectors, nVectors], значения от 0 до 2.
    /// Нормализует вход заранее.
    /// </summary>
    public MatrixFloat_ColumnMajor ComputeCosineDistanceMatrix(MatrixFloat_ColumnMajor X)
    {
        NormalizeColumns(X); // Нормализация перед вычислением
        int nVectors = X.Dimensions[1];
        MatrixFloat_ColumnMajor result = new MatrixFloat_ColumnMajor(new[] { nVectors, nVectors });
        for (int i = 0; i < nVectors; ++i)
        {
            Span<float> vi = X.GetColumn(i);
            for (int j = i + 1; j < nVectors; ++j) // Симметрично, пропускаем i==j
            {
                Span<float> vj = X.GetColumn(j);
                float dot = TensorPrimitives.Dot(vi, vj);
                float dist = 1f - Math.Clamp(dot, -1f, 1f);
                result[i, j] = dist;
                result[j, i] = dist;
            }
            result[i, i] = 0f; // Сам с собой расстояние 0
        }
        return result;
    }

    // ------------ Дополнение для уникального сопоставления ------------

    /// <summary>
    /// Вычисляет уникальное (один-к-одному) сопоставление векторов из rotatedA к B на основе косинусного сходства.
    /// Использует Hungarian algorithm для оптимального максимального взвешенного matching.
    /// Учитывает кластеры: Hungarian глобально оптимизирует сумму сходств, что естественно сохраняет кластерные соответствия.
    /// Вход: rotatedA - повернутое множество A; B - целевое множество.
    /// Выход: int[] matching, где matching[i] - индекс вектора в B, сопоставленного вектору i в rotatedA.
    /// </summary>
    public int[] ComputeUniqueMatching(MatrixFloat_ColumnMajor rotatedA, MatrixFloat_ColumnMajor B)
    {
        int nVectors = rotatedA.Dimensions[1];
        float[,] similarity = ComputeCosineSimilarityMatrix(rotatedA, B);

        long[,] costMatrix = new long[nVectors, nVectors];
        for (int i = 0; i < nVectors; ++i)
            for (int j = 0; j < nVectors; ++j)
                costMatrix[i, j] = (long)(similarity[i, j] * -10000.0f);
        //costMatrix[i, j] = (int)(-MathF.Max(0f, similarity[i, j]) * 100.0f); // Только положительные

        int[] matching = HungarianAlgorithm.FindAssignments(costMatrix); // Предполагаемая реализация Hungarian

        // Проверка на полное matching
        for (int i = 0; i < nVectors; ++i)
            if (matching[i] == -1)
                throw new InvalidOperationException("Неполное сопоставление в ComputeUniqueMatching.");

        return matching;
    }
}