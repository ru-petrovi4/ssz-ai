using System;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using Ssz.AI.Models;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Utils;

/// <summary>
/// Утилиты для высокопроизводительных математических операций.
/// Использует System.Numerics.Tensors.TensorPrimitives и MathNet для SIMD-ускорения.
/// Оптимизирован для работы с MatrixFloat_RowMajor и операций линейной алгебры.
/// </summary>
public static class MathUtils
{
    /// <summary>
    /// Нормализация каждой строки матрицы (L2 нормализация).
    /// Используется для нормализации эмбеддингов перед обучением.
    /// Применяет векторизованные операции для максимальной производительности.
    /// </summary>
    /// <param name="matrix">Матрица для нормализации</param>
    public static void NormalizeEmbeddings(MatrixFloat_RowMajor matrix)
    {
        int numRows = matrix.Dimensions[0];
        int numCols = matrix.Dimensions[1];

        // Параллельная обработка каждой строки
        Parallel.For(0, numRows, i =>
        {
            var row = matrix.Data.AsSpan(i * numCols, numCols);

            // Вычисление L2 нормы через TensorPrimitives
            float norm = MathF.Sqrt(TensorPrimitives.SumOfSquares(row));

            // Нормализация только если норма больше epsilon
            if (norm > 1e-8f)
            {
                // Векторизованное деление на норму
                TensorPrimitives.Divide(row, norm, row);
            }
        });
    }

    /// <summary>
    /// Высокопроизводительное матричное умножение: result = matrixA * matrixB.
    /// Использует блочное умножение и параллелизацию для оптимальной производительности.
    /// Оптимизировано для кэш-эффективности и SIMD операций.
    /// </summary>
    /// <param name="matrixA">Левая матрица [M, K]</param>
    /// <param name="matrixB">Правая матрица [K, N]</param>
    /// <param name="result">Результирующая матрица [M, N]</param>
    public static void MatrixMultiply(MatrixFloat_RowMajor matrixA, MatrixFloat_RowMajor matrixB, MatrixFloat_RowMajor result)
    {
        int M = matrixA.Dimensions[0];
        int K = matrixA.Dimensions[1];
        int N = matrixB.Dimensions[1];

        if (matrixB.Dimensions[0] != K)
            throw new ArgumentException("Несовместимые размерности матриц для умножения");

        if (result.Dimensions[0] != M || result.Dimensions[1] != N)
            throw new ArgumentException("Неверная размерность результирующей матрицы");

        // Блочное матричное умножение для кэш-эффективности
        const int blockSize = 64; // Оптимальный размер блока для L1 кэша

        // Обнуление результирующей матрицы
        result.Clear();

        // Параллельная обработка блоков по строкам
        Parallel.For(0, (M + blockSize - 1) / blockSize, iBlock =>
        {
            int iStart = iBlock * blockSize;
            int iEnd = Math.Min(iStart + blockSize, M);

            for (int jBlock = 0; jBlock < (N + blockSize - 1) / blockSize; jBlock++)
            {
                int jStart = jBlock * blockSize;
                int jEnd = Math.Min(jStart + blockSize, N);

                for (int kBlock = 0; kBlock < (K + blockSize - 1) / blockSize; kBlock++)
                {
                    int kStart = kBlock * blockSize;
                    int kEnd = Math.Min(kStart + blockSize, K);

                    // Умножение блоков
                    MultiplyBlock(matrixA, matrixB, result,
                                iStart, iEnd, jStart, jEnd, kStart, kEnd);
                }
            }
        });
    }

    /// <summary>
    /// Умножение блоков матриц с использованием SIMD операций.
    /// Внутренняя функция для оптимизированного матричного умножения.
    /// </summary>
    private static void MultiplyBlock(MatrixFloat_RowMajor matrixA, MatrixFloat_RowMajor matrixB, MatrixFloat_RowMajor result,
                                    int iStart, int iEnd, int jStart, int jEnd, int kStart, int kEnd)
    {
        for (int i = iStart; i < iEnd; i++)
        {
            for (int k = kStart; k < kEnd; k++)
            {
                float aik = matrixA[i, k];
                if (MathF.Abs(aik) < 1e-8f) continue; // Пропуск нулевых элементов

                // Векторизованное умножение строки на скаляр и добавление к результату
                var resultRow = result.Data.AsSpan(i * result.Dimensions[1] + jStart, jEnd - jStart);
                var bRow = matrixB.Data.AsSpan(k * matrixB.Dimensions[1] + jStart, jEnd - jStart);

                // result[i, j:jEnd] += aik * matrixB[k, j:jEnd]
                TensorPrimitives.MultiplyAdd(bRow, aik, resultRow, resultRow);
            }
        }
    }

    /// <summary>
    /// Procrustes анализ для нахождения оптимальной ортогональной матрицы.
    /// Решает задачу: W = argmin ||A*W - B||_F при условии W^T*W = I.
    /// Использует SVD разложение через MathNet для точного решения.
    /// </summary>
    /// <param name="sourceMatrix">Исходная матрица A [d, n]</param>
    /// <param name="targetMatrix">Целевая матрица B [d, n]</param>
    /// <returns>Оптимальная ортогональная матрица W [d, d]</returns>
    public static MatrixFloat_RowMajor ProcrustesAlignment(MatrixFloat_RowMajor sourceMatrix, MatrixFloat_RowMajor targetMatrix)
    {
        if (sourceMatrix.Dimensions[0] != targetMatrix.Dimensions[0] ||
            sourceMatrix.Dimensions[1] != targetMatrix.Dimensions[1])
            throw new ArgumentException("Матрицы должны иметь одинаковые размерности");

        // Конвертация в MathNet матрицы для SVD
        var A = ConvertToMathNet(sourceMatrix);
        var B = ConvertToMathNet(targetMatrix);

        // Вычисление M = B * A^T
        var M = B.Multiply(A.Transpose());

        // SVD разложение: M = U * Σ * V^T
        var svd = M.Svd(computeVectors: true);

        // Оптимальная матрица: W = U * V^T
        var W = svd.U.Multiply(svd.VT);

        // Корректировка для обеспечения det(W) = 1 (правильная ориентация)
        if (W.Determinant() < 0)
        {
            // Изменение знака последнего столбца U
            var U = svd.U;
            int lastCol = U.ColumnCount - 1;
            for (int i = 0; i < U.RowCount; i++)
            {
                U[i, lastCol] = -U[i, lastCol];
            }
            W = U.Multiply(svd.VT);
        }

        // Конвертация обратно в MatrixFloat_RowMajor
        return ConvertFromMathNet(W);
    }

    /// <summary>
    /// Ортогонализация матрицы через QR разложение.
    /// Обеспечивает ортогональность столбцов для сохранения геометрических свойств.
    /// </summary>
    /// <param name="matrix">Матрица для ортогонализации (изменяется на месте)</param>
    public static void OrthogonalizeMatrix(MatrixFloat_RowMajor matrix)
    {
        if (matrix.Dimensions[0] != matrix.Dimensions[1])
            throw new ArgumentException("Ортогонализация возможна только для квадратных матриц");

        // Конвертация в MathNet для QR разложения
        var mathNetMatrix = ConvertToMathNet(matrix);

        // QR разложение: A = Q * R
        var qr = mathNetMatrix.QR();
        var Q = qr.Q;

        // Копирование ортогональной матрицы Q обратно
        int size = matrix.Dimensions[0];
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                matrix[i, j] = Q[i, j];
            }
        }
    }

    /// <summary>
    /// Вычисление косинусного расстояния между двумя векторами.
    /// Использует оптимизированные SIMD операции для максимальной скорости.
    /// </summary>
    /// <param name="vector1">Первый вектор</param>
    /// <param name="vector2">Второй вектор</param>
    /// <returns>Косинусное расстояние (0 - идентичные, 2 - противоположные)</returns>
    public static float CosineDistance(Span<float> vector1, Span<float> vector2)
    {
        //if (vector1.Length != vector2.Length)
        //    throw new ArgumentException("Векторы должны иметь одинаковую длину");

        // Скалярное произведение
        float cosineSimilarity = TensorPrimitives.CosineSimilarity(vector1, vector2);

        //// Нормы векторов
        //float norm1 = MathF.Sqrt(TensorPrimitives.SumOfSquares(vector1));
        //float norm2 = MathF.Sqrt(TensorPrimitives.SumOfSquares(vector2));

        //// Косинусное сходство
        //float cosineSimilarity = dotProduct / (norm1 * norm2 + 1e-8f);

        // Косинусное расстояние
        return 1.0f - cosineSimilarity;
    }

    /// <summary>
    /// Поиск k ближайших соседей через косинусное расстояние.
    /// Оптимизированная реализация для больших матриц эмбеддингов.
    /// </summary>
    /// <param name="queryVector">Вектор запроса</param>
    /// <param name="embeddings">Матрица эмбеддингов для поиска</param>
    /// <param name="k">Количество ближайших соседей</param>
    /// <returns>Индексы k ближайших соседей, отсортированные по расстоянию</returns>
    public static int[] FindKNearestNeighbors(Memory<float> queryVector, MatrixFloat_RowMajor embeddings, int k)
    {
        int vocabSize = embeddings.Dimensions[0];
        int embeddingDim = embeddings.Dimensions[1];

        if (queryVector.Length != embeddingDim)
            throw new ArgumentException("Размерность вектора запроса не соответствует размерности эмбеддингов");

        // Вычисление расстояний до всех эмбеддингов
        var distances = new (float distance, int index)[vocabSize];
        
        Parallel.For(0, vocabSize, i =>
        {
            var embeddingVector = embeddings.Data.AsSpan(i * embeddingDim, embeddingDim);
            float distance = CosineDistance(queryVector.Span, embeddingVector);
            distances[i] = (distance, i);
        });

        // Частичная сортировка для получения k минимальных элементов
        Array.Sort(distances, (a, b) => a.distance.CompareTo(b.distance));

        // Возврат индексов k ближайших соседей
        var result = new int[Math.Min(k, vocabSize)];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = distances[i].index;
        }

        return result;
    }

    /// <summary>
    /// Конвертация MatrixFloat_RowMajor в MathNet Matrix для использования специализированных алгоритмов.
    /// </summary>
    /// <param name="matrix">Исходная MatrixFloat_RowMajor</param>
    /// <returns>MathNet Matrix</returns>
    private static Matrix<float> ConvertToMathNet(MatrixFloat_RowMajor matrix)
    {
        int rows = matrix.Dimensions[0];
        int cols = matrix.Dimensions[1];
        var result = Matrix<float>.Build.Dense(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = matrix[i, j];
            }
        }

        return result;
    }

    /// <summary>
    /// Конвертация MathNet Matrix обратно в MatrixFloat_RowMajor.
    /// </summary>
    /// <param name="matrix">MathNet Matrix</param>
    /// <returns>MatrixFloat_RowMajor</returns>
    private static MatrixFloat_RowMajor ConvertFromMathNet(Matrix<float> matrix)
    {
        var result = new MatrixFloat_RowMajor(new[] { matrix.RowCount, matrix.ColumnCount });

        for (int i = 0; i < matrix.RowCount; i++)
        {
            for (int j = 0; j < matrix.ColumnCount; j++)
            {
                result[i, j] = matrix[i, j];
            }
        }

        return result;
    }

    /// <summary>
    /// Копирование одной матрицы в другую.
    /// </summary>
    /// <param name="source">Исходная матрица</param>
    /// <param name="destination">Целевая матрица</param>
    public static void CopyMatrix(MatrixFloat_RowMajor source, MatrixFloat_RowMajor destination)
    {
        if (source.Dimensions[0] != destination.Dimensions[0] ||
            source.Dimensions[1] != destination.Dimensions[1])
            throw new ArgumentException("Матрицы должны иметь одинаковые размерности");

        Array.Copy(source.Data, destination.Data, source.Data.Length);
    }
}