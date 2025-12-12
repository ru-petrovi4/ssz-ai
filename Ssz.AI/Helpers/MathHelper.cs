using Ssz.AI.Models;
using System;
using System.Collections.Generic;
using static TorchSharp.torch;

namespace Ssz.AI.Helpers;

public static class MathHelper
{
    /// <summary>
    ///     Radians to Degrees [0..360)
    /// </summary>
    /// <param name="radians"></param>
    /// <returns></returns>
    public static float RadiansToDegrees(float radians)
    {
        float degrees = 180 * radians / MathF.PI;                        
        return NormalizeAngleDegrees(degrees);
    }

    /// <summary>
    ///     Degrees [0..360)
    /// </summary>
    /// <param name="degrees"></param>
    /// <returns></returns>
    public static float NormalizeAngleDegrees(float degrees)
    {
        degrees = degrees % 360.0f;
        if (degrees < 0.000001f)
        {
            if (degrees < -0.000001f)
                degrees += 360.0f;
            else
                degrees = 0.0f;
        }
        return degrees;
    }

    /// <summary>
    ///     Degrees to Radians [-pi, pi)
    /// </summary>
    /// <param name="degrees"></param>
    /// <returns></returns>
    public static float DegreesToRadians(float degrees)
    {
        float radians = MathF.PI * degrees / 180.0f;            
        return NormalizeAngle(radians);
    }

    /// <summary>
    ///     Returns Radians [-pi, pi)
    /// </summary>
    /// <param name="v"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public static float NormalizeAngle(float radians)
    {
        radians = radians % (2.0f * MathF.PI);
        if (radians > MathF.PI - 0.00001f)
        {
            if (radians > MathF.PI + 0.00001f)
                radians -= 2 * MathF.PI;
            else
                radians = -MathF.PI;
        }
        else if (radians < -MathF.PI + 0.00001f)
        {
            if (radians < -MathF.PI - 0.00001f)
                radians += 2 * MathF.PI;
            else
                radians = -MathF.PI;
        }
        return radians;
    }

    public static float GetInterpolatedValue(float[] points, float x)
    {   
        if (x < 0.00001f)
            return points[0];
        int xi = (int)x;
        if (xi + 1 >= points.Length)
            return points[points.Length - 1];
        return points[xi] + (points[xi + 1] - points[xi]) * (x - (float)xi);
    }

    public static GradientInPoint GetInterpolatedGradient(double centerX, double centerY, DenseMatrix<GradientInPoint> gradientMatrix)
    {   
        int x = (int)centerX;
        int y = (int)centerY;
        if (x < 0 ||
                y < 0 ||
                x + 1 >= gradientMatrix.Dimensions[0] ||
                y + 1 >= gradientMatrix.Dimensions[1])
            return new();

        double tx = centerX - x;
        double ty = centerY - y;
        double gradX = (1 - tx) * (1 - ty) * gradientMatrix[x, y].GradX +
            tx * (1 - ty) * gradientMatrix[x + 1, y].GradX +
            (1 - tx) * ty * gradientMatrix[x, y + 1].GradX +
            tx * ty * gradientMatrix[x + 1, y + 1].GradX;
        double gradY = (1 - tx) * (1 - ty) * gradientMatrix[x, y].GradY +
            tx * (1 - ty) * gradientMatrix[x + 1, y].GradY +
            (1 - tx) * ty * gradientMatrix[x, y + 1].GradY +
            tx * ty * gradientMatrix[x + 1, y + 1].GradY;

        double magnitude = Math.Sqrt(gradX * gradX + gradY * gradY);

        double angle = Math.Atan2(gradY, gradX); // Угол в радианах    

        if (angle > Math.PI - 0.000001)
            angle = -Math.PI;

        return new GradientInPoint
        {
            GradX = gradX,
            GradY = gradY,
            Angle = angle,
            Magnitude = magnitude
        };
    }

    /// <summary>
    ///     Returns [-pi, pi)
    /// </summary>
    /// <param name="centerX"></param>
    /// <param name="centerY"></param>
    /// <param name="gradientMatrix"></param>
    /// <returns></returns>
    public static (double magnitude, double angle) GetInterpolatedGradient_Obsolete(double centerX, double centerY, GradientInPoint[,] gradientMatrix)
    {
        int x = (int)centerX;
        int y = (int)centerY;
        if (x < 0 ||
                y < 0 ||
                x + 1 >= gradientMatrix.GetLength(0) ||
                y + 1 >= gradientMatrix.GetLength(1))
            return (0.0, 0.0);

        double tx = centerX - x;
        double ty = centerY - y;
        double gradX = (1 - tx) * (1 - ty) * gradientMatrix[x, y].GradX +
            tx * (1 - ty) * gradientMatrix[x + 1, y].GradX +
            (1 - tx) * ty * gradientMatrix[x, y + 1].GradX +
            tx * ty * gradientMatrix[x + 1, y + 1].GradX;
        double gradY = (1 - tx) * (1 - ty) * gradientMatrix[x, y].GradY +
            tx * (1 - ty) * gradientMatrix[x + 1, y].GradY +
            (1 - tx) * ty * gradientMatrix[x, y + 1].GradY +
            tx * ty * gradientMatrix[x + 1, y + 1].GradY;

        double magnitude = Math.Sqrt(gradX * gradX + gradY * gradY);

        double angle = Math.Atan2(gradY, gradX); // Угол в радианах    

        if (angle > Math.PI - 0.000001)
            angle = -Math.PI;

        return (magnitude, angle);
    }

    /// <summary>
    /// Метод для выбора 30 максимальных значений в массиве float длиной 300
    /// и модификации массива: устанавливает выбранные в 1.0f, остальные в 0.0f.
    /// Работает in-place; использует reusable PriorityQueue для zero-allocation после init.
    /// minHeap = new PriorityQueue<int, float>(k + 1);
    /// minHeap Должен быть пуст. На выходе он пуст.
    /// </summary>
    /// <param name="data">
    /// Входной/выходной Span<float>.
    /// - n=300: длина, фиксирована.
    /// - data[i]: исходное значение на позиции i (0 <= i < n); после: 1.0f для топ-30 max, иначе 0.0f.
    /// </param>
    /// <param name="k">Количество топ-элементов для выбора</param>
    /// <remarks>
    /// Сложность: O(n log k + k log k) на операцию, где 
    /// - n=300: сканирование массива;
    /// - k=30: размер heap; log k ≈ 4.9 (log₂(30));
    /// - + k log k: на очистку/извлечение.
    /// Общее ≈ 300*4.9 + 30*4.9 ≈ 1650 операций — оптимально.
    /// Reuse: очищает heap via Dequeue (O(k log k)); для 1000+ операций аллокации только init (200 байт).
    /// Если thread-multi: используйте lock или ThreadLocal<PriorityQueue<int, float>>.
    /// Формула очистки: while (heap.Count > 0) { Dequeue(); } — log k per Dequeue, k раз.
    /// </remarks>
    /// <exception cref="ArgumentException">Если data.Length != 300.</exception>
    public static void SelectTopKMaxAndSetToOne(float[] data, int k, PriorityQueue<int, float> pq)
    {
        //var dataTensor = tensor(data, dtype: ScalarType.Float32, device: CPU);
        //var idxSpan = dataTensor.topk(k, dim: 0, largest: true, sorted: false).indexes.data<long>();
        //Array.Clear(data);
        //for (int i = 0; i < idxSpan.Count; i += 1)
        //{
        //    data[idxSpan[i]] = 1.0f;
        //}

        // Шаг 1: Заполняем heap начальными k+1 элементами (i=0..30).
        // Enqueue: O(log (k+1)) per вставка; всего O((k+1) log k).        
        int currentIndex = 0;
        for (; currentIndex < k; currentIndex += 1)
        {
            float currentValue = data[currentIndex];

            pq.Enqueue(currentIndex, currentValue); // Элемент=индекс, приоритет=значение            
        }
        // Шаг 2: Оставшиеся элементы (i=31..299).
        // Для каждого: если data[i] >= min в heap, заменяем (2 * O(log k)).
        for (; currentIndex < data.Length; currentIndex += 1)
        {
            float currentValue = data[currentIndex];

            pq.TryPeek(out var minIndex, out var minValue);
            if (currentValue > minValue)
            {
                pq.Dequeue();
                pq.Enqueue(currentIndex, currentValue);
            }
        }

        // Шаг 3: Обнуляем весь массив.
        // Формула: для i=0..n-1: data[i] = 0.0f; O(n) времени.
        Array.Clear(data);

        // Шаг 4: Извлекаем k=30 топ-индексов и устанавливаем 1.0f.
        // Пропускаем первый Dequeue (31-й max), затем 30 топ.
        // Dequeue уже очищает; O(k log k).        
        foreach (var item in pq.UnorderedItems)
        {
            data[item.Element] = 1.0f;
        }
        pq.Clear();
        // Heap теперь пуст для следующей операции.
    }

    public static double NormalPdf(double x, double mu, double sigma)
    {
        // Защитная проверка: стандартное отклонение не может быть <= 0.
        if (sigma <= 0.0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(sigma),
                "Стандартное отклонение sigma должно быть строго положительным."
            );
        }

        // Вычисляем отклонение от среднего: diff = x - mu.
        double diff = x - mu;

        // Вычисляем квадрат отклонения: diffSquared = diff * diff.
        double diffSquared = diff * diff;

        // Вычисляем квадрат стандартного отклонения: sigmaSquared = sigma * sigma.
        double sigmaSquared = sigma * sigma;

        // В знаменателе показателя экспоненты стоит 2 * sigma^2.
        double twoSigmaSquared = 2.0f * sigmaSquared;

        // Вычисляем показатель экспоненты:
        // exponent = - diff^2 / (2 * sigma^2).
        double exponent = -diffSquared / twoSigmaSquared;

        // Нормирующий коэффициент: 1 / (sigma * sqrt(2 * π)).
        double normCoeff = 1.0f / (sigma * SqrtTwoPi);

        // Конечное значение плотности: normCoeff * exp(exponent).
        double pdf = normCoeff * Math.Exp(exponent);

        return pdf;
    }

    public static float NormalPdfF(float x, float mu, float sigma)
    {
        // Защитная проверка: стандартное отклонение не может быть <= 0.
        if (sigma <= 0.0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(sigma),
                "Стандартное отклонение sigma должно быть строго положительным."
            );
        }

        // Вычисляем отклонение от среднего: diff = x - mu.
        float diff = x - mu;

        // Вычисляем квадрат отклонения: diffSquared = diff * diff.
        float diffSquared = diff * diff;

        // Вычисляем квадрат стандартного отклонения: sigmaSquared = sigma * sigma.
        float sigmaSquared = sigma * sigma;

        // В знаменателе показателя экспоненты стоит 2 * sigma^2.
        float twoSigmaSquared = 2.0f * sigmaSquared;

        // Вычисляем показатель экспоненты:
        // exponent = - diff^2 / (2 * sigma^2).
        float exponent = -diffSquared / twoSigmaSquared;

        // Нормирующий коэффициент: 1 / (sigma * sqrt(2 * π)).
        float normCoeff = 1.0f / (sigma * SqrtTwoPi);

        // Конечное значение плотности: normCoeff * exp(exponent).
        float pdf = normCoeff * MathF.Exp(exponent);

        return pdf;
    }

    // Предвычисленные константы для плотности.
    // SqrtTwoPi = sqrt(2 * π).
    private static readonly float SqrtTwoPi = MathF.Sqrt(2.0f * MathF.PI);
}
