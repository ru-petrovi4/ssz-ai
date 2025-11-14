using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Models.AdvancedEmbeddingModel2.EmbeddingsEvaluation;

/// <summary>
/// Класс для вычисления корреляции Спирмена между двумя наборами оценок
/// </summary>
public class SpearmanCorrelation
{
    /// <summary>
    /// Вычисляет коэффициент корреляции Спирмена (ρ) между двумя наборами значений
    /// Корреляция Спирмена измеряет монотонную зависимость между переменными
    /// на основе рангов значений, а не самих значений.
    /// 
    /// Формула: ρ = 1 - (6 * Σ(d²)) / (n * (n² - 1))
    /// где:
    ///   d - разность рангов соответствующих значений
    ///   n - количество пар наблюдений
    /// </summary>
    /// <param name="x">Первый набор значений (например, человеческие оценки)</param>
    /// <param name="y">Второй набор значений (например, косинусные близости)</param>
    /// <returns>Коэффициент корреляции Спирмена в диапазоне [-1, 1]</returns>
    public double Calculate(List<double> x, List<double> y)
    {
        // Проверяем, что наборы данных имеют одинаковый размер
        if (x.Count != y.Count)
        {
            throw new ArgumentException(
                $"Наборы данных должны иметь одинаковый размер. " +
                $"x.Count = {x.Count}, y.Count = {y.Count}");
        }

        // Проверяем, что наборы не пустые
        if (x.Count == 0)
        {
            throw new ArgumentException("Наборы данных не могут быть пустыми");
        }

        int n = x.Count;

        // Вычисляем ранги для первого набора значений
        // Ранг - это позиция значения в упорядоченном списке
        List<double> ranksX = CalculateRanks(x);

        // Вычисляем ранги для второго набора значений
        List<double> ranksY = CalculateRanks(y);

        // Вычисляем сумму квадратов разностей рангов
        // sumDSquared = Σ((ranksX[i] - ranksY[i])²) для всех i
        double sumDSquared = 0.0;

        for (int i = 0; i < n; i += 1)
        {
            // Вычисляем разность рангов для i-го элемента
            double d = ranksX[i] - ranksY[i];

            // Добавляем квадрат разности к сумме
            sumDSquared += d * d;
        }

        // Вычисляем коэффициент корреляции Спирмена по формуле
        // ρ = 1 - (6 * sumDSquared) / (n * (n² - 1))
        double spearmanRho = 1.0 - (6.0 * sumDSquared) / (n * (n * n - 1));

        return spearmanRho;
    }

    /// <summary>
    /// Вычисляет ранги для набора значений с учетом одинаковых значений (ties)
    /// При наличии одинаковых значений им присваивается средний ранг
    /// </summary>
    /// <param name="values">Набор значений для ранжирования</param>
    /// <returns>Список рангов, соответствующих исходным значениям</returns>
    private List<double> CalculateRanks(List<double> values)
    {
        int n = values.Count;

        // Создаем список пар (значение, исходный_индекс)
        // Это нужно для того, чтобы после сортировки знать исходную позицию каждого элемента
        List<Tuple<double, int>> indexedValues = new List<Tuple<double, int>>();

        for (int index = 0; index < n; index += 1)
        {
            indexedValues.Add(new Tuple<double, int>(values[index], index));
        }

        // Сортируем пары по значениям в возрастающем порядке
        indexedValues.Sort((a, b) => a.Item1.CompareTo(b.Item1));

        // Создаем массив для хранения рангов в исходном порядке
        double[] ranks = new double[n];

        // Присваиваем ранги с учетом одинаковых значений (ties)
        int i = 0;
        while (i < n)
        {
            // Находим группу элементов с одинаковыми значениями
            int j = i;

            // j указывает на первый элемент, отличный от текущего
            while (j < n && Math.Abs(indexedValues[j].Item1 - indexedValues[i].Item1) < 1e-10)
            {
                j += 1;
            }

            // Количество элементов с одинаковым значением
            int tieCount = j - i;

            // Вычисляем средний ранг для группы одинаковых значений
            // Ранги начинаются с 1, поэтому добавляем 1 к индексу
            // Средний ранг = (первый_ранг + последний_ранг) / 2
            double averageRank = (i + 1 + j) / 2.0;

            // Присваиваем средний ранг всем элементам группы
            for (int k = i; k < j; k += 1)
            {
                // Получаем исходный индекс элемента
                int originalIndex = indexedValues[k].Item2;

                // Присваиваем ранг элементу в исходной позиции
                ranks[originalIndex] = averageRank;
            }

            // Переходим к следующей группе
            i = j;
        }

        return ranks.ToList();
    }
}

/// <summary>
/// Класс для вычисления коэффициента корреляции Пирсона
/// </summary>
public class PearsonCorrelation
{
    /// <summary>
    /// Вычисляет коэффициент корреляции Пирсона (r) между двумя наборами значений
    /// Корреляция Пирсона измеряет линейную зависимость между переменными
    /// 
    /// Формула: r = Σ((x[i] - mean_x) * (y[i] - mean_y)) / 
    ///              sqrt(Σ((x[i] - mean_x)²) * Σ((y[i] - mean_y)²))
    /// где:
    ///   mean_x - среднее значение x
    ///   mean_y - среднее значение y
    /// </summary>
    /// <param name="x">Первый набор значений</param>
    /// <param name="y">Второй набор значений</param>
    /// <returns>Коэффициент корреляции Пирсона в диапазоне [-1, 1]</returns>
    public double Calculate(List<double> x, List<double> y)
    {
        if (x.Count != y.Count)
        {
            throw new ArgumentException("Наборы данных должны иметь одинаковый размер");
        }

        if (x.Count == 0)
        {
            throw new ArgumentException("Наборы данных не могут быть пустыми");
        }

        int n = x.Count;

        // Вычисляем среднее значение для x
        // mean_x = (Σ x[i]) / n для всех i
        double meanX = 0.0;
        for (int i = 0; i < n; i += 1)
        {
            meanX += x[i];
        }
        meanX /= n;

        // Вычисляем среднее значение для y
        // mean_y = (Σ y[i]) / n для всех i
        double meanY = 0.0;
        for (int i = 0; i < n; i += 1)
        {
            meanY += y[i];
        }
        meanY /= n;

        // Вычисляем ковариацию
        // covariance = Σ((x[i] - mean_x) * (y[i] - mean_y)) для всех i
        double covariance = 0.0;
        for (int i = 0; i < n; i += 1)
        {
            covariance += (x[i] - meanX) * (y[i] - meanY);
        }

        // Вычисляем стандартное отклонение для x
        // stdDev_x = sqrt(Σ((x[i] - mean_x)²)) для всех i
        double varianceX = 0.0;
        for (int i = 0; i < n; i += 1)
        {
            double deviation = x[i] - meanX;
            varianceX += deviation * deviation;
        }
        double stdDevX = Math.Sqrt(varianceX);

        // Вычисляем стандартное отклонение для y
        // stdDev_y = sqrt(Σ((y[i] - mean_y)²)) для всех i
        double varianceY = 0.0;
        for (int i = 0; i < n; i += 1)
        {
            double deviation = y[i] - meanY;
            varianceY += deviation * deviation;
        }
        double stdDevY = Math.Sqrt(varianceY);

        // Проверяем деление на ноль
        if (stdDevX < 1e-10 || stdDevY < 1e-10)
        {
            return 0.0;
        }

        // Вычисляем коэффициент корреляции Пирсона
        // r = covariance / (stdDev_x * stdDev_y)
        double pearsonR = covariance / (stdDevX * stdDevY);

        return pearsonR;
    }
}