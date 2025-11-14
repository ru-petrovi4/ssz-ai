using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace Ssz.AI.Models.AdvancedEmbeddingModel2.EmbeddingsEvaluation;

/// <summary>
/// Утилиты для работы с различными форматами эмбеддингов
/// </summary>
public static class EmbeddingUtilities
{
    /// <summary>
    /// Вычисляет и выводит корреляции между человеческими оценками и косинусной близостью
    /// </summary>
    /// <param name="pairs">Список пар слов с вычисленными косинусными близостями</param>
    public static void EvaluateCorrelation(List<WordPair> pairs, ILogger logger)
    {
        // Отфильтровываем пары, для которых не удалось вычислить косинусную близость
        List<WordPair> validPairs = pairs
            .Where(p => p.CosineSimilarity != 0.0)
            .ToList();

        if (validPairs.Count == 0)
        {
            logger.LogInformation("Нет валидных пар для вычисления корреляции");
            return;
        }

        logger.LogInformation($"Валидных пар для оценки: {validPairs.Count}");

        // Извлекаем человеческие оценки
        List<double> humanScores = validPairs
            .Select(p => p.HumanScore)
            .ToList();

        // Извлекаем косинусные близости
        List<double> cosineSimilarities = validPairs
            .Select(p => p.CosineSimilarity)
            .ToList();

        // Вычисляем корреляцию Спирмена
        SpearmanCorrelation spearman = new SpearmanCorrelation();
        double spearmanRho = spearman.Calculate(humanScores, cosineSimilarities);

        // Вычисляем корреляцию Пирсона
        PearsonCorrelation pearson = new PearsonCorrelation();
        double pearsonR = pearson.Calculate(humanScores, cosineSimilarities);

        // Выводим результаты
        logger.LogInformation($"Корреляция Спирмена (ρ): {spearmanRho:F4}");
        logger.LogInformation($"Корреляция Пирсона (r):  {pearsonR:F4}");

        // Интерпретация результатов
        logger.LogInformation("\nИнтерпретация корреляции Спирмена:");
        InterpretCorrelation(spearmanRho, logger);
    }

    /// <summary>
    /// Интерпретирует значение коэффициента корреляции
    /// </summary>
    /// <param name="correlation">Значение коэффициента корреляции</param>
    private static void InterpretCorrelation(double correlation, ILogger logger)
    {
        double absCorr = Math.Abs(correlation);

        string strength;
        if (absCorr >= 0.7)
        {
            strength = "Сильная";
        }
        else if (absCorr >= 0.4)
        {
            strength = "Умеренная";
        }
        else if (absCorr >= 0.2)
        {
            strength = "Слабая";
        }
        else
        {
            strength = "Очень слабая";
        }

        string direction = correlation >= 0 ? "положительная" : "отрицательная";

        logger.LogInformation($"{strength} {direction} корреляция");
    }

    /// <summary>
    /// Сохраняет результаты в CSV файл
    /// </summary>
    /// <param name="pairs">Список пар слов с результатами</param>
    /// <param name="outputPath">Путь к выходному файлу</param>
    public static void SaveResults(List<WordPair> pairs, string outputPath, ILogger logger)
    {
        logger.LogInformation($"Сохранение результатов в файл: {outputPath}");

        using (StreamWriter writer = new StreamWriter(File.Create(outputPath), new UTF8Encoding(true)))
        {
            // Записываем заголовок
            writer.WriteLine("#Word1,Word2,HumanScore,CosineSimilarity");

            // Записываем данные
            foreach (WordPair pair in pairs)
            {
                writer.WriteLine($"{pair.Word1},{pair.Word2}," +
                               $"{pair.HumanScore:F4},{pair.CosineSimilarity:F4}");
            }
        }

        logger.LogInformation($"Результаты сохранены: {pairs.Count} записей");
    }

    /// <summary>
    /// Получает статистику о файле эмбеддингов
    /// </summary>
    /// <param name="filePath">Путь к файлу эмбеддингов</param>
    public static void PrintEmbeddingsStatistics(string filePath, ILogger logger)
    {
        logger.LogInformation("==========================================================");
        logger.LogInformation($"Статистика файла эмбеддингов: {filePath}");
        logger.LogInformation("==========================================================");

        int wordCount = 0;
        int dimension = 0;
        List<double> norms = new List<double>();

        using (StreamReader reader = new StreamReader(filePath))
        {
            string line;

            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                string[] parts = line.Split(new[] { ' ', '\t' }, 
                    StringSplitOptions.RemoveEmptyEntries);

                if (parts.Length < 2)
                {
                    continue;
                }

                wordCount += 1;

                if (dimension == 0)
                {
                    dimension = parts.Length - 1;
                }

                // Вычисляем норму вектора
                double normSquared = 0.0;

                for (int i = 1; i < parts.Length; i += 1)
                {
                    if (float.TryParse(parts[i], out float value))
                    {
                        normSquared += value * value;
                    }
                }

                norms.Add(Math.Sqrt(normSquared));
            }
        }

        // Вычисляем статистики норм
        norms.Sort();

        double minNorm = norms[0];
        double maxNorm = norms[norms.Count - 1];
        double medianNorm = norms[norms.Count / 2];

        double sumNorm = 0.0;
        for (int i = 0; i < norms.Count; i += 1)
        {
            sumNorm += norms[i];
        }
        double meanNorm = sumNorm / norms.Count;

        // Выводим результаты
        logger.LogInformation($"Количество слов: {wordCount}");
        logger.LogInformation($"Размерность векторов: {dimension}");
        logger.LogInformation($"\nСтатистика норм векторов:");
        logger.LogInformation($"  Минимум: {minNorm:F4}");
        logger.LogInformation($"  Максимум: {maxNorm:F4}");
        logger.LogInformation($"  Среднее: {meanNorm:F4}");
        logger.LogInformation($"  Медиана: {medianNorm:F4}");
        logger.LogInformation("==========================================================");
    }
}