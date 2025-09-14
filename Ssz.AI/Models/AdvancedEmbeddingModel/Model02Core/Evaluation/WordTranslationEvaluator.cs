using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using Microsoft.Extensions.Logging;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Utils;
using Ssz.Utils.Logging;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Evaluation;

/// <summary>
/// Оценщик качества перевода слов для MUSE.
/// Реализует метрики точности перевода на различных уровнях (P@1, P@5, P@10).
/// Использует высокопроизводительные алгоритмы поиска ближайших соседей.
/// </summary>
public class WordTranslationEvaluator
{
    /// <summary>
    /// Оценка точности перевода слов.
    /// </summary>
    /// <param name="sourceEmbeddings">Исходные эмбеддинги</param>
    /// <param name="targetEmbeddings">Целевые эмбеддинги</param>
    /// <param name="sourceDictionary">Словарь исходного языка</param>
    /// <param name="targetDictionary">Словарь целевого языка</param>
    /// <param name="evaluationSize">Размер тестового набора</param>
    /// <returns>Точность перевода</returns>
    public float EvaluateAccuracy(MatrixFloat sourceEmbeddings, MatrixFloat targetEmbeddings,
                                Dictionary.Dictionary sourceDictionary, Dictionary.Dictionary? targetDictionary,
                                int evaluationSize = 1500)
    {
        var logger = LoggersSet.Default.UserFriendlyLogger;
        logger.LogDebug($"Оценка точности перевода на {evaluationSize} словах...");

        // Загрузка тестового словаря (если доступен)
        var testDictionary = LoadTestDictionary(sourceDictionary.Language, targetDictionary?.Language ?? "");

        if (testDictionary.Count == 0)
        {
            logger.LogWarning("Тестовый словарь не найден, используется частотная оценка");
            return EvaluateFrequencyBasedAccuracy(sourceEmbeddings, targetEmbeddings, evaluationSize);
        }

        // Нормализация эмбеддингов для косинусного расстояния
        var normalizedSource = sourceEmbeddings.Clone();
        var normalizedTarget = targetEmbeddings.Clone();
        Utils.MathUtils.NormalizeEmbeddings(normalizedSource);
        Utils.MathUtils.NormalizeEmbeddings(normalizedTarget);

        int correctTranslations = 0;
        int totalTranslations = 0;
        int embeddingDim = sourceEmbeddings.Dimensions[1];

        foreach (var (sourceWord, targetWord) in testDictionary.Take(evaluationSize))
        {   
            int sourceId = sourceDictionary.GetId(sourceWord);
            int targetId = targetDictionary?.GetId(targetWord) ?? -1;

            if (sourceId == -1 || targetId == -1)
                continue;

            // Поиск ближайшего соседа для исходного слова
            var sourceVector = normalizedSource.Data.AsMemory(sourceId * embeddingDim, embeddingDim);
            var nearestNeighbors = Utils.MathUtils.FindKNearestNeighbors(sourceVector, normalizedTarget, 1);

            if (nearestNeighbors.Length > 0 && nearestNeighbors[0] == targetId)
            {
                correctTranslations++;
            }

            totalTranslations++;
        }

        float accuracy = totalTranslations > 0 ? (float)correctTranslations / totalTranslations : 0.0f;
        logger.LogDebug($"Точность перевода: {accuracy:F4} ({correctTranslations}/{totalTranslations})");

        return accuracy;
    }

    /// <summary>
    /// Подробная оценка с метриками P@K (Precision at K).
    /// </summary>
    /// <param name="sourceEmbeddings">Исходные эмбеддинги</param>
    /// <param name="targetEmbeddings">Целевые эмбеддинги</param>
    /// <param name="sourceDictionary">Словарь исходного языка</param>
    /// <param name="targetDictionary">Словарь целевого языка</param>
    /// <param name="kValues">Значения K для оценки P@K</param>
    /// <param name="evaluationSize">Размер тестового набора</param>
    /// <returns>Результаты оценки для каждого K</returns>
    public Dictionary<int, float> EvaluatePrecisionAtK(MatrixFloat sourceEmbeddings, MatrixFloat targetEmbeddings,
                                                      Dictionary.Dictionary sourceDictionary, Dictionary.Dictionary? targetDictionary,
                                                      int[] kValues, int evaluationSize = 1500)
    {
        var logger = LoggersSet.Default.UserFriendlyLogger;
        logger.LogDebug($"Подробная оценка P@K для K={string.Join(",", kValues)}...");

        var testDictionary = LoadTestDictionary(sourceDictionary.Language, targetDictionary?.Language ?? "");
        var results = new Dictionary<int, float>();

        if (testDictionary.Count == 0)
        {
            logger.LogWarning("Тестовый словарь недоступен");
            foreach (int k in kValues)
                results[k] = 0.0f;
            return results;
        }

        // Нормализация эмбеддингов
        var normalizedSource = sourceEmbeddings.Clone();
        var normalizedTarget = targetEmbeddings.Clone();
        Utils.MathUtils.NormalizeEmbeddings(normalizedSource);
        Utils.MathUtils.NormalizeEmbeddings(normalizedTarget);

        int embeddingDim = sourceEmbeddings.Dimensions[1];
        int maxK = kValues.Max();
        var correctCounts = new Dictionary<int, int>();
        foreach (int k in kValues)
            correctCounts[k] = 0;

        int totalTranslations = 0;

        foreach (var (sourceWord, targetWord) in testDictionary.Take(evaluationSize))
        {
            int sourceId = sourceDictionary.GetId(sourceWord);
            int targetId = targetDictionary?.GetId(targetWord) ?? -1;

            if (sourceId == -1 || targetId == -1)
                continue;

            // Поиск K ближайших соседей
            var sourceVector = normalizedSource.Data.AsMemory(sourceId * embeddingDim, embeddingDim);
            var nearestNeighbors = Utils.MathUtils.FindKNearestNeighbors(sourceVector, normalizedTarget, maxK);

            // Проверка для каждого значения K
            foreach (int k in kValues)
            {
                if (nearestNeighbors.Take(k).Contains(targetId))
                {
                    correctCounts[k]++;
                }
            }

            totalTranslations++;
        }

        // Вычисление точности для каждого K
        foreach (int k in kValues)
        {
            results[k] = totalTranslations > 0 ? (float)correctCounts[k] / totalTranslations : 0.0f;
            logger.LogDebug($"P@{k}: {results[k]:F4}");
        }

        return results;
    }

    /// <summary>
    /// Оценка точности на основе частотности слов (fallback метод).
    /// Используется когда тестовый словарь недоступен.
    /// </summary>
    private float EvaluateFrequencyBasedAccuracy(MatrixFloat sourceEmbeddings, MatrixFloat targetEmbeddings, int evaluationSize)
    {
        var logger = LoggersSet.Default.UserFriendlyLogger;
        logger.LogDebug("Использование частотной оценки...");

        // Нормализация эмбеддингов
        var normalizedSource = sourceEmbeddings.Clone();
        var normalizedTarget = targetEmbeddings.Clone();
        Utils.MathUtils.NormalizeEmbeddings(normalizedSource);
        Utils.MathUtils.NormalizeEmbeddings(normalizedTarget);

        int sourceVocabSize = Math.Min(sourceEmbeddings.Dimensions[0], evaluationSize);
        int embeddingDim = sourceEmbeddings.Dimensions[1];

        float totalSimilarity = 0.0f;

        // Оценка качества отображения через среднее косинусное сходство
        for (int i = 0; i < sourceVocabSize; i++)
        {
            var sourceVector = normalizedSource.Data.AsMemory(i * embeddingDim, embeddingDim);
            var nearestNeighbors = Utils.MathUtils.FindKNearestNeighbors(sourceVector, normalizedTarget, 1);

            if (nearestNeighbors.Length > 0)
            {
                int targetId = nearestNeighbors[0];
                var targetVector = normalizedTarget.Data.AsSpan(targetId * embeddingDim, embeddingDim);
                float similarity = TensorPrimitives.Dot(sourceVector.Span, targetVector);
                totalSimilarity += similarity;
            }
        }

        float averageSimilarity = totalSimilarity / sourceVocabSize;
        logger.LogDebug($"Среднее косинусное сходство: {averageSimilarity:F4}");

        // Преобразование в оценку точности (эвристика)
        return MathF.Max(0.0f, (averageSimilarity - 0.5f) * 2.0f);
    }

    /// <summary>
    /// Загрузка тестового словаря для оценки.
    /// Ищет файлы словарей в стандартных местах.
    /// </summary>
    private List<(string source, string target)> LoadTestDictionary(string sourceLang, string targetLang)
    {
        var dictionary = new List<(string, string)>();

        // Стандартные пути к словарям
        var possiblePaths = new[]
        {
            $"data/dictionaries/{sourceLang}-{targetLang}.txt",
            $"data/dictionaries/{sourceLang}_{targetLang}.txt",
            $"dictionaries/{sourceLang}-{targetLang}.txt",
            $"eval/{sourceLang}-{targetLang}.5000-6500.txt"
        };

        foreach (var path in possiblePaths)
        {
            if (File.Exists(path))
            {
                try
                {
                    using var reader = new StreamReader(path);
                    string? line;

                    while ((line = reader.ReadLine()) != null)
                    {
                        var parts = line.Split('\t', ' ');
                        if (parts.Length >= 2)
                        {
                            dictionary.Add((parts[0].Trim(), parts[1].Trim()));
                        }
                    }

                    if (dictionary.Count > 0)
                    {
                        var logger = LoggersSet.Default.UserFriendlyLogger;
                        logger.LogInformation($"Загружен тестовый словарь: {path} ({dictionary.Count} пар)");
                        break;
                    }
                }
                catch (Exception ex)
                {
                    var logger = LoggersSet.Default.UserFriendlyLogger;
                    logger.LogWarning($"Ошибка загрузки словаря {path}: {ex.Message}");
                }
            }
        }

        return dictionary;
    }
}
