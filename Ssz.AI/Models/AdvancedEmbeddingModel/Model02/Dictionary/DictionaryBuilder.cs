using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02.Utils;
using Ssz.Utils.Logging;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02.Dictionary;

/// <summary>
/// Строитель словарей переводов для MUSE.
/// Реализует алгоритмы поиска ближайших соседей и CSLS (Cross-domain Similarity Local Scaling).
/// Оптимизирован для работы с большими матрицами эмбеддингов.
/// </summary>
public static class DictionaryBuilder
{
    /// <summary>
    /// Построение двуязычного словаря через поиск ближайших соседей.
    /// Использует CSLS метрику для улучшения качества соответствий.
    /// </summary>
    /// <param name="sourceEmbeddings">Исходные эмбеддинги</param>
    /// <param name="targetEmbeddings">Целевые эмбеддинги</param>
    /// <param name="parameters">Параметры обучения</param>
    /// <returns>Список пар (источник_id, цель_id)</returns>
    public static List<(int sourceId, int targetId)> BuildDictionary(
        MatrixFloat sourceEmbeddings, MatrixFloat targetEmbeddings, Training.Parameters parameters)
    {
        var logger = LoggersSet.Default.UserFriendlyLogger;
        logger.LogDebug($"Построение словаря методом {parameters.DictionaryMethod}...");

        return parameters.DictionaryMethod switch
        {
            "nn" => BuildNearestNeighborDictionary(sourceEmbeddings, targetEmbeddings, parameters),
            "csls_knn_10" => BuildCSLSDictionary(sourceEmbeddings, targetEmbeddings, parameters, k: 10),
            "csls_knn_5" => BuildCSLSDictionary(sourceEmbeddings, targetEmbeddings, parameters, k: 5),
            _ => throw new NotSupportedException($"Неподдерживаемый метод построения словаря: {parameters.DictionaryMethod}")
        };
    }

    /// <summary>
    /// Построение словаря методом ближайших соседей.
    /// Простой и быстрый метод, но менее точный чем CSLS.
    /// </summary>
    private static List<(int, int)> BuildNearestNeighborDictionary(
        MatrixFloat sourceEmbeddings, MatrixFloat targetEmbeddings, Training.Parameters parameters)
    {
        var logger = LoggersSet.Default.UserFriendlyLogger;
        logger.LogDebug("Использование метода ближайших соседей...");

        int sourceVocabSize = Math.Min(sourceEmbeddings.Dimensions[0], parameters.DictionaryMaxVocab);
        int embeddingDim = sourceEmbeddings.Dimensions[1];
        var dictionary = new List<(int, int)>();

        // Нормализация эмбеддингов для косинусного расстояния
        var normalizedSource = sourceEmbeddings.Clone();
        var normalizedTarget = targetEmbeddings.Clone();
        Utils.MathUtils.NormalizeEmbeddings(normalizedSource);
        Utils.MathUtils.NormalizeEmbeddings(normalizedTarget);

        // Поиск ближайшего соседа для каждого исходного слова
        Parallel.For(0, sourceVocabSize, sourceId =>
        {
            var sourceVector = normalizedSource.Data.AsMemory(sourceId * embeddingDim, embeddingDim);
            var nearestNeighbors = Utils.MathUtils.FindKNearestNeighbors(sourceVector, normalizedTarget, 1);

            if (nearestNeighbors.Length > 0)
            {
                lock (dictionary)
                {
                    dictionary.Add((sourceId, nearestNeighbors[0]));
                }
            }
        });

        logger.LogDebug($"Построен словарь с {dictionary.Count} парами (NN метод)");
        return dictionary;
    }

    /// <summary>
    /// Построение словаря методом CSLS (Cross-domain Similarity Local Scaling).
    /// Более точный метод, учитывающий локальную плотность распределения эмбеддингов.
    /// </summary>
    private static List<(int, int)> BuildCSLSDictionary(
        MatrixFloat sourceEmbeddings, MatrixFloat targetEmbeddings, Training.Parameters parameters, int k)
    {
        var logger = LoggersSet.Default.UserFriendlyLogger;
        logger.LogDebug($"Использование CSLS метода с k={k}...");

        int sourceVocabSize = Math.Min(sourceEmbeddings.Dimensions[0], parameters.DictionaryMaxVocab);
        int targetVocabSize = targetEmbeddings.Dimensions[0];
        int embeddingDim = sourceEmbeddings.Dimensions[1];

        // Нормализация эмбеддингов
        var normalizedSource = sourceEmbeddings.Clone();
        var normalizedTarget = targetEmbeddings.Clone();
        Utils.MathUtils.NormalizeEmbeddings(normalizedSource);
        Utils.MathUtils.NormalizeEmbeddings(normalizedTarget);

        // Предвычисление средних косинусных сходств для CSLS
        logger.LogDebug("Предвычисление средних косинусных сходств...");
        var sourceKNNMeans = ComputeKNNMeans(normalizedSource, normalizedTarget, k, sourceVocabSize);
        var targetKNNMeans = ComputeKNNMeans(normalizedTarget, normalizedSource, k, targetVocabSize);

        // Построение словаря с CSLS метрикой
        var dictionary = new List<(int, int)>();

        Parallel.For(0, sourceVocabSize, sourceId =>
        {
            int bestTargetId = -1;
            float bestCSLSScore = float.MinValue;

            var sourceVector = normalizedSource.Data.AsSpan(sourceId * embeddingDim, embeddingDim);

            // Поиск лучшего соответствия по CSLS метрике
            for (int targetId = 0; targetId < targetVocabSize; targetId++)
            {
                var targetVector = normalizedTarget.Data.AsSpan(targetId * embeddingDim, embeddingDim);

                // Косинусное сходство
                float cosineSimilarity = TensorPrimitives.Dot(sourceVector, targetVector);

                // CSLS метрика: 2 * cos(x,y) - r_T(x) - r_S(y)
                float cslsScore = 2 * cosineSimilarity - sourceKNNMeans[sourceId] - targetKNNMeans[targetId];

                if (cslsScore > bestCSLSScore)
                {
                    bestCSLSScore = cslsScore;
                    bestTargetId = targetId;
                }
            }

            if (bestTargetId != -1 && bestCSLSScore > parameters.DictionaryThreshold)
            {
                lock (dictionary)
                {
                    dictionary.Add((sourceId, bestTargetId));
                }
            }
        });

        logger.LogDebug($"Построен словарь с {dictionary.Count} парами (CSLS метод)");
        return dictionary;
    }

    /// <summary>
    /// Вычисление средних косинусных сходств k ближайших соседей для CSLS.
    /// </summary>
    /// <param name="queryEmbeddings">Эмбеддинги запросов</param>
    /// <param name="candidateEmbeddings">Эмбеддинги кандидатов</param>
    /// <param name="k">Количество ближайших соседей</param>
    /// <param name="numQueries">Количество запросов для обработки</param>
    /// <returns>Массив средних сходств для каждого запроса</returns>
    private static float[] ComputeKNNMeans(MatrixFloat queryEmbeddings, MatrixFloat candidateEmbeddings,
                                         int k, int numQueries)
    {
        int embeddingDim = queryEmbeddings.Dimensions[1];
        var knnMeans = new float[numQueries];

        Parallel.For(0, numQueries, queryId =>
        {
            var queryVector = queryEmbeddings.Data.AsMemory(queryId * embeddingDim, embeddingDim);
            var nearestNeighbors = Utils.MathUtils.FindKNearestNeighbors(queryVector, candidateEmbeddings, k);

            // Вычисление среднего косинусного сходства с k ближайшими соседями
            float totalSimilarity = 0.0f;
            foreach (int neighborId in nearestNeighbors)
            {
                var neighborVector = candidateEmbeddings.Data.AsSpan(neighborId * embeddingDim, embeddingDim);
                float similarity = TensorPrimitives.Dot(queryVector.Span, neighborVector);
                totalSimilarity += similarity;
            }

            knnMeans[queryId] = totalSimilarity / nearestNeighbors.Length;
        });

        return knnMeans;
    }

    /// <summary>
    /// Фильтрация словаря по рангу частотности слов.
    /// Оставляет только переводы для наиболее частотных слов.
    /// </summary>
    /// <param name="dictionary">Исходный словарь</param>
    /// <param name="maxRank">Максимальный ранг слова</param>
    /// <returns>Отфильтрованный словарь</returns>
    public static List<(int sourceId, int targetId)> FilterByRank(
        List<(int sourceId, int targetId)> dictionary, int maxRank)
    {
        return dictionary
            .Where(pair => pair.sourceId < maxRank && pair.targetId < maxRank)
            .ToList();
    }

    /// <summary>
    /// Обеспечение взаимности переводов в словаре.
    /// Оставляет только те пары, где перевод является взаимным.
    /// </summary>
    /// <param name="sourceToTarget">Словарь исходный -> целевой</param>
    /// <param name="targetToSource">Словарь целевой -> исходный</param>
    /// <returns>Взаимный словарь</returns>
    public static List<(int sourceId, int targetId)> EnforceMutuality(
        List<(int sourceId, int targetId)> sourceToTarget,
        List<(int sourceId, int targetId)> targetToSource)
    {
        // Создание индекса для быстрого поиска
        var targetToSourceMap = targetToSource.ToDictionary(pair => pair.sourceId, pair => pair.targetId);

        var mutualDictionary = new List<(int, int)>();

        foreach (var (sourceId, targetId) in sourceToTarget)
        {
            // Проверка взаимности: target -> source должен указывать обратно на source
            if (targetToSourceMap.TryGetValue(targetId, out int backTranslation) &&
                backTranslation == sourceId)
            {
                mutualDictionary.Add((sourceId, targetId));
            }
        }

        return mutualDictionary;
    }

    /// <summary>
    /// Оценка качества построенного словаря.
    /// </summary>
    /// <param name="dictionary">Словарь для оценки</param>
    /// <param name="sourceEmbeddings">Исходные эмбеддинги</param>
    /// <param name="targetEmbeddings">Целевые эмбеддинги</param>
    /// <returns>Статистика качества словаря</returns>
    public static DictionaryQualityStats EvaluateDictionaryQuality(
        List<(int sourceId, int targetId)> dictionary,
        MatrixFloat sourceEmbeddings, MatrixFloat targetEmbeddings)
    {
        if (dictionary.Count == 0)
            return new DictionaryQualityStats { Size = 0 };

        int embeddingDim = sourceEmbeddings.Dimensions[1];
        var similarities = new List<float>();

        foreach (var (sourceId, targetId) in dictionary)
        {
            var sourceVector = sourceEmbeddings.Data.AsSpan(sourceId * embeddingDim, embeddingDim);
            var targetVector = targetEmbeddings.Data.AsSpan(targetId * embeddingDim, embeddingDim);

            float similarity = TensorPrimitives.Dot(sourceVector, targetVector);
            similarities.Add(similarity);
        }

        return new DictionaryQualityStats
        {
            Size = dictionary.Count,
            MeanSimilarity = similarities.Average(),
            MedianSimilarity = similarities.OrderBy(x => x).Skip(similarities.Count / 2).First(),
            MinSimilarity = similarities.Min(),
            MaxSimilarity = similarities.Max(),
            StdSimilarity = (float)MathF.Sqrt(similarities.Select(x => MathF.Pow(x - similarities.Average(), 2)).Average())
        };
    }
}

/// <summary>
/// Статистика качества построенного словаря.
/// </summary>
public class DictionaryQualityStats
{
    public int Size { get; set; }
    public float MeanSimilarity { get; set; }
    public float MedianSimilarity { get; set; }
    public float MinSimilarity { get; set; }
    public float MaxSimilarity { get; set; }
    public float StdSimilarity { get; set; }
}