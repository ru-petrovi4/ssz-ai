using Microsoft.Extensions.Logging;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Utils;
using Ssz.Utils.Logging;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

/// <summary>
/// Основной класс сравнения эмбеддингов по задаче word similarity (Spearman correlation)
/// Сравнение с человеческой оценкой
/// </summary>
public static class EmbeddingEvaluator
{
    /// <summary>
    /// Считает косинусное сходство для двух векторов
    /// </summary>
    public static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float dot = TensorPrimitives.Dot(a, b);
        float normA = MathF.Sqrt(TensorPrimitives.SumOfSquares(a));
        float normB = MathF.Sqrt(TensorPrimitives.SumOfSquares(b));
        return dot / (normA * normB + 1e-8f);
    }

    /// <summary>
    /// Считает корреляцию Спирмена между двумя списками (оценивает "похожесть порядка")
    /// </summary>
    public static double SpearmanCorrelation(List<float> x, List<float> y)
    {
        if (x.Count != y.Count) throw new ArgumentException("Вектора должны быть одной длины.");

        var n = x.Count;
        var rx = Ranks(x);
        var ry = Ranks(y);
        double sumDiff2 = 0.0;

        for (int i = 0; i < n; i++)
            sumDiff2 += Math.Pow(rx[i] - ry[i], 2);

        return 1.0 - (6.0 * sumDiff2) / (n * (n * n - 1));
    }

    /// <summary>
    /// Преобразует список в ранги
    /// </summary>
    public static double[] Ranks(List<float> values)
    {
        var indexed = values.Select((v, i) => (Value: v, Index: i)).OrderBy(pair => pair.Value).ToList();
        var ranks = new double[values.Count];
        for (int i = 0; i < indexed.Count; i++)
            ranks[indexed[i].Index] = i + 1;
        return ranks;
    }

    /// <summary>
    /// Сравнивает качество моделей на датасете word similarity
    /// </summary>
    /// <param name="pairsFile">Файл: "слово1 слово2 human_score"</param>
    /// <param name="emb1">Эмбеддинги 1</param>
    /// <param name="emb2">Эмбеддинги 2</param>
    /// <param name="vecSize1">Размерность 1</param>
    /// <param name="vecSize2">Размерность 2</param>
    public static void CompareOnSimilarityDataset(
        string pairsFile, 
        Dictionary<string, float[]> emb1, 
        Dictionary<string, float[]> emb2, 
        int vecSize1, 
        int vecSize2,
        ILoggersSet loggersSet)
    {
        var humanScores = new List<float>();
        var model1Scores = new List<float>();
        var model2Scores = new List<float>();

        foreach (var line in File.ReadLines(pairsFile))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var parts = line.Split(new[] { ',', ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != 3) continue;
            var w1 = parts[0];
            var w2 = parts[1];

            // score можа быть дробным
            if (!float.TryParse(parts[2], NumberStyles.Any, CultureInfo.InvariantCulture, out float humanScore))
                continue;

            if (!emb1.TryGetValue(w1, out var v1a) || !emb1.TryGetValue(w2, out var v1b)) continue;
            if (!emb2.TryGetValue(w1, out var v2a) || !emb2.TryGetValue(w2, out var v2b)) continue;

            // Сравниваем только пары, у которых есть оба эмбединга в обеих моделях
            model1Scores.Add(CosineSimilarity(v1a, v1b));
            model2Scores.Add(CosineSimilarity(v2a, v2b));
            humanScores.Add(humanScore);
        }

        var corr1 = SpearmanCorrelation(humanScores, model1Scores);
        var corr2 = SpearmanCorrelation(humanScores, model2Scores);

        loggersSet.UserFriendlyLogger.LogInformation($"Spearman correlation (OldVector): {corr1:F4}");
        loggersSet.UserFriendlyLogger.LogInformation($"Spearman correlation (DiscreteVector): {corr2:F4}");
    }
}

/// <summary>
/// Сравнивает структуры двух эмбеддингов по совпадению top-N соседей
/// </summary>
public static class NeighborStructureComparer
{
    /// <summary>
    /// Находит топ-N ближайших соседей для каждого слова, считает процент совпадений двух моделей
    /// </summary>
    /// <param name="labels">Общие слова</param>
    /// <param name="embA">Эмбеддинги A (MatrixFloat)</param>
    /// <param name="embB">Эмбеддинги B (MatrixFloat)</param>
    /// <param name="topN">Сколько брать ближайших соседей</param>
    public static double Compare(string[] labels, MatrixFloat embA, MatrixFloat embB, int topN = 10)
    {
        int count = labels.Length;
        int totalMatches = 0;
        int totalChecked = 0;

        for (int i = 0; i < count; i++)
        {
            // Вектор целевого слова
            var aVec = embA.GetColumn(i);
            var bVec = embB.GetColumn(i);

            List<(int idx, float dist)> distA = new();
            List<(int idx, float dist)> distB = new();

            for (int j = 0; j < count; j++)
            {
                if (i == j) continue;

                distA.Add((j, CosineDistance(aVec, embA.GetColumn(j))));
                distB.Add((j, CosineDistance(bVec, embB.GetColumn(j))));
            }
            // Сортировка и выбор топ-N ближайших
            var topA = distA.OrderBy(x => x.dist).Take(topN).Select(x => x.idx).ToHashSet();
            var topB = distB.OrderBy(x => x.dist).Take(topN).Select(x => x.idx).ToHashSet();

            int intersection = topA.Intersect(topB).Count();
            totalMatches += intersection;
            totalChecked += topN;
        }
        return 100.0 * totalMatches / totalChecked;
    }

    public static float CosineDistance(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float dot = TensorPrimitives.Dot(a, b);
        float normA = MathF.Sqrt(TensorPrimitives.SumOfSquares(a));
        float normB = MathF.Sqrt(TensorPrimitives.SumOfSquares(b));
        return 1.0f - (dot / (normA * normB + 1e-8f));
    }
}
