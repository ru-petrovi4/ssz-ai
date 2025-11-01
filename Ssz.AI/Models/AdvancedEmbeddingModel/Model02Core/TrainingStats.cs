using Ssz.Utils;
using System;
using System.Collections.Generic;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core;

/// <summary>
/// Статистики обучения
/// </summary>
public sealed record TrainingStats
{
    /// <summary>
    /// Потери дискриминатора
    /// </summary>
    public List<float> DiscriminatorLosses { get; init; } = new();

    /// <summary>
    /// Потери маппинга
    /// </summary>
    public List<float> MappingLosses { get; init; } = new();

    /// <summary>
    /// Точность дискриминатора
    /// </summary>
    public float DiscriminatorAccuracy { get; set; }

    /// <summary>
    /// Количество обработанных слов
    /// </summary>
    public long ProcessedWords { get; set; }

    /// <summary>
    /// Время обучения
    /// </summary>
    public TimeSpan ElapsedTime { get; set; }

    public CaseInsensitiveOrderedDictionary<float> ToLog { get; init; } = new();
}
