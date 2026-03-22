using System.Collections.Generic;
using System.Numerics;

namespace Ssz.AI.Models.MiniColumnDetailedModel;

// ============================================================
//  РЕЗУЛЬТАТ ПОИСКА АКТИВНЫХ ЗОН
//  Содержит центр сферической зоны и список уникальных активных
//  аксонов, синапсы которых попали в эту зону.
// ============================================================
public sealed class ActiveZone
{
    /// <summary>Центр найденной зоны (мкм).</summary>
    public Vector3 Center;

    /// <summary>
    /// Множество индексов РАЗНЫХ активных аксонов,
    /// синапсы которых попали в радиус зоны.
    /// </summary>
    public readonly HashSet<int> ActiveAxonIndices = new();

    /// <summary>Число уникальных активных аксонов в зоне.</summary>
    public int UniqueAxonCount => ActiveAxonIndices.Count;
}
