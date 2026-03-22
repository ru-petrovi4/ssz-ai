namespace Ssz.AI.Models.MiniColumnDetailedModel;

// ============================================================
//  АКСОН
//  Биологический аксон пирамидального нейрона коры мозга.
//
//  Морфология соответствует реальной:
//    - Аксон выходит из основания сомы (AIS, axon initial segment)
//    - Идёт вертикально вниз ~100–300 мкм
//    - Затем горизонтально расходится в пределах миниколонки
//    - Точек ветвления в среднем ~4.5, максимум ~17
//    - Суммарная длина ветвей — несколько миллиметров
//    - 10 000 исходящих синапсов, распределённых по ветвям
//
//  Источники: Mohan et al. 2015 (Cerebral Cortex),
//             Bhatt et al. 2009 (activity-dependent branching).
// ============================================================
public sealed class Axon
{
    /// <summary>Индекс аксона в миниколонке (0–199).</summary>
    public readonly int Index;

    /// <summary>
    /// Корневой узел дерева аксона — точка начала (AIS, у сомы).
    /// Всё дерево обходится через Next-ссылки.
    /// </summary>
    public readonly AxonPoint Root;

    /// <summary>
    /// Все 10 000 исходящих синапсов этого аксона.
    /// Координаты синапсов близки к точкам аксонального дерева.
    /// </summary>
    public readonly Synapse[] Synapses;

    /// <summary>Соответствующий бит в векторе активности (0–199).</summary>
    public int BitIndex => Index;

    public Axon(int index, AxonPoint root, Synapse[] synapses)
    {
        Index = index;
        Root = root;
        Synapses = synapses;
    }
}
