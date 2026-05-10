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
public sealed class PyramidalAxon : IAxon
{
    public PyramidalAxon(AxonPoint root, Synapse[] synapses)
    {        
        Root = root;
        Synapses = synapses;
    }

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

    public bool Temp_IsActive;

    AxonPoint IAxon.Root => Root;

    Synapse[] IAxon.Synapses => Synapses;

    bool IAxon.IsActive => Temp_IsActive;        
}
