namespace Ssz.AI.Models.MiniColumnDetailedModel;

public interface IAxon
{
    /// <summary>
    /// Корневой узел дерева аксона — точка начала (AIS, у сомы).
    /// Всё дерево обходится через Next-ссылки.
    /// </summary>
    public AxonPoint Root { get; }

    /// <summary>
    /// Все 10 000 исходящих синапсов этого аксона.
    /// Координаты синапсов близки к точкам аксонального дерева.
    /// </summary>
    public Synapse[] Synapses { get; }

    public bool IsActive { get; }
}
