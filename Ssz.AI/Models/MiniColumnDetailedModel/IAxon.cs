namespace Ssz.AI.Models.MiniColumnDetailedModel;

public interface IAxon
{
    /// <summary>
    /// Корневой узел дерева аксона — точка начала (AIS, у сомы).
    /// Всё дерево обходится через Next-ссылки.
    /// </summary>
    public AxonPoint Root { get; }

    /// <summary>
    /// Все исходящие синапсы этого аксона.    
    /// </summary>
    public Synapse[] Synapses { get; }

    public bool IsActive { get; }
}
