using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;

public class WordWithDiscreteEmbedding : IOwnedDataSerializable
{
    public string Name = null!;

    /// <summary>
    ///     Index in Words Array.    
    /// </summary>
    public int Index;

    /// <summary>
    ///     Cluster Index.    
    /// </summary>
    public int ClusterIndex;

    /// <summary>
    ///     Original vector.
    /// </summary>
    public float[] OldVector = null!;

    /// <summary>
    ///     Original normalized vector (module 1).
    /// </summary>
    public float[] OldVectorNormalized = null!;

    public float[] DiscreteVector = null!;

    /// <summary>
    ///     New vectors for each word with primary bits only.
    /// </summary>
    public float[] DiscreteVector_PrimaryBitsOnly = null!;

    /// <summary>
    ///     New vectors for each  word with secondary bits only.
    /// </summary>
    public float[] DiscreteVector_SecondaryBitsOnly = null!;

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.Write(Name);
        writer.WriteOptimized(Index);
        writer.WriteOptimized(ClusterIndex);
        writer.WriteArray(OldVector);
        writer.WriteArray(OldVectorNormalized);
        writer.WriteArray(DiscreteVector);
        writer.WriteArray(DiscreteVector_PrimaryBitsOnly);
        writer.WriteArray(DiscreteVector_SecondaryBitsOnly);
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        Name = reader.ReadString();
        Index = reader.ReadOptimizedInt32();
        ClusterIndex = reader.ReadOptimizedInt32();
        OldVector = reader.ReadArray<float>()!;
        OldVectorNormalized = reader.ReadArray<float>()!;
        DiscreteVector = reader.ReadArray<float>()!;
        DiscreteVector_PrimaryBitsOnly = reader.ReadArray<float>()!;
        DiscreteVector_SecondaryBitsOnly = reader.ReadArray<float>()!;
    }
}
