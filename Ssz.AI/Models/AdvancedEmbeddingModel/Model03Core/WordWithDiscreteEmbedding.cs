using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;

public class WordWithDiscreteEmbedding : IOwnedDataSerializable
{
    public string Name = null!;

    /// <summary>
    ///     Index in Words Array.    
    /// </summary>
    public int Index;

    public float[] OldVector = null!;

    public float[] DiscreteVector = null!;

    /// <summary>
    ///     New vectors for each word with primary bits only.
    /// </summary>
    public float[] DiscreteVector_PrimaryBitsOnly = null!;

    /// <summary>
    ///     New vectors for each  word with secondary bits only.
    /// </summary>
    public float[] DiscreteVector_SecondaryBitsOnly = null!;

    public void SerializeOwnedData(SerializationWriter serializationWriter, object? context)
    {
        serializationWriter.Write(Name);
        serializationWriter.WriteOptimized(Index);
        serializationWriter.WriteArray(OldVector);
        serializationWriter.WriteArray(DiscreteVector);
        serializationWriter.WriteArray(DiscreteVector_PrimaryBitsOnly);
        serializationWriter.WriteArray(DiscreteVector_SecondaryBitsOnly);
    }

    public void DeserializeOwnedData(SerializationReader serializationReader, object? context)
    {
        Name = serializationReader.ReadString();
        Index = serializationReader.ReadOptimizedInt32();
        OldVector = serializationReader.ReadArray<float>()!;
        DiscreteVector = serializationReader.ReadArray<float>()!;
        DiscreteVector_PrimaryBitsOnly = serializationReader.ReadArray<float>()!;
        DiscreteVector_SecondaryBitsOnly = serializationReader.ReadArray<float>()!;
    }
}
