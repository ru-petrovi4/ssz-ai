using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;

public class ClusterInfo : IOwnedDataSerializable
{
    public float[] CentroidOldVectorNormalized = null!;

    public int HashProjectionIndex;

    public int WordsCount;

    public float[] Temp_CentroidOldVectorNormalized_Mapped = null!;

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.WriteArray(CentroidOldVectorNormalized);
        writer.Write(HashProjectionIndex);
        writer.Write(WordsCount);
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        CentroidOldVectorNormalized = reader.ReadArray<float>()!;
        HashProjectionIndex = reader.ReadInt32();
        WordsCount = reader.ReadInt32();
    }
}
