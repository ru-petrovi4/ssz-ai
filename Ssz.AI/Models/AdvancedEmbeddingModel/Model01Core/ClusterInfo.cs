using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;

public class ClusterInfo : IOwnedDataSerializable
{
    public float[] CentroidOldVectorNormalized = null!;

    public int HashProjectionIndex;

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.WriteArray(CentroidOldVectorNormalized);
        writer.Write(HashProjectionIndex);
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        CentroidOldVectorNormalized = reader.ReadArray<float>()!;
        HashProjectionIndex = reader.ReadInt32();
    }
}
