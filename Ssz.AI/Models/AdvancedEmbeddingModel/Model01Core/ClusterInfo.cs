using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;

public class ClusterInfo : IOwnedDataSerializable
{
    public float[] CentroidOldVectorNormalized = null!;

    public int HashProjection;

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.WriteArray(CentroidOldVectorNormalized);
        writer.Write(HashProjection);
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        CentroidOldVectorNormalized = reader.ReadArray<float>()!;
        HashProjection = reader.ReadInt32();
    }
}
