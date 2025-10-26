using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;

public class ClusterInfo : IOwnedDataSerializable
{
    public float[] CentroidOldVectorNormalized = null!;

    public int HashProjectionIndex;

    public int WordsCount;

    public float[]? CentroidOldVectorNormalized_Mapped;

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {
            writer.WriteArray(CentroidOldVectorNormalized);
            writer.Write(HashProjectionIndex);
            writer.Write(WordsCount);
            writer.WriteArray(CentroidOldVectorNormalized_Mapped);
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    CentroidOldVectorNormalized = reader.ReadArray<float>()!;
                    HashProjectionIndex = reader.ReadInt32();
                    WordsCount = reader.ReadInt32();
                    CentroidOldVectorNormalized_Mapped = reader.ReadArray<float>()!;
                    break;
            }
        }
    }
}
