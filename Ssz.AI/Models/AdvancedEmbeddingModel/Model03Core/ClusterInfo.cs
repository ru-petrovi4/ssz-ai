using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;

public class ClusterInfo : IOwnedDataSerializable
{
    public float[] CentroidOldVectorNormalized = null!;

    public int HashProjectionIndex;

    public int WordsCount;

    public float[]? CentroidOldVectorNormalized_Mapped;

    /// <summary>
    /// κ_k - параметр концентрации
    /// </summary>
    /// <remarks>device: CPU</remarks>
    public float Concentration;

    /// <summary>
    /// α_k - коэффициент смешивания [K]
    /// </summary>
    /// <remarks>device: CPU</remarks>
    public float MixingCoefficient;
    
    public float AverageWordsNorm;

    public float AverageWordsNormTop10;

    public int Temp_ClusterIndex;

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(4))
        {
            writer.WriteArray(CentroidOldVectorNormalized);
            writer.Write(HashProjectionIndex);
            writer.Write(WordsCount);
            writer.WriteArray(CentroidOldVectorNormalized_Mapped);
            writer.Write(Concentration);
            writer.Write(MixingCoefficient);
            writer.Write(AverageWordsNorm);
            writer.Write(AverageWordsNormTop10);
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
                case 2:
                    CentroidOldVectorNormalized = reader.ReadArray<float>()!;
                    HashProjectionIndex = reader.ReadInt32();
                    WordsCount = reader.ReadInt32();
                    CentroidOldVectorNormalized_Mapped = reader.ReadArray<float>()!;
                    Concentration = reader.ReadSingle();
                    MixingCoefficient = reader.ReadSingle();
                    break;
                case 3:
                    CentroidOldVectorNormalized = reader.ReadArray<float>()!;
                    HashProjectionIndex = reader.ReadInt32();
                    WordsCount = reader.ReadInt32();
                    CentroidOldVectorNormalized_Mapped = reader.ReadArray<float>()!;
                    Concentration = reader.ReadSingle();
                    MixingCoefficient = reader.ReadSingle();
                    AverageWordsNorm = reader.ReadSingle();
                    break;
                case 4:
                    CentroidOldVectorNormalized = reader.ReadArray<float>()!;
                    HashProjectionIndex = reader.ReadInt32();
                    WordsCount = reader.ReadInt32();
                    CentroidOldVectorNormalized_Mapped = reader.ReadArray<float>()!;
                    Concentration = reader.ReadSingle();
                    MixingCoefficient = reader.ReadSingle();
                    AverageWordsNorm = reader.ReadSingle();
                    AverageWordsNormTop10 = reader.ReadSingle();
                    break;
            }
        }
    }
}
