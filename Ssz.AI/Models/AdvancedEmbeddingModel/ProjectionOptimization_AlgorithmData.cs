using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public class ProjectionOptimization_AlgorithmData : ISerializableModelObject
{
    public string Name = null!;

    public int[] WordsProjectionIndices = null!;

    public void GenerateOwnedData(int wordsCount)
    {
        WordsProjectionIndices = new int[wordsCount];
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {
            writer.WriteArray(WordsProjectionIndices);
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    WordsProjectionIndices = reader.ReadArray<int>()!;
                    break;
            }
        }
    }
}
