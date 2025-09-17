using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;

public class ProjectionOptimization_AlgorithmData : ISerializableModelObject
{
    #region construction and destruction

    public ProjectionOptimization_AlgorithmData(string name)
    {
        Name = name;
    }

    #endregion

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
