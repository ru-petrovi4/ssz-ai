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

    public int[] WordsHashProjectionIndices = null!;

    public void GenerateOwnedData(int wordsCount)
    {
        WordsHashProjectionIndices = new int[wordsCount];
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {
            writer.WriteArray(WordsHashProjectionIndices);
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    WordsHashProjectionIndices = reader.ReadArray<int>()!;
                    break;
            }
        }
    }
}
