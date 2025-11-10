using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel2;

public class Word : IOwnedDataSerializable
{
    public Word()
    {
        DiscreteRandomVector = new float[Model01.Constants.DiscreteVectorLength];
        DiscreteOptimizedVector = new float[Model01.Constants.DiscreteVectorLength];
        DiscreteOptimizedVector_PrimaryBitsOnly = new float[Model01.Constants.DiscreteVectorLength];
        DiscreteOptimizedVector_SecondaryBitsOnly = new float[Model01.Constants.DiscreteVectorLength];
    }

    public string Name = null!;

    public float CorpusFreq;

    /// <summary>
    ///     Original discrete unoptimized vector.
    /// </summary>
    public float[] DiscreteRandomVector = null!;    

    public float[] DiscreteOptimizedVector = null!;

    /// <summary>
    ///     New vectors for each word with primary bits only.
    /// </summary>
    public float[] DiscreteOptimizedVector_PrimaryBitsOnly = null!;

    /// <summary>
    ///     New vectors for each  word with secondary bits only.
    /// </summary>
    public float[] DiscreteOptimizedVector_SecondaryBitsOnly = null!;

    public int Temp_InCorpusCount;    

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.Write(Name);
        writer.Write(CorpusFreq);
        writer.WriteArray(DiscreteRandomVector);        
        writer.WriteArray(DiscreteOptimizedVector);
        writer.WriteArray(DiscreteOptimizedVector_PrimaryBitsOnly);
        writer.WriteArray(DiscreteOptimizedVector_SecondaryBitsOnly);
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        Name = reader.ReadString();
        CorpusFreq = reader.ReadSingle();
        DiscreteRandomVector = reader.ReadArray<float>()!;
        DiscreteOptimizedVector = reader.ReadArray<float>()!;
        DiscreteOptimizedVector_PrimaryBitsOnly = reader.ReadArray<float>()!;
        DiscreteOptimizedVector_SecondaryBitsOnly = reader.ReadArray<float>()!;
    }
}
