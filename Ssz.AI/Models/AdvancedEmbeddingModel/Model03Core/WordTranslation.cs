using Ssz.AI.Core.Grafana;
using Ssz.Utils.Serialization;
using System.Collections.Generic;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;

public class WordTranslation : IOwnedDataSerializable
{    
    /// <summary>
    ///     Index in Words Array A.    
    /// </summary>
    public int IndexA;

    /// <summary>
    ///     Index in Words Array B.    
    /// </summary>
    public int IndexB;

    public float K;

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.Write(IndexA);
        writer.Write(IndexB);
        writer.Write(K);
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        IndexA = reader.ReadInt32();
        IndexB = reader.ReadInt32();
        K = reader.ReadSingle();
    }
}

public class WordTranslationsCollection : IOwnedDataSerializable
{
    /// <summary>
    ///     Index in Words Array A.    
    /// </summary>
    public List<WordTranslation> WordTranslations = null!;

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.WriteListOfOwnedDataSerializable(WordTranslations, null);
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        WordTranslations = reader.ReadListOfOwnedDataSerializable(() => new WordTranslation(), null);
    }
}