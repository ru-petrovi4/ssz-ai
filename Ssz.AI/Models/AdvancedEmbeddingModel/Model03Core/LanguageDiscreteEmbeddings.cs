using Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;

public class LanguageDiscreteEmbeddings : IOwnedDataSerializable
{
    /// <summary>        
    ///     <para>Ordered Descending by Freq</para>      
    /// </summary>
    public List<WordWithDiscreteEmbedding> Words = null!;

    /// <summary>        
    ///     <para>Ordered Descending by Freq</para>      
    /// </summary>
    public List<WordWithDiscreteEmbedding> PrimaryWords = null!;

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {
            writer.WriteListOfOwnedDataSerializable(Words, null);
            writer.WriteList(PrimaryWords.Select(w => w.Index).ToList());            
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {   
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    Words = reader.ReadListOfOwnedDataSerializable(() => new WordWithDiscreteEmbedding(), null);

                    var primaryWordIndices = reader.ReadList<int>()!;
                    PrimaryWords = primaryWordIndices.Select(i => Words[i]).ToList();
                    break;
            }
        }
    }
}

public static class LanguageDiscreteEmbeddingsExtensions
{
    public static MatrixFloat GetOldEmbeddingsMatrix(this LanguageDiscreteEmbeddings embeddings)
    {
        MatrixFloat matrixFloat = new MatrixFloat(embeddings.Words[0].DiscreteVector.Length, embeddings.Words.Count);
        foreach (var i in Enumerable.Range(0, embeddings.Words.Count))
        {
            var word = embeddings.Words[i];
            word.OldVector.AsSpan().CopyTo(matrixFloat.GetColumn(i));
        }
        return matrixFloat;
    }

    /// <summary>
    ///     Returns Columns Major matrix.
    /// </summary>
    /// <param name="embeddings"></param>
    /// <returns></returns>
    public static MatrixFloat GetDiscreteEmbeddingsMatrix(this LanguageDiscreteEmbeddings embeddings)
    {
        MatrixFloat matrixFloat = new MatrixFloat(embeddings.Words[0].DiscreteVector.Length, embeddings.Words.Count);
        foreach (var i in Enumerable.Range(0, embeddings.Words.Count))
        {
            var word = embeddings.Words[i];
            word.DiscreteVector.AsSpan().CopyTo(matrixFloat.GetColumn(i));
        }
        return matrixFloat;
    }

    /// <summary>
    ///     Returns Columns Major matrix.
    /// </summary>
    /// <param name="embeddings"></param>
    /// <returns></returns>
    public static MatrixFloat GetDiscrete_PrimaryBitsOnlyEmbeddingsMatrix(this LanguageDiscreteEmbeddings embeddings)
    {
        MatrixFloat matrixFloat = new MatrixFloat(embeddings.Words[0].DiscreteVector.Length, embeddings.Words.Count);
        foreach (var i in Enumerable.Range(0, embeddings.Words.Count))
        {
            var word = embeddings.Words[i];
            word.DiscreteVector_PrimaryBitsOnly.AsSpan().CopyTo(matrixFloat.GetColumn(i));
        }
        return matrixFloat;
    }
}