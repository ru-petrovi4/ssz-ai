using Ssz.AI.Helpers;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;

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
    public List<ClusterInfo> ClusterInfos = null!;

    // Параметры модели - изучаются в процессе обучения
    /// <summary>
    /// μ_k - направления средних для каждого кластера [K x D]
    /// </summary>
    /// <remarks>device: CPU</remarks>
    public Tensor MeanDirections = null!;

    /// <summary>
    /// κ_k - параметры концентрации для каждого кластера [K]
    /// </summary>
    /// <remarks>device: CPU</remarks>
    public Tensor Concentrations = null!;

    /// <summary>
    /// α_k - коэффициенты смешивания [K]
    /// </summary>
    /// <remarks>device: CPU</remarks>
    public Tensor MixingCoefficients = null!;

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(2))
        {
            writer.WriteListOfOwnedDataSerializable(Words, null);
            writer.WriteListOfOwnedDataSerializable(ClusterInfos, null);

            TorchSharpHelper.WriteTensor(MeanDirections, writer);
            TorchSharpHelper.WriteTensor(Concentrations, writer);
            TorchSharpHelper.WriteTensor(MixingCoefficients, writer);
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
                    ClusterInfos = reader.ReadListOfOwnedDataSerializable(() => new ClusterInfo(), null);
                    break;
                case 2:
                    Words = reader.ReadListOfOwnedDataSerializable(() => new WordWithDiscreteEmbedding(), null);
                    ClusterInfos = reader.ReadListOfOwnedDataSerializable(() => new ClusterInfo(), null);

                    MeanDirections = TorchSharpHelper.ReadTensor(reader);
                    Concentrations = TorchSharpHelper.ReadTensor(reader);
                    MixingCoefficients = TorchSharpHelper.ReadTensor(reader);
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