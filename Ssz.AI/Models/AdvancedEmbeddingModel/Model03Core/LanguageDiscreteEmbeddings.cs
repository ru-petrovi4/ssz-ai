using Ssz.AI.Helpers;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;

public class LanguageDiscreteEmbeddings : IOwnedDataSerializable
{
    /// <summary>        
    ///     <para>Ordered Descending by Freq</para>      
    /// </summary>
    public List<Word> Words = null!;

    /// <summary>        
    ///     <para>Ordered Descending by Freq</para>      
    /// </summary>
    public List<Model01Core.ClusterInfo> ClusterInfos = null!;

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

    /// <summary>
    ///     [WordIndex1, WordIndex2] Words energy matrix.
    /// </summary>    
    public MatrixFloat Temp_WordsDistancesOldMatrix = null!;

    public ProjectionOptimization_AlgorithmData Temp_ProjectionOptimization_AlgorithmData = null!;

    public DiscreteVectorsAndMatrices Temp_DiscreteVectorsAndMatrices = null!;

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
                    Words = reader.ReadListOfOwnedDataSerializable(() => new Word(), null);
                    ClusterInfos = reader.ReadListOfOwnedDataSerializable(() => new Model01Core.ClusterInfo(), null);
                    break;
                case 2:
                    Words = reader.ReadListOfOwnedDataSerializable(() => new Word(), null);
                    ClusterInfos = reader.ReadListOfOwnedDataSerializable(() => new Model01Core.ClusterInfo(), null);

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
    /// <summary>
    /// 
    /// </summary>
    /// <param name="embeddings"></param>
    /// <returns></returns>
    public static MatrixFloat GetClustersCosineSimilarityMatrixFloat(this LanguageDiscreteEmbeddings embeddings)
    {
        int dimension = embeddings.ClusterInfos.Count;
        var matrixFloat = new MatrixFloat(dimension, dimension);
        foreach (var i in Enumerable.Range(0, dimension))
        {
            foreach (var j in Enumerable.Range(0, dimension))
            {
                matrixFloat[i, j] = System.Numerics.Tensors.TensorPrimitives.CosineSimilarity(
                    embeddings.ClusterInfos[i].CentroidOldVectorNormalized,
                    embeddings.ClusterInfos[j].CentroidOldVectorNormalized);
            }
        }
        return matrixFloat;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="embeddings"></param>
    /// <returns></returns>
    public static torch.Tensor GetDiscrete_PrimaryBitsOnlyEmbeddingsTensor(this LanguageDiscreteEmbeddings embeddings, int wordsCount, Device device)
    {        
        int dimension = embeddings.Words[0].DiscreteVector.Length;
        float[] data = new float[wordsCount * dimension];
        foreach (var i in Enumerable.Range(0, wordsCount))
        {
            var word = embeddings.Words[i];
            word.DiscreteVector_PrimaryBitsOnly.AsSpan().CopyTo(data.AsSpan(i * dimension, dimension));
        }
        return tensor(data, device: device)
                .reshape(wordsCount, dimension);
    }

    public static int GetClusterIndex(this LanguageDiscreteEmbeddings embeddings, float[] oldVectorNormalized)
    {
        // Вычисляем логарифмические вероятности для численной стабильности
        using var logProbabilities = torch.zeros(size: new long[] { 1, embeddings.ClusterInfos.Count });

        using var oldVectorsTensor = torch.tensor(oldVectorNormalized).reshape([1, oldVectorNormalized.Length]);

        for (int k = 0; k < embeddings.ClusterInfos.Count; k += 1)
        {
            // Вычисляем cosine similarity между данными и k-м центром
            var cosineSimilarities = torch.matmul(oldVectorsTensor, embeddings.MeanDirections[k, ..].t());

            // Вычисляем логарифм нормализующей константы c_d(κ)
            var logNormalizingConstant = VonMisesFisherClusterer.ComputeLogNormalizingConstant(
                embeddings.Concentrations[k].item<float>(),
                oldVectorsTensor.shape[1]
            );

            // Логарифм vMF плотности: log c_d(κ) + κ * μ^T * x
            logProbabilities[.., k] = logNormalizingConstant +
                embeddings.Concentrations[k].item<float>() * cosineSimilarities +
                torch.log(embeddings.MixingCoefficients[k]);
        }

        // Жёсткое назначение: назначаем каждую точку кластеру с максимальной вероятностью
        var assignments = torch.argmax(logProbabilities, dim: 1);
        return (int)assignments[0].item<long>();
    }

    /// <summary>
    /// Obsolete
    /// </summary>
    /// <param name="embeddings"></param>
    /// <returns></returns>
    public static MatrixFloat_ColumnMajor GetOldEmbeddingsMatrix(this LanguageDiscreteEmbeddings embeddings)
    {
        MatrixFloat_ColumnMajor matrixFloat = new MatrixFloat_ColumnMajor(embeddings.Words[0].DiscreteVector.Length, embeddings.Words.Count);
        foreach (var i in Enumerable.Range(0, embeddings.Words.Count))
        {
            var word = embeddings.Words[i];
            word.OldVector.AsSpan().CopyTo(matrixFloat.GetColumn(i));
        }
        return matrixFloat;
    }    

    /// <summary>
    /// Obsolete
    ///     Returns Columns Major matrix.
    /// </summary>
    /// <param name="embeddings"></param>
    /// <returns></returns>
    public static MatrixFloat_ColumnMajor GetDiscreteEmbeddingsMatrix(this LanguageDiscreteEmbeddings embeddings)
    {
        MatrixFloat_ColumnMajor matrixFloat = new MatrixFloat_ColumnMajor(embeddings.Words[0].DiscreteVector.Length, embeddings.Words.Count);
        foreach (var i in Enumerable.Range(0, embeddings.Words.Count))
        {
            var word = embeddings.Words[i];
            word.DiscreteVector.AsSpan().CopyTo(matrixFloat.GetColumn(i));
        }
        return matrixFloat;
    }

    /// <summary>
    /// Obsolete
    ///     Returns Columns Major matrix.
    /// </summary>
    /// <param name="embeddings"></param>
    /// <returns></returns>
    public static MatrixFloat_ColumnMajor GetDiscrete_PrimaryBitsOnlyEmbeddingsMatrix(this LanguageDiscreteEmbeddings embeddings)
    {
        MatrixFloat_ColumnMajor matrixFloat = new MatrixFloat_ColumnMajor(embeddings.Words[0].DiscreteVector.Length, embeddings.Words.Count);
        foreach (var i in Enumerable.Range(0, embeddings.Words.Count))
        {
            var word = embeddings.Words[i];
            word.DiscreteVector_PrimaryBitsOnly.AsSpan().CopyTo(matrixFloat.GetColumn(i));
        }
        return matrixFloat;
    }
}