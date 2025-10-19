using Microsoft.Extensions.Logging;
using Ssz.Utils;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using TorchSharp;
using TorchSharp.Modules; // Для TensorPrimitives
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;

/// <summary>
/// Сопоставление кластеров на основе поворота и их центров float[]
/// </summary>
public class ClustersOneToOneMatcher_MappingLinear : IOwnedDataSerializable
{
    public ClustersOneToOneMatcher_MappingLinear(IUserFriendlyLogger userFriendlyLogger)
    {
        _userFriendlyLogger = userFriendlyLogger;        
    }

    /// <summary>
    /// Индексы как в ClusterInfos[]
    /// </summary>
    public int[] ClustersMapping = null!;

    /// <summary>
    ///     Cosine similarity
    /// </summary>
    /// <param name="source"></param>
    /// <param name="target"></param>
    public void CalculateClustersMapping_V1(LanguageDiscreteEmbeddings source, LanguageDiscreteEmbeddings target)
    {
        ClustersMapping = new int[source.ClusterInfos.Count];

        using Linear mappingLinear = Linear(
            inputSize: source.Words[0].OldVector.Length,
            outputSize: source.Words[0].OldVector.Length,
            hasBias: false);
        using (var _ = no_grad())
        {
            var loadedWeights = load(Path.Combine(@"Data", Model02.FileName_MUSE_Best_Mapping_RU_EN));
            mappingLinear.weight!.copy_(loadedWeights);
        }
        //mappingLinear.to();

        for (int sourceClusterIndex = 0; sourceClusterIndex < source.ClusterInfos.Count; sourceClusterIndex += 1)
        {
            var sourceClusterInfo = source.ClusterInfos[sourceClusterIndex];

            //var norm = TensorPrimitives.Norm(sourcePrimaryWord.OldVectorNormalized); // TEST
            //norm = TensorPrimitives.Norm(sourcePrimaryWord.OldVector); // TEST

            var oldVectorTensor = torch.tensor(sourceClusterInfo.CentroidOldVectorNormalized).reshape([1, sourceClusterInfo.CentroidOldVectorNormalized.Length]);
            var mappedOldVectorTensor = mappingLinear.forward(oldVectorTensor);
            float[] mappedOldVectorNormalized = mappedOldVectorTensor.data<float>().ToArray();
            float norm = TensorPrimitives.Norm(mappedOldVectorNormalized);            
            TensorPrimitives.Divide(mappedOldVectorNormalized, norm, mappedOldVectorNormalized);
            
            float max = float.MinValue;
            int selected = -1;
            for (int targetClusterIndex = 0; targetClusterIndex < target.ClusterInfos.Count; targetClusterIndex += 1)
            {
                var targetClusterInfo = target.ClusterInfos[targetClusterIndex];

                float cosineSimilarity = TensorPrimitives.Dot(mappedOldVectorNormalized, targetClusterInfo.CentroidOldVectorNormalized);
                if (cosineSimilarity > max)
                {
                    max = cosineSimilarity;
                    selected = targetClusterIndex;
                }
            }
            if (selected != -1)
            {
                ClustersMapping[sourceClusterIndex] = selected;                
            }
            else
            {
            }
        }
    }

    /// <summary>
    ///     Clusters distribution
    /// </summary>
    /// <param name="source"></param>
    /// <param name="target"></param>
    public void CalculateClustersMapping_ClustersDistribution(LanguageDiscreteEmbeddings source, LanguageDiscreteEmbeddings target)
    {
        ClustersMapping = new int[source.ClusterInfos.Count];

        using Linear mappingLinear = Linear(
            inputSize: source.Words[0].OldVector.Length,
            outputSize: source.Words[0].OldVector.Length,
            hasBias: false);
        using (var _ = no_grad())
        {
            var loadedWeights = load(Path.Combine(@"Data", Model02.FileName_MUSE_Best_Mapping_RU_EN));
            mappingLinear.weight!.copy_(loadedWeights);
        }
        //mappingLinear.to();

        for (int sourceClusterIndex = 0; sourceClusterIndex < source.ClusterInfos.Count; sourceClusterIndex += 1)
        {
            var sourceClusterInfo = source.ClusterInfos[sourceClusterIndex];

            //var norm = TensorPrimitives.Norm(sourcePrimaryWord.OldVectorNormalized); // TEST
            //norm = TensorPrimitives.Norm(sourcePrimaryWord.OldVector); // TEST

            var oldVectorTensor = torch.tensor(sourceClusterInfo.CentroidOldVectorNormalized).reshape([1, sourceClusterInfo.CentroidOldVectorNormalized.Length]);
            var mappedOldVectorTensor = mappingLinear.forward(oldVectorTensor);
            float[] mappedOldVectorNormalized = mappedOldVectorTensor.data<float>().ToArray();
            float norm = TensorPrimitives.Norm(mappedOldVectorNormalized);
            TensorPrimitives.Divide(mappedOldVectorNormalized, norm, mappedOldVectorNormalized);

            ClustersMapping[sourceClusterIndex] = target.GetClusterIndex(mappedOldVectorNormalized);
        }
    }

    /// <summary>
    /// Energy matrix.
    /// </summary>
    /// <param name="source"></param>
    /// <param name="target"></param>
    public void CalculateClustersMapping_EnergyMatrix(LanguageDiscreteEmbeddings source, LanguageDiscreteEmbeddings target)
    {
        ClustersMapping = new int[source.ClusterInfos.Count];

        using Linear mappingLinear = Linear(
            inputSize: source.Words[0].OldVector.Length,
            outputSize: source.Words[0].OldVector.Length,
            hasBias: false);
        using (var _ = no_grad())
        {
            var loadedWeights = load(Path.Combine(@"Data", Model02.FileName_MUSE_Best_Mapping_RU_EN));
            mappingLinear.weight!.copy_(loadedWeights);
        }
        //mappingLinear.to();

        for (int sourceClusterIndex = 0; sourceClusterIndex < source.ClusterInfos.Count; sourceClusterIndex += 1)
        {
            var sourceClusterInfo = source.ClusterInfos[sourceClusterIndex];

            //var norm = TensorPrimitives.Norm(sourcePrimaryWord.OldVectorNormalized); // TEST
            //norm = TensorPrimitives.Norm(sourcePrimaryWord.OldVector); // TEST

            var oldVectorTensor = torch.tensor(sourceClusterInfo.CentroidOldVectorNormalized).reshape([1, sourceClusterInfo.CentroidOldVectorNormalized.Length]);
            var mappedOldVectorTensor = mappingLinear.forward(oldVectorTensor);
            float[] mappedOldVectorNormalized = mappedOldVectorTensor.data<float>().ToArray();
            float norm = TensorPrimitives.Norm(mappedOldVectorNormalized);
            TensorPrimitives.Divide(mappedOldVectorNormalized, norm, mappedOldVectorNormalized);
            sourceClusterInfo.Temp_CentroidOldVectorNormalized_Mapped = mappedOldVectorNormalized;

            // Ищем позицию B с максимальным весом среди неиспользованных
            float minEnergy = float.MaxValue;
            int selected = -1;
            for (int targetClusterIndex = 0; targetClusterIndex < target.ClusterInfos.Count; targetClusterIndex += 1)
            {
                var targetClusterInfo = target.ClusterInfos[targetClusterIndex];

                float energy = ModelHelper.GetEnergy(sourceClusterInfo.Temp_CentroidOldVectorNormalized_Mapped, targetClusterInfo.CentroidOldVectorNormalized);
                if (energy < minEnergy)
                {
                    minEnergy = energy;
                    selected = targetClusterIndex;
                }
            }
            if (selected != -1)
            {
                ClustersMapping[sourceClusterIndex] = selected;
            }
        }
        //var hsA = ClustersMapping.ToHashSet();

        //var clustersMapping_Reverse = new int[source.ClusterInfos.Count];
        //for (int targetClusterIndex = 0; targetClusterIndex < target.ClusterInfos.Count; targetClusterIndex += 1)
        //{
        //    var targetClusterInfo = target.ClusterInfos[targetClusterIndex];              

        //    // Ищем позицию B с максимальным весом среди неиспользованных
        //    float minEnergy = float.MaxValue;
        //    int selected = -1;
        //    for (int sourceClusterIndex = 0; sourceClusterIndex < source.ClusterInfos.Count; sourceClusterIndex += 1)
        //    {
        //        var sourceClusterInfo = source.ClusterInfos[sourceClusterIndex];

        //        float energy = ModelHelper.GetEnergy(sourceClusterInfo.Temp_CentroidOldVectorNormalized_Mapped, targetClusterInfo.CentroidOldVectorNormalized);
        //        if (energy < minEnergy)
        //        {
        //            minEnergy = energy;
        //            selected = sourceClusterIndex;
        //        }
        //    }
        //    if (selected != -1)
        //    {
        //        clustersMapping_Reverse[targetClusterIndex] = selected;
        //    }
        //}
        //var hsB = clustersMapping_Reverse.ToHashSet();
        //_userFriendlyLogger.LogInformation($"Количество уникальных сопоставлений: {hsA.Count}, {hsB.Count}");
    }

    /// <summary>
    /// Energy matrix. Hungarian algorithm
    /// </summary>
    /// <param name="source"></param>
    /// <param name="target"></param>
    public void CalculateClustersMapping_EnergyMatrixHungarian(LanguageDiscreteEmbeddings source, LanguageDiscreteEmbeddings target)
    {
        using Linear mappingLinear = Linear(
            inputSize: source.Words[0].OldVector.Length,
            outputSize: source.Words[0].OldVector.Length,
            hasBias: false);
        using (var _ = no_grad())
        {
            var loadedWeights = load(Path.Combine(@"Data", Model02.FileName_MUSE_Best_Mapping_RU_EN));
            mappingLinear.weight!.copy_(loadedWeights);
        }
        //mappingLinear.to();

        long[,] costMatrix = new long[source.ClusterInfos.Count, target.ClusterInfos.Count];
        for (int sourceClusterIndex = 0; sourceClusterIndex < source.ClusterInfos.Count; sourceClusterIndex += 1)
        {
            var sourceClusterInfo = source.ClusterInfos[sourceClusterIndex];            

            var oldVectorTensor = torch.tensor(sourceClusterInfo.CentroidOldVectorNormalized).reshape([1, sourceClusterInfo.CentroidOldVectorNormalized.Length]);
            var mappedOldVectorTensor = mappingLinear.forward(oldVectorTensor);
            float[] mappedOldVectorNormalized = mappedOldVectorTensor.data<float>().ToArray();
            float norm = TensorPrimitives.Norm(mappedOldVectorNormalized);
            TensorPrimitives.Divide(mappedOldVectorNormalized, norm, mappedOldVectorNormalized);
                        
            for (int targetClusterIndex = 0; targetClusterIndex < target.ClusterInfos.Count; targetClusterIndex += 1)
            {
                var targetClusterInfo = target.ClusterInfos[targetClusterIndex];

                float energy = ModelHelper.GetEnergy(mappedOldVectorNormalized, targetClusterInfo.CentroidOldVectorNormalized);

                costMatrix[sourceClusterIndex, targetClusterIndex] = (long)(energy * 10000.0f);
            }            
        }

        ClustersMapping = HungarianAlgorithm.FindAssignments(costMatrix);
    }    

    public void ComputeDetailedEvaluationReport(LanguageDiscreteEmbeddings languageDiscreteEmbeddings, ILogger logger)
    {
        var count = languageDiscreteEmbeddings.Words.Count;
        var dim = languageDiscreteEmbeddings.Words[0].OldVectorNormalized.Length;

        float[] allData = new float[count * dim];
        long[] allLabels = new long[count];

        for (int wordIndex = 0; wordIndex < languageDiscreteEmbeddings.Words.Count; wordIndex += 1)
        {
            var word = languageDiscreteEmbeddings.Words[wordIndex];
            Array.Copy(word.OldVectorNormalized, 0, allData, wordIndex * dim, dim);          
            allLabels[wordIndex] = word.ClusterIndex;
        }

        var data = torch.tensor(allData, dimensions: [count, dim]);
        var labels = torch.tensor(allLabels);

        SphericalClusteringMetrics.ComputeDetailedEvaluationReport(data, labels, logger);
    }         

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.WriteArray(ClustersMapping);
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        ClustersMapping = reader.ReadArray<int>()!;
    }    

    #region private fields

    private IUserFriendlyLogger _userFriendlyLogger;    

    #endregion
}


//public void ShowWords(LanguageDiscreteEmbeddings source, LanguageDiscreteEmbeddings target)
//    {
//        for (int sourceClusterIndex = 0; sourceClusterIndex < source.ClusterInfos.Count; sourceClusterIndex += 1)
//        {
//            ShowWords(source, sourceClusterIndex);
//            ShowWords(target, ClustersMapping[sourceClusterIndex]);
//            _userFriendlyLogger.LogInformation($"------------------------");
//        }
//   }   


//private void ShowWords(LanguageDiscreteEmbeddings source, int clusterIndex)
//    {
//        _userFriendlyLogger.LogInformation($"Кластер: {clusterIndex}");

//        var clusterInfo = source.ClusterInfos[clusterIndex];

//        foreach (var word in source.Words
//            .Where(w => w.ClusterIndex == clusterIndex)
//            .OrderByDescending(w => TensorPrimitives.Dot(w.OldVectorNormalized, clusterInfo.CentroidOldVectorNormalized))
//            .Take(10))
//        {
//            _userFriendlyLogger.LogInformation(word.Name);
//        }

//        _userFriendlyLogger.LogInformation($"------------------------");
//    }