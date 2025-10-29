using Microsoft.Extensions.Logging;
using Ssz.Utils;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
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
    public void OptimizeClusters(LanguageDiscreteEmbeddings source, LanguageDiscreteEmbeddings target)
    {
        var clustersMapping = new int[source.ClusterInfos.Count];
        Array.Fill(clustersMapping, -1);

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
            if (sourceClusterInfo is null)
                continue;

            //var norm = TensorPrimitives.Norm(sourcePrimaryWord.OldVectorNormalized); // TEST
            //norm = TensorPrimitives.Norm(sourcePrimaryWord.OldVector); // TEST

            var oldVectorTensor = torch.tensor(sourceClusterInfo.CentroidOldVectorNormalized).reshape([1, sourceClusterInfo.CentroidOldVectorNormalized.Length]);
            var mappedOldVectorTensor = mappingLinear.forward(oldVectorTensor);
            float[] mappedOldVectorNormalized = mappedOldVectorTensor.data<float>().ToArray();
            float norm = TensorPrimitives.Norm(mappedOldVectorNormalized);
            TensorPrimitives.Divide(mappedOldVectorNormalized, norm, mappedOldVectorNormalized);
            sourceClusterInfo.CentroidOldVectorNormalized_Mapped = mappedOldVectorNormalized;

            sourceClusterInfo.Concentration = source.Concentrations[sourceClusterIndex].item<float>();
            sourceClusterInfo.MixingCoefficient = source.MixingCoefficients[sourceClusterIndex].item<float>();

            // Ищем позицию B с максимальным весом среди неиспользованных
            float minEnergy = float.MaxValue;
            int selected = -1;
            for (int targetClusterIndex = 0; targetClusterIndex < target.ClusterInfos.Count; targetClusterIndex += 1)
            {
                var targetClusterInfo = target.ClusterInfos[targetClusterIndex];
                if (targetClusterInfo is null)
                    continue;

                float energy = ModelHelper.GetEnergy(sourceClusterInfo.CentroidOldVectorNormalized_Mapped, targetClusterInfo.CentroidOldVectorNormalized);
                if (energy < minEnergy)
                {
                    minEnergy = energy;
                    selected = targetClusterIndex;
                }
            }
            if (selected != -1)
            {
                clustersMapping[sourceClusterIndex] = selected;

                var selectedTarget = target.ClusterInfos[selected];
                selectedTarget.Concentration = target.Concentrations[selected].item<float>();
                selectedTarget.MixingCoefficient = target.MixingCoefficients[selected].item<float>();
            }
        }
        var targetClusterIndices = clustersMapping.ToHashSet();
        foreach (var targetClusterIndex in Enumerable.Range(0, target.ClusterInfos.Count))
        {
            if (targetClusterIndices.Contains(targetClusterIndex))
                continue;

            target.ClusterInfos[targetClusterIndex] = null!;
        }

        var clustersMapping_Reverse = new int[target.ClusterInfos.Count];
        Array.Fill(clustersMapping_Reverse, -1);
        for (int targetClusterIndex = 0; targetClusterIndex < target.ClusterInfos.Count; targetClusterIndex += 1)
        {
            var targetClusterInfo = target.ClusterInfos[targetClusterIndex];
            if (targetClusterInfo is null)
                continue;

            // Ищем позицию B с максимальным весом среди неиспользованных
            float minEnergy = float.MaxValue;
            int selected = -1;
            for (int sourceClusterIndex = 0; sourceClusterIndex < source.ClusterInfos.Count; sourceClusterIndex += 1)
            {
                var sourceClusterInfo = source.ClusterInfos[sourceClusterIndex];
                if (sourceClusterInfo is null)
                    continue;

                float energy = ModelHelper.GetEnergy(sourceClusterInfo.CentroidOldVectorNormalized_Mapped!, targetClusterInfo.CentroidOldVectorNormalized);
                if (energy < minEnergy)
                {
                    minEnergy = energy;
                    selected = sourceClusterIndex;
                }
            }
            if (selected != -1)
            {
                clustersMapping_Reverse[targetClusterIndex] = selected;
            }
        }
        var sourceClusterIndices = clustersMapping_Reverse.ToHashSet();
        foreach (var sourceClusterIndex in Enumerable.Range(0, source.ClusterInfos.Count))
        {
            if (sourceClusterIndices.Contains(sourceClusterIndex))
                continue;

            source.ClusterInfos[sourceClusterIndex] = null!;
        }

        //_userFriendlyLogger.LogInformation($"Количество уникальных сопоставлений: {hsA.Count}, {hsB.Count}");
    }

    public void Fix(LanguageDiscreteEmbeddings embeddings_A, LanguageDiscreteEmbeddings embeddings_B, Random r)
    {        
        foreach (int i in Enumerable.Range(0, embeddings_A.ClusterInfos.Count))
        {
            ModelHelper.SetClusterStatistics(embeddings_A.ClusterInfos[i], embeddings_A.Words.Where(w => w.ClusterIndex == i).ToArray());
        }
        
        foreach (int i in Enumerable.Range(0, embeddings_B.ClusterInfos.Count))
        {
            ModelHelper.SetClusterStatistics(embeddings_B.ClusterInfos[i], embeddings_B.Words.Where(w => w.ClusterIndex == i).ToArray());
        }
    }

    public void FilterOptimized(LanguageDiscreteEmbeddings embeddings_A, LanguageDiscreteEmbeddings embeddings_B, Random r)
    {
        int[] newClusterIndices_A = new int[embeddings_A.ClusterInfos.Count];
        Array.Fill(newClusterIndices_A, -1);
        List<ClusterInfo> newClusters_A = new List<ClusterInfo>(embeddings_A.ClusterInfos.Count);
        foreach (int i in Enumerable.Range(0, embeddings_A.ClusterInfos.Count))
        {
            var clusterInfo = embeddings_A.ClusterInfos[i];
            if (clusterInfo is null)
                continue;

            clusterInfo.Temp_ClusterIndex = i;
            newClusterIndices_A[i] = newClusters_A.Count;
            newClusters_A.Add(clusterInfo);
        }
        embeddings_A.ClusterInfos = newClusters_A.ToList();
        embeddings_A.Words = embeddings_A.Words.Where(w => newClusterIndices_A[w.ClusterIndex] != -1)
            .Select(w =>
            {
                w.ClusterIndex = newClusterIndices_A[w.ClusterIndex];
                return w;
            }
            ).ToList();
        for (int i = 0; i < embeddings_A.Words.Count; i += 1)
        {
            Model03Core.Word word = embeddings_A.Words[i];
            word.Index = i;
        }
        foreach (int i in Enumerable.Range(0, embeddings_A.ClusterInfos.Count))
        {
            ModelHelper.SetClusterStatistics(embeddings_A.ClusterInfos[i], embeddings_A.Words.Where(w => w.ClusterIndex == i).ToArray());            
        }
        SetHashProjectionIndices(embeddings_A.ClusterInfos, r);        

        foreach (int i in Enumerable.Range(0, embeddings_B.ClusterInfos.Count))
        {
            var clusterInfo = embeddings_B.ClusterInfos[i];
            if (clusterInfo is null)
                continue;
            clusterInfo.Temp_ClusterIndex = i;
        }
        var newClusterIndices_B = new int[embeddings_B.ClusterInfos.Count];
        Array.Fill(newClusterIndices_B, -1);
        var newClusters_B = new List<ClusterInfo>(embeddings_B.ClusterInfos.Count);        
        foreach (int i in Enumerable.Range(0, embeddings_A.ClusterInfos.Count))
        {
            var clusterInfo_A = embeddings_A.ClusterInfos[i];
            int mappedClusterIndex_B = ClustersMapping[clusterInfo_A.Temp_ClusterIndex];
            var clusterInfo_B = embeddings_B.ClusterInfos[mappedClusterIndex_B];

            Debug.Assert(clusterInfo_B is not null);
            newClusterIndices_B[clusterInfo_B.Temp_ClusterIndex] = newClusters_B.Count;            
            newClusters_B.Add(clusterInfo_B);
        }
        embeddings_B.ClusterInfos = newClusters_B.ToList();
        embeddings_B.Words = embeddings_B.Words.Where(w => newClusterIndices_B[w.ClusterIndex] != -1)
            .Select(w =>
            {
                w.ClusterIndex = newClusterIndices_B[w.ClusterIndex];
                return w;
            }
            ).ToList();
        for (int i = 0; i < embeddings_B.Words.Count; i += 1)
        {
            Model03Core.Word word = embeddings_B.Words[i];
            word.Index = i;
        }
        foreach (int i in Enumerable.Range(0, embeddings_B.ClusterInfos.Count))
        {
            ModelHelper.SetClusterStatistics(embeddings_B.ClusterInfos[i], embeddings_B.Words.Where(w => w.ClusterIndex == i).ToArray());            
        }
        SetHashProjectionIndices(embeddings_B.ClusterInfos, r);
    }

    public static void SetHashProjectionIndices(List<ClusterInfo> clusterInfos, Random r)
    {
        int[] hashProjectionIndices = new int[clusterInfos.Count];
        foreach (int clusterIndex in Enumerable.Range(0, clusterInfos.Count))
        {
            hashProjectionIndices[clusterIndex] = clusterIndex;
        }
        r.Shuffle(hashProjectionIndices);
        foreach (int clusterIndex in Enumerable.Range(0, clusterInfos.Count))
        {
            clusterInfos[clusterIndex].HashProjectionIndex = hashProjectionIndices[clusterIndex];
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
        Array.Fill(ClustersMapping, -1);

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
            if (sourceClusterInfo is null)
                continue;

            //var norm = TensorPrimitives.Norm(sourcePrimaryWord.OldVectorNormalized); // TEST
            //norm = TensorPrimitives.Norm(sourcePrimaryWord.OldVector); // TEST

            var oldVectorTensor = torch.tensor(sourceClusterInfo.CentroidOldVectorNormalized).reshape([1, sourceClusterInfo.CentroidOldVectorNormalized.Length]);
            var mappedOldVectorTensor = mappingLinear.forward(oldVectorTensor);
            float[] mappedOldVectorNormalized = mappedOldVectorTensor.data<float>().ToArray();
            float norm = TensorPrimitives.Norm(mappedOldVectorNormalized);
            TensorPrimitives.Divide(mappedOldVectorNormalized, norm, mappedOldVectorNormalized);
            sourceClusterInfo.CentroidOldVectorNormalized_Mapped = mappedOldVectorNormalized;

            // Ищем позицию B с максимальным весом среди неиспользованных
            float minEnergy = float.MaxValue;
            int selected = -1;
            for (int targetClusterIndex = 0; targetClusterIndex < target.ClusterInfos.Count; targetClusterIndex += 1)
            {
                var targetClusterInfo = target.ClusterInfos[targetClusterIndex];
                if (targetClusterInfo is null)
                    continue;

                float energy = ModelHelper.GetEnergy(sourceClusterInfo.CentroidOldVectorNormalized_Mapped, targetClusterInfo.CentroidOldVectorNormalized);
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

