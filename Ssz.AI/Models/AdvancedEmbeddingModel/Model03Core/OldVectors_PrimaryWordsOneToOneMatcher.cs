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

public class OldVectors_PrimaryWordsOneToOneMatcher : IOwnedDataSerializable
{
    public OldVectors_PrimaryWordsOneToOneMatcher(IUserFriendlyLogger userFriendlyLogger, Model03.Parameters parameters)
    {
        _userFriendlyLogger = userFriendlyLogger;
        _parameters = parameters;
    }

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
            var loadedWeights = load(Path.Combine(@"Data", Model02.FileName_MUSE_Procrustes_RU_EN));
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

            // Ищем позицию B с максимальным весом среди неиспользованных
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

        var hs = ClustersMapping.ToHashSet();
        _userFriendlyLogger.LogInformation($"Количество уникальных сопоставлений: {hs.Count}");
    }

    /// <summary>
    ///     Clusters similarity
    /// </summary>
    /// <param name="source"></param>
    /// <param name="target"></param>
    public void CalculateClustersMapping_V2(LanguageDiscreteEmbeddings source, LanguageDiscreteEmbeddings target)
    {
        ClustersMapping = new int[source.ClusterInfos.Count];

        using Linear mappingLinear = Linear(
            inputSize: source.Words[0].OldVector.Length,
            outputSize: source.Words[0].OldVector.Length,
            hasBias: false);
        using (var _ = no_grad())
        {
            var loadedWeights = load(Path.Combine(@"Data", Model02.FileName_MUSE_Procrustes_RU_EN));
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

        var hs = ClustersMapping.ToHashSet();
        _userFriendlyLogger.LogInformation($"Количество уникальных сопоставлений: {hs.Count}");
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
    private Model03.Parameters _parameters;

    #endregion
}

