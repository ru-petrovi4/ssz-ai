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

    public int[] PrimaryWordsMapping = null!;

    public void CalculatePrimaryWordsMapping(LanguageDiscreteEmbeddings source, LanguageDiscreteEmbeddings target)
    {
        PrimaryWordsMapping = new int[source.PrimaryWords.Count];

        using Linear mappingLinear = Linear(
            inputSize: source.PrimaryWords[0].OldVector.Length,
            outputSize: source.PrimaryWords[0].OldVector.Length,
            hasBias: false);
        using (var _ = no_grad())
        {
            var loadedWeights = load(Path.Combine(@"Data", "best_mapping.pt"));
            mappingLinear.weight!.copy_(loadedWeights);
        }
        //mappingLinear.to();

        for (int sourcePrimaryWordIndex = 0; sourcePrimaryWordIndex < source.PrimaryWords.Count; sourcePrimaryWordIndex += 1)
        {
            var sourcePrimaryWord = source.PrimaryWords[sourcePrimaryWordIndex];

            //var norm = TensorPrimitives.Norm(sourcePrimaryWord.OldVectorNormalized); // TEST
            //norm = TensorPrimitives.Norm(sourcePrimaryWord.OldVector); // TEST

            var oldVectorTensor = torch.tensor(sourcePrimaryWord.OldVectorNormalized);
            var mappedOldVectorTensor = mappingLinear.forward(oldVectorTensor);
            float[] mappedOldVectorNormalized = mappedOldVectorTensor.data<float>().ToArray();
            float norm = TensorPrimitives.Norm(mappedOldVectorNormalized);            
            TensorPrimitives.Divide(mappedOldVectorNormalized, norm, mappedOldVectorNormalized);

            // Ищем позицию B с максимальным весом среди неиспользованных
            float max = float.MinValue;
            int selected = -1;
            for (int targetPrimaryWordIndex = 0; targetPrimaryWordIndex < target.PrimaryWords.Count; targetPrimaryWordIndex += 1)
            {
                var targetPrimaryWord = target.PrimaryWords[targetPrimaryWordIndex];

                float cosineSimilarity = TensorPrimitives.CosineSimilarity(mappedOldVectorNormalized, targetPrimaryWord.OldVectorNormalized);
                if (cosineSimilarity > max)
                {
                    max = cosineSimilarity;
                    selected = targetPrimaryWordIndex;
                }
            }
            if (selected != -1)
            {
                PrimaryWordsMapping[sourcePrimaryWordIndex] = selected;                
            }
            else
            {
            }
        }

        var hs = PrimaryWordsMapping.ToHashSet();
        _userFriendlyLogger.LogInformation($"Количество уникальных сопоставлений: {hs.Count}");
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {        
        writer.WriteArray(PrimaryWordsMapping);
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {   
        PrimaryWordsMapping = reader.ReadArray<int>()!;
    }

    #region private fields

    private IUserFriendlyLogger _userFriendlyLogger;
    private Model03.Parameters _parameters;

    #endregion
}

