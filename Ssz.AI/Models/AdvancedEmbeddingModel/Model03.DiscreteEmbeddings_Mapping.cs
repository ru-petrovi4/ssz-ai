using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Numerics.Tensors;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Providers.LinearAlgebra;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Ssz.AI.Grafana;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using TorchSharp;
using TorchSharp.Modules; // Для TensorPrimitives
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public partial class Model03
{
    // Obsolete
    public const string FileName_HypothesisSupport = "AdvancedEmbedding_HypothesisSupport.bin";
    public const string FileName_DistanceMatrixA = "AdvancedEmbedding_DistanceMatrixA.bin";
    public const string FileName_NearestA = "AdvancedEmbedding_NearestA.bin";
    public const string FileName_DistanceMatrixB = "AdvancedEmbedding_DistanceMatrixB.bin";
    public const string FileName_NearestB = "AdvancedEmbedding_NearestB.bin";
    //public const string FileName_OldVectors_PrimaryWordsOneToOneMatcher = "OldVectors_PrimaryWordsOneToOneMatcher.bin";

    public const string FileName_PrimaryWordsOneToOneMatcher_V2 = "AdvancedEmbedding_PrimaryWordsOneToOneMatcher_V2.bin";

    #region construction and destruction

    public Model03()
    {
        _loggersSet = new LoggersSet(NullLogger.Instance, new UserFriendlyLogger((l, id, s) => DebugWindow.Instance.AddLine(s)));
    }

    #endregion

    public void OptimizeClusters()
    {
        //LanguageDiscreteEmbeddings languageDiscreteEmbeddings_RU = new();
        //Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_RU, null, null);

        //LanguageDiscreteEmbeddings languageDiscreteEmbeddings_EN = new();
        //Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_EN, languageDiscreteEmbeddings_EN, null, null);


        //ClustersOneToOneMatcher_MappingLinear clustersOneToOneMatcher_MappingLinear = new(_loggersSet.UserFriendlyLogger);
        //clustersOneToOneMatcher_MappingLinear.CalculateClustersMapping_V1(languageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_EN);

        //clustersOneToOneMatcher_MappingLinear = new(_loggersSet.UserFriendlyLogger);
        //clustersOneToOneMatcher_MappingLinear.CalculateClustersMapping_V2(languageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_EN);

        //clustersOneToOneMatcher_MappingLinear.ComputeDetailedEvaluationReport(languageDiscreteEmbeddings_RU, _loggersSet.UserFriendlyLogger);
        //clustersOneToOneMatcher_MappingLinear.ComputeDetailedEvaluationReport(languageDiscreteEmbeddings_EN, _loggersSet.UserFriendlyLogger);        

        //Helpers.SerializationHelper.SaveToFile(FileName_OldVectors_PrimaryWordsOneToOneMatcher, clustersOneToOneMatcher_MappingLinear, null, _loggersSet.UserFriendlyLogger);
    }

    public void Find_ClustersOneToOneMatcher_MappingLinear()
    {
        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_RU = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_RU, null, null);

        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_EN = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_EN, languageDiscreteEmbeddings_EN, null, null);

        bool calculate = true;
        ClustersOneToOneMatcher_MappingLinear clustersOneToOneMatcher_MappingLinear = new(_loggersSet.UserFriendlyLogger);
        if (calculate)
        {
            clustersOneToOneMatcher_MappingLinear.CalculateClustersMapping_EnergyMatrixHungarian(languageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_EN);
            //Helpers.SerializationHelper.SaveToFile(FileName_OldVectors_PrimaryWordsOneToOneMatcher, clustersOneToOneMatcher_MappingLinear, null, _loggersSet.UserFriendlyLogger);
        }
        else
        {
            //Helpers.SerializationHelper.LoadFromFileIfExists(FileName_OldVectors_PrimaryWordsOneToOneMatcher, clustersOneToOneMatcher_MappingLinear, null, null);
        }
        ModelHelper.ShowWords(languageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_EN, clustersOneToOneMatcher_MappingLinear.ClustersMapping, _loggersSet.UserFriendlyLogger);
    }

    /// <summary>
    /// Поддержка гипотез ближайших
    /// </summary>
    public void Find_ClustersOneToOneMatcher_Hypothesis()
    {
        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_RU = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_RU, null, null);

        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_EN = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_EN, languageDiscreteEmbeddings_EN, null, null);          

        var clustersOneToOneMatcher_Hypothesis = new ClustersOneToOneMatcher_Hypothesis(
            _loggersSet.UserFriendlyLogger,
            languageDiscreteEmbeddings_RU,
            languageDiscreteEmbeddings_EN);        

        bool calculate = true;
        if (calculate)
        {
            clustersOneToOneMatcher_Hypothesis.GenerateOwnedData();
            clustersOneToOneMatcher_Hypothesis.Prepare();

            clustersOneToOneMatcher_Hypothesis.SupportHypotheses_V1();
            //Helpers.SerializationHelper.SaveToFile(FileName_HypothesisSupport, matcher.HypothesisSupport, null, _loggersSet.UserFriendlyLogger);
        }
        else
        {
            //Helpers.SerializationHelper.LoadFromFileIfExists(FileName_HypothesisSupport, matcher, null, _loggersSet.UserFriendlyLogger);
        }

        //var resultMapping = matcher.GetFinalMapping();

        //// Выводим часть соответствий
        //_loggersSet.UserFriendlyLogger.LogInformation("Top 10 соответствий:");
        //foreach (var pair in resultMapping.Take(10))
        //{
        //    _loggersSet.UserFriendlyLogger.LogInformation($"{pair.Key} -> {pair.Value}");
        //}

        //var clustersMapping = matcher.GetFinalMappingForcedExclusive();
        var clustersMapping = clustersOneToOneMatcher_Hypothesis.GetFinalMapping();

        var resultBits = clustersMapping.ToHashSet();
        // Выводим часть соответствий
        _loggersSet.UserFriendlyLogger.LogInformation($"resultBits.Count: {resultBits.Count}");

        ModelHelper.ShowWords(languageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_EN, clustersMapping, _loggersSet.UserFriendlyLogger);
    }

    /// <summary>
    /// Перестановки
    /// </summary>
    public void Find_ClustersOneToOneMatcher_Swapping()
    {
        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_RU = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_RU, null, null);

        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_EN = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_EN, languageDiscreteEmbeddings_EN, null, null);        

        var clustersOneToOneMatcher_Swapping = new ClustersOneToOneMatcher_Swapping(_loggersSet, languageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_EN);
        bool calculateFromBeginning = false;
        if (calculateFromBeginning)
        {
            clustersOneToOneMatcher_Swapping.GenerateOwnedData(languageDiscreteEmbeddings_RU.ClusterInfos.Count);

            clustersOneToOneMatcher_Swapping.Prepare();
            clustersOneToOneMatcher_Swapping.CalculateMapping();
            
            Helpers.SerializationHelper.SaveToFile(FileName_PrimaryWordsOneToOneMatcher_V2, clustersOneToOneMatcher_Swapping, null, _loggersSet.UserFriendlyLogger);            
        }
        else
        {
            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_PrimaryWordsOneToOneMatcher_V2, clustersOneToOneMatcher_Swapping, null, _loggersSet.UserFriendlyLogger);

            clustersOneToOneMatcher_Swapping.Prepare();
            clustersOneToOneMatcher_Swapping.CalculateMapping();

            Helpers.SerializationHelper.SaveToFile(FileName_PrimaryWordsOneToOneMatcher_V2, clustersOneToOneMatcher_Swapping, null, _loggersSet.UserFriendlyLogger);
        }

        var clustersMapping = clustersOneToOneMatcher_Swapping.Mapping_RU_EN;

        var resultBits = clustersMapping.ToHashSet();
        // Выводим часть соответствий
        _loggersSet.UserFriendlyLogger.LogInformation($"resultBits.Count: {resultBits.Count}");

        ModelHelper.ShowWords(languageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_EN, clustersMapping, _loggersSet.UserFriendlyLogger);
    }    

    /// <summary>
    ///     Показывает количество примеров в каждом бите [0..300)
    /// </summary>
    public void VisualizeData_WordsBitsDistribution()
    {
        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_RU = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_RU, null, null);

        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_EN = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_EN, languageDiscreteEmbeddings_EN, null, null);

        //var matcher = new OneToOneMatcher(_loggersSet.UserFriendlyLogger, new Parameters());
        ////Helpers.SerializationHelper.LoadFromFileIfExists(FileName_HypothesisSupport, matcher.HypothesisSupport, _loggersSet.UserFriendlyLogger);
        //matcher.NearestA = new OneToOneMatcher.Nearest();
        //Helpers.SerializationHelper.LoadFromFileIfExists(FileName_NearestA, matcher.NearestA, null, _loggersSet.UserFriendlyLogger);        

        var dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
        dataToDisplayHolder.Distribution = new ulong[Model01.Constants.DiscreteVectorLength];

        var embeddings = languageDiscreteEmbeddings_EN;
        foreach (var i in Enumerable.Range(0, embeddings.Words.Count))
        {
            var word = embeddings.Words[i];
            var vec = word.DiscreteVector_PrimaryBitsOnly.AsSpan();
            for (int idx = 0; idx < Model01.Constants.DiscreteVectorLength; idx += 1)
            {
                if (vec[idx] > 0.5f)
                {
                    dataToDisplayHolder.Distribution[idx] += 1;
                }
            }
        }

        _loggersSet.UserFriendlyLogger.LogInformation($"VisualizeData_V1() Done.");
    }

    public void VisualizeData_V2()
    {
        //var matcher = new PrimaryWordsOneToOneMatcher(_loggersSet.UserFriendlyLogger, new Parameters());        
        //matcher.NearestA = new PrimaryWordsOneToOneMatcher.Nearest();
        //Helpers.SerializationHelper.LoadFromFileIfExists(FileName_NearestA, matcher.NearestA, null, _loggersSet.UserFriendlyLogger);
        //matcher.NearestB = new PrimaryWordsOneToOneMatcher.Nearest();
        //Helpers.SerializationHelper.LoadFromFileIfExists(FileName_NearestB, matcher.NearestB, null, _loggersSet.UserFriendlyLogger);

        //var dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
        //dataToDisplayHolder.Distribution = new ulong[Model01.Constants.DiscreteVectorLength];

        //var nearest = matcher.NearestB.Array;
        //for (int idx = 0; idx < Model01.Constants.DiscreteVectorLength; idx += 1)
        //{
        //    //dataToDisplayHolder.Distribution[idxB] += 1;
        //    // Подкрепляем также все пары в 16 ближайших
        //    foreach (var nearIdx in nearest[idx].Items)
        //    {
        //        dataToDisplayHolder.Distribution[nearIdx] += 1;
        //    }
        //}

        //_loggersSet.UserFriendlyLogger.LogInformation($"VisualizeData_V1() Done.");
    }

    /// <summary>
    /// Показывает распредление косинусных расстояний клстеров
    /// </summary>
    public void VisualizeData_GetСlustersCosineSimilarityMatrix()
    {
        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_RU = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_RU, null, null);

        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_EN = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_EN, languageDiscreteEmbeddings_EN, null, null);
        
        var clustersCosineSimilarityMatrix = languageDiscreteEmbeddings_EN.GetClustersCosineSimilarityMatrixFloat();

        //var matcher = new OneToOneMatcher(_loggersSet.UserFriendlyLogger, new Parameters());
        ////Helpers.SerializationHelper.LoadFromFileIfExists(FileName_HypothesisSupport, matcher.HypothesisSupport, _loggersSet.UserFriendlyLogger);
        //matcher.NearestA = new OneToOneMatcher.Nearest();
        //Helpers.SerializationHelper.LoadFromFileIfExists(FileName_NearestA, matcher.NearestA, null, _loggersSet.UserFriendlyLogger);        

        int n = 1000;
        var dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
        dataToDisplayHolder.Distribution = new ulong[n];
        dataToDisplayHolder.DistributionMin = 0.0f;
        dataToDisplayHolder.DistributionMax = 2.0f;

        foreach (var i in Enumerable.Range(0, clustersCosineSimilarityMatrix.Data.Length))
        {
            var v = 1 - clustersCosineSimilarityMatrix.Data[i];            
            dataToDisplayHolder.Distribution[(int)((v - dataToDisplayHolder.DistributionMin) * n / (dataToDisplayHolder.DistributionMax - dataToDisplayHolder.DistributionMin))] += 1;
        }

        _loggersSet.UserFriendlyLogger.LogInformation($"VisualizeData_GetСlustersCosineSimilarityMatrix() Done.");
    }

    /// <summary>
    /// Показывает распредление энергий клстеров
    /// </summary>
    public void VisualizeData_СlustersEnergyMatrix()
    {
        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_RU = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_RU, null, null);

        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_EN = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_EN, languageDiscreteEmbeddings_EN, null, null);

        var clustersEnergy_Matrix = ModelHelper.GetClustersEnergy_Matrix(languageDiscreteEmbeddings_EN);

        //var matcher = new OneToOneMatcher(_loggersSet.UserFriendlyLogger, new Parameters());
        ////Helpers.SerializationHelper.LoadFromFileIfExists(FileName_HypothesisSupport, matcher.HypothesisSupport, _loggersSet.UserFriendlyLogger);
        //matcher.NearestA = new OneToOneMatcher.Nearest();
        //Helpers.SerializationHelper.LoadFromFileIfExists(FileName_NearestA, matcher.NearestA, null, _loggersSet.UserFriendlyLogger);        

        int n = 1000;
        var dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
        dataToDisplayHolder.Distribution = new ulong[n];
        dataToDisplayHolder.DistributionMin = 0.0f;
        dataToDisplayHolder.DistributionMax = 10.0f;

        foreach (var i in Enumerable.Range(0, clustersEnergy_Matrix.Data.Length))
        {
            var v = clustersEnergy_Matrix.Data[i];            
            dataToDisplayHolder.Distribution[(int)((v - dataToDisplayHolder.DistributionMin) * n / (dataToDisplayHolder.DistributionMax - dataToDisplayHolder.DistributionMin))] += 1;
        }

        _loggersSet.UserFriendlyLogger.LogInformation($"VisualizeData_СlustersEnergyMatrix() Done.");
    }

    /// <summary>
    /// Показывает распредление энергий клстеров после линейного преобразования
    /// </summary>
    public void VisualizeData_СlustersEnergyMatrix_MappingLinear()
    {
        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_RU = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_RU, null, null);

        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_EN = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_EN, languageDiscreteEmbeddings_EN, null, null);

        using Linear mappingLinear = Linear(
            inputSize: languageDiscreteEmbeddings_RU.Words[0].OldVector.Length,
            outputSize: languageDiscreteEmbeddings_RU.Words[0].OldVector.Length,
            hasBias: false);
        using (var _ = no_grad())
        {
            var loadedWeights = load(Path.Combine(@"Data", Model02.FileName_MUSE_Procrustes_RU_EN));
            mappingLinear.weight!.copy_(loadedWeights);
        }

        int dimension = languageDiscreteEmbeddings_RU.ClusterInfos.Count;
        var clustersEnergy_Matrix = new MatrixFloat(dimension, dimension);
        foreach (var i in Enumerable.Range(0, dimension))
        {
            foreach (var j in Enumerable.Range(0, dimension))
            {
                var clusterI = languageDiscreteEmbeddings_RU.ClusterInfos[i];
                var clusterJ = languageDiscreteEmbeddings_RU.ClusterInfos[j];

                var oldVectorTensorI = torch.tensor(clusterI.CentroidOldVectorNormalized).reshape([1, clusterI.CentroidOldVectorNormalized.Length]);
                var mappedOldVectorTensorI = mappingLinear.forward(oldVectorTensorI);
                float[] mappedOldVectorNormalizedI = mappedOldVectorTensorI.data<float>().ToArray();
                float norm = TensorPrimitives.Norm(mappedOldVectorNormalizedI);
                TensorPrimitives.Divide(mappedOldVectorNormalizedI, norm, mappedOldVectorNormalizedI);

                var oldVectorTensorJ = torch.tensor(clusterJ.CentroidOldVectorNormalized).reshape([1, clusterJ.CentroidOldVectorNormalized.Length]);
                var mappedOldVectorTensorJ = mappingLinear.forward(oldVectorTensorJ);
                float[] mappedOldVectorNormalizedJ = mappedOldVectorTensorJ.data<float>().ToArray();
                norm = TensorPrimitives.Norm(mappedOldVectorNormalizedJ);
                TensorPrimitives.Divide(mappedOldVectorNormalizedJ, norm, mappedOldVectorNormalizedJ);

                float v = ModelHelper.GetEnergy(
                    mappedOldVectorNormalizedI,
                    mappedOldVectorNormalizedJ);
                clustersEnergy_Matrix[clusterI.HashProjectionIndex, clusterJ.HashProjectionIndex] = v;
            }
        }

        int n = 1000;
        var dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
        dataToDisplayHolder.Distribution = new ulong[n];
        dataToDisplayHolder.DistributionMin = 0.0f;
        dataToDisplayHolder.DistributionMax = 10.0f;
        
        foreach (var i in Enumerable.Range(0, clustersEnergy_Matrix.Data.Length))
        {
            var v = clustersEnergy_Matrix.Data[i];
            dataToDisplayHolder.Distribution[(int)((v - dataToDisplayHolder.DistributionMin) * n / (dataToDisplayHolder.DistributionMax - dataToDisplayHolder.DistributionMin))] += 1;
        }

        _loggersSet.UserFriendlyLogger.LogInformation($"VisualizeData_СlustersEnergyMatrix_MappingLinear() Done.");
    }

    /// <summary>
    /// Показывает распредление энергий слов
    /// </summary>
    public void VisualizeData_WordsEnergy()
    {
        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_RU = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_RU, null, null);

        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_EN = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_EN, languageDiscreteEmbeddings_EN, null, null);

        var matrix = ModelHelper.GetClustersEnergy_Matrix(languageDiscreteEmbeddings_EN);

        //float energy = 0.0f;
        //int wordsCount = 10000; 
        //for (int wordIndex = 0; wordIndex < wordsCount; wordIndex += 1)
        //{
        //    energy += PrimaryWordsOneToOneMatcher_V2.GetWordEnergy(languageDiscreteEmbeddings_RU.Words[wordIndex], matrix);
        //}
        //energy = energy / wordsCount;

        int n = 500;
        var dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
        dataToDisplayHolder.Distribution = new ulong[n];
        dataToDisplayHolder.DistributionMin = 0.0f;
        dataToDisplayHolder.DistributionMax = 5000.0f;

        int wordsCount = 10000;
        foreach (var wordIndex in Enumerable.Range(0, wordsCount))
        {
            var v = ModelHelper.GetWord_PrimaryBitsOnly_Energy(languageDiscreteEmbeddings_EN.Words[wordIndex], matrix);
            dataToDisplayHolder.Distribution[(int)((v - dataToDisplayHolder.DistributionMin) * n / (dataToDisplayHolder.DistributionMax - dataToDisplayHolder.DistributionMin))] += 1;
        }

        _loggersSet.UserFriendlyLogger.LogInformation($"VisualizeData_V4() Done.");
    }

    /// <summary>
    /// Показывает разницу между энергиями слова и его перевода.
    /// </summary>
    public void VisualizeData_WordsTranslationsEnergyDifference()
    {
        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_RU = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_RU, null, null);

        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_EN = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_EN, languageDiscreteEmbeddings_EN, null, null);

        WordTranslationsCollection wordTranslationsCollection = new();
        Helpers.SerializationHelper.LoadFromFileIfExists("WordTranslationsCollection.bin", wordTranslationsCollection, null, null);

        var matrix_RU = ModelHelper.GetClustersEnergy_Matrix(languageDiscreteEmbeddings_RU);
        var matrix_EN = ModelHelper.GetClustersEnergy_Matrix(languageDiscreteEmbeddings_EN);        

        int n = 500;
        var dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
        dataToDisplayHolder.Distribution = new ulong[n];
        dataToDisplayHolder.DistributionMin = -5000.0f;
        dataToDisplayHolder.DistributionMax = 5000.0f;
        
        foreach (var index in Enumerable.Range(0, wordTranslationsCollection.WordTranslations.Count))
        {
            var wordTranslation = wordTranslationsCollection.WordTranslations[index];
            var name_RU = languageDiscreteEmbeddings_RU.Words[wordTranslation.IndexA].Name; // For Assert
            var name_EN = languageDiscreteEmbeddings_EN.Words[wordTranslation.IndexB].Name; // For Assert
            var energy_RU = ModelHelper.GetWord_PrimaryBitsOnly_Energy(languageDiscreteEmbeddings_RU.Words[wordTranslation.IndexA], matrix_RU);
            var energy_EN = ModelHelper.GetWord_PrimaryBitsOnly_Energy(languageDiscreteEmbeddings_EN.Words[wordTranslation.IndexB], matrix_EN);
            float v = energy_EN - energy_RU;
            dataToDisplayHolder.Distribution[(int)((v - dataToDisplayHolder.DistributionMin) * n / (dataToDisplayHolder.DistributionMax - dataToDisplayHolder.DistributionMin))] += 1;
        }

        _loggersSet.UserFriendlyLogger.LogInformation($"VisualizeData_V5() Done.");
    }

    #region private fields

    private readonly ILoggersSet _loggersSet;

    #endregion
}

//public const string FileName_Mapping_V1 = "Mapping_V1.bin";
//public const string FileName_Mapping_V2 = "Mapping_V2.bin";

//public void FindDiscreteEmbeddings_Mapping_V1()
//{
//    WordsHelper.InitializeWords_RU(LanguageInfo_RU, _loggersSet, loadOldVectors: false);
//    WordsHelper.InitializeWords_EN(LanguageInfo_EN, _loggersSet, loadOldVectors: false);

//    LanguageInfo_RU.Clusterization_AlgorithmData = new Clusterization_AlgorithmData(LanguageInfo_RU, name: "KMeans");
//    Helpers.SerializationHelper.LoadFromFileIfExists(FileName_Clusterization_AlgorithmData_KMeans_RU, LanguageInfo_RU.Clusterization_AlgorithmData, null);
//    LanguageInfo_EN.Clusterization_AlgorithmData = new Clusterization_AlgorithmData(LanguageInfo_EN, name: "KMeans");
//    Helpers.SerializationHelper.LoadFromFileIfExists(FileName_Clusterization_AlgorithmData_KMeans_EN, LanguageInfo_EN.Clusterization_AlgorithmData, null);

//    LanguageInfo_RU.ProjectionOptimization_AlgorithmData = new ProjectionOptimization_AlgorithmData(name: "Variant3");
//    Helpers.SerializationHelper.LoadFromFileIfExists(FileName_ProjectionOptimization_AlgorithmData_Variant3_RU, LanguageInfo_RU.ProjectionOptimization_AlgorithmData, null);
//    LanguageInfo_EN.ProjectionOptimization_AlgorithmData = new ProjectionOptimization_AlgorithmData(name: "Variant3");
//    Helpers.SerializationHelper.LoadFromFileIfExists(FileName_ProjectionOptimization_AlgorithmData_Variant3_EN, LanguageInfo_EN.ProjectionOptimization_AlgorithmData, null);

//    LanguageInfo_RU.DiscreteVectorsAndMatrices = new DiscreteVectorsAndMatrices();
//    Helpers.SerializationHelper.LoadFromFileIfExists(FileName_DiscreteVectors_RU, LanguageInfo_RU.DiscreteVectorsAndMatrices, null);
//    LanguageInfo_EN.DiscreteVectorsAndMatrices = new DiscreteVectorsAndMatrices();
//    Helpers.SerializationHelper.LoadFromFileIfExists(FileName_DiscreteVectors_EN, LanguageInfo_EN.DiscreteVectorsAndMatrices, null);

//    MappingData_V1 mappingData = new(_loggersSet, LanguageInfo_RU, LanguageInfo_EN);
//    bool calculate = true;
//    if (calculate)
//    {
//        mappingData.GenerateOwnedData(Constants.DiscreteVectorLength);
//        mappingData.Prepare();
//        mappingData.CalculateMapping();
//        Helpers.SerializationHelper.SaveToFile(FileName_Mapping_V1, mappingData, null);
//    }
//    else
//    {
//        Helpers.SerializationHelper.LoadFromFileIfExists(FileName_Mapping_V1, mappingData, null);
//        mappingData.Prepare();
//        mappingData.CalculateMapping();
//    }
//}

//public void FindDiscreteEmbeddings_Mapping_V2()
//{
//    WordsHelper.InitializeWords_RU(LanguageInfo_RU, _loggersSet, loadOldVectors: true);
//    WordsHelper.InitializeWords_EN(LanguageInfo_EN, _loggersSet, loadOldVectors: true);

//    LanguageInfo_RU.Clusterization_AlgorithmData = new Clusterization_AlgorithmData(LanguageInfo_RU, name: "KMeans");
//    Helpers.SerializationHelper.LoadFromFileIfExists(FileName_Clusterization_AlgorithmData_KMeans_RU, LanguageInfo_RU.Clusterization_AlgorithmData, null);
//    LanguageInfo_EN.Clusterization_AlgorithmData = new Clusterization_AlgorithmData(LanguageInfo_EN, name: "KMeans");
//    Helpers.SerializationHelper.LoadFromFileIfExists(FileName_Clusterization_AlgorithmData_KMeans_EN, LanguageInfo_EN.Clusterization_AlgorithmData, null);

//    LanguageInfo_RU.ProjectionOptimization_AlgorithmData = new ProjectionOptimization_AlgorithmData(name: "Variant3");
//    Helpers.SerializationHelper.LoadFromFileIfExists(FileName_ProjectionOptimization_AlgorithmData_Variant3_RU, LanguageInfo_RU.ProjectionOptimization_AlgorithmData, null);
//    LanguageInfo_EN.ProjectionOptimization_AlgorithmData = new ProjectionOptimization_AlgorithmData(name: "Variant3");
//    Helpers.SerializationHelper.LoadFromFileIfExists(FileName_ProjectionOptimization_AlgorithmData_Variant3_EN, LanguageInfo_EN.ProjectionOptimization_AlgorithmData, null);

//    LanguageInfo_RU.DiscreteVectorsAndMatrices = new DiscreteVectorsAndMatrices();
//    Helpers.SerializationHelper.LoadFromFileIfExists(FileName_DiscreteVectors_RU, LanguageInfo_RU.DiscreteVectorsAndMatrices, null);
//    LanguageInfo_EN.DiscreteVectorsAndMatrices = new DiscreteVectorsAndMatrices();
//    Helpers.SerializationHelper.LoadFromFileIfExists(FileName_DiscreteVectors_EN, LanguageInfo_EN.DiscreteVectorsAndMatrices, null);

//    MappingData_V2 mappingData = new(_loggersSet, LanguageInfo_RU, LanguageInfo_EN);
//    bool calculate = true;
//    if (calculate)
//    {
//        mappingData.GenerateOwnedData(Constants.DiscreteVectorLength);
//        mappingData.Prepare2_DiscreteVectors();            
//        mappingData.CalculatePrimaryWordsMapping();
//        mappingData.DisplayWords();
//        Helpers.SerializationHelper.SaveToFile(FileName_Mapping_V2, mappingData, null);            
//    }
//    else
//    {
//        Helpers.SerializationHelper.LoadFromFileIfExists(FileName_Mapping_V2, mappingData, null);
//        mappingData.Prepare();
//        mappingData.DisplayWords();
//    }
//}

//public void FindEmbeddings_Mapping_V3()
//{
//    WordsHelper.InitializeWords_RU(LanguageInfo_RU, _loggersSet, loadOldVectors: true);        
//    WordsHelper.InitializeWords_EN(LanguageInfo_EN, _loggersSet, loadOldVectors: true);

//    LanguageInfo_RU.Clusterization_AlgorithmData = new Clusterization_AlgorithmData(LanguageInfo_RU, name: "KMeans");
//    Helpers.SerializationHelper.LoadFromFileIfExists(FileName_Clusterization_AlgorithmData_KMeans_RU, LanguageInfo_RU.Clusterization_AlgorithmData, null);
//    LanguageInfo_EN.Clusterization_AlgorithmData = new Clusterization_AlgorithmData(LanguageInfo_EN, name: "KMeans");
//    Helpers.SerializationHelper.LoadFromFileIfExists(FileName_Clusterization_AlgorithmData_KMeans_EN, LanguageInfo_EN.Clusterization_AlgorithmData, null);

//    LanguageInfo_RU.ProjectionOptimization_AlgorithmData = new ProjectionOptimization_AlgorithmData(name: "Variant3");
//    Helpers.SerializationHelper.LoadFromFileIfExists(FileName_ProjectionOptimization_AlgorithmData_Variant3_RU, LanguageInfo_RU.ProjectionOptimization_AlgorithmData, null);
//    LanguageInfo_EN.ProjectionOptimization_AlgorithmData = new ProjectionOptimization_AlgorithmData(name: "Variant3");
//    Helpers.SerializationHelper.LoadFromFileIfExists(FileName_ProjectionOptimization_AlgorithmData_Variant3_EN, LanguageInfo_EN.ProjectionOptimization_AlgorithmData, null);

//    LanguageInfo_RU.DiscreteVectorsAndMatrices = new DiscreteVectorsAndMatrices();
//    Helpers.SerializationHelper.LoadFromFileIfExists(FileName_DiscreteVectors_RU, LanguageInfo_RU.DiscreteVectorsAndMatrices, null);
//    LanguageInfo_EN.DiscreteVectorsAndMatrices = new DiscreteVectorsAndMatrices();
//    Helpers.SerializationHelper.LoadFromFileIfExists(FileName_DiscreteVectors_EN, LanguageInfo_EN.DiscreteVectorsAndMatrices, null);

//    MappingData_V3 mappingData = new(_loggersSet, LanguageInfo_RU, LanguageInfo_EN);
//    bool calculate = true;
//    if (calculate)
//    {
//        mappingData.GenerateOwnedData(Constants.DiscreteVectorLength);
//        mappingData.Prepare3_AnalogVectors();
//        mappingData.CalculatePrimaryWordsMapping();
//        mappingData.DisplayWords();
//        Helpers.SerializationHelper.SaveToFile(FileName_Mapping_V2, mappingData, null);
//    }
//    else
//    {
//        Helpers.SerializationHelper.LoadFromFileIfExists(FileName_Mapping_V2, mappingData, null);
//        //mappingData.Prepare();
//        //mappingData.DisplayWords();
//    }
//}