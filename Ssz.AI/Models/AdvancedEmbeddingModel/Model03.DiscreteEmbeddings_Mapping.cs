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

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public partial class Model03
{
    public const string FileName_HypothesisSupport = "AdvancedEmbedding_HypothesisSupport.bin";
    public const string FileName_DistanceMatrixA = "AdvancedEmbedding_DistanceMatrixA.bin";
    public const string FileName_NearestA = "AdvancedEmbedding_NearestA.bin";
    public const string FileName_DistanceMatrixB = "AdvancedEmbedding_DistanceMatrixB.bin";
    public const string FileName_NearestB = "AdvancedEmbedding_NearestB.bin";

    #region construction and destruction

    public Model03()
    {
        _loggersSet = new LoggersSet(NullLogger.Instance, new UserFriendlyLogger((l, id, s) => DebugWindow.Instance.AddLine(s)));
    }

    #endregion

    public void FindDiscreteEmbeddings_Mapping()
    {
        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_RU = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_RU, null);

        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_EN = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_EN, languageDiscreteEmbeddings_EN, null);

        var setA = languageDiscreteEmbeddings_RU.GetDiscrete_PrimaryBitsOnlyEmbeddingsMatrix();
        var setB = languageDiscreteEmbeddings_EN.GetDiscrete_PrimaryBitsOnlyEmbeddingsMatrix();          

        var matcher = new OneToOneMatcher(_loggersSet.UserFriendlyLogger, new Parameters());
        bool calculateDistanceMatrices = false;
        if (calculateDistanceMatrices)
        {
            matcher.BuildDistanceMatrix_V2(setA, matcher.DistanceMatrixA);
            Helpers.SerializationHelper.SaveToFile(FileName_DistanceMatrixA, matcher.DistanceMatrixA, null, _loggersSet.UserFriendlyLogger);
            matcher.NearestA = matcher.BuildNearest(matcher.DistanceMatrixA);
            Helpers.SerializationHelper.SaveToFile(FileName_NearestA, matcher.NearestA, null, _loggersSet.UserFriendlyLogger);

            matcher.BuildDistanceMatrix_V2(setB, matcher.DistanceMatrixB);
            Helpers.SerializationHelper.SaveToFile(FileName_DistanceMatrixB, matcher.DistanceMatrixB, null, _loggersSet.UserFriendlyLogger);
            matcher.NearestB = matcher.BuildNearest(matcher.DistanceMatrixB);
            Helpers.SerializationHelper.SaveToFile(FileName_NearestB, matcher.NearestB, null, _loggersSet.UserFriendlyLogger);
        }
        else
        {
            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_DistanceMatrixA, matcher.DistanceMatrixA, null, _loggersSet.UserFriendlyLogger);
            matcher.NearestA = new OneToOneMatcher.Nearest();
            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_NearestA, matcher.NearestA, null, _loggersSet.UserFriendlyLogger);

            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_DistanceMatrixB, matcher.DistanceMatrixB, null, _loggersSet.UserFriendlyLogger);
            matcher.NearestB = new OneToOneMatcher.Nearest();
            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_NearestB, matcher.NearestB, null, _loggersSet.UserFriendlyLogger);
        }

        matcher.SupportHypotheses_V2(setA, setB, count: 500);

        Helpers.SerializationHelper.SaveToFile(FileName_HypothesisSupport, matcher.HypothesisSupport, _loggersSet.UserFriendlyLogger);

        //var resultMapping = matcher.GetFinalMappingForcedExclusive();

        //// Выводим часть соответствий
        //_loggersSet.UserFriendlyLogger.LogInformation("Top 10 соответствий:");
        //foreach (var pair in resultMapping.Take(10))
        //{
        //    _loggersSet.UserFriendlyLogger.LogInformation($"{pair.Key} -> {pair.Value}");
        //}
    }

    public void TestQuality_DiscreteEmbeddings_Mapping()
    {
        var matcher = new OneToOneMatcher(_loggersSet.UserFriendlyLogger, new Parameters());
        Helpers.SerializationHelper.LoadFromFileIfExists(FileName_HypothesisSupport, matcher.HypothesisSupport, _loggersSet.UserFriendlyLogger);

        var resultBits = matcher.GetFinalMapping().Values.ToHashSet();

        // Выводим часть соответствий
        _loggersSet.UserFriendlyLogger.LogInformation($"resultBits.Count: {resultBits.Count}");
    }

    public void VisualizeData_V1()
    {
        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_RU = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_RU, null);

        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_EN = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_EN, languageDiscreteEmbeddings_EN, null);

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
        var matcher = new OneToOneMatcher(_loggersSet.UserFriendlyLogger, new Parameters());        
        matcher.NearestA = new OneToOneMatcher.Nearest();
        Helpers.SerializationHelper.LoadFromFileIfExists(FileName_NearestA, matcher.NearestA, null, _loggersSet.UserFriendlyLogger);
        matcher.NearestB = new OneToOneMatcher.Nearest();
        Helpers.SerializationHelper.LoadFromFileIfExists(FileName_NearestB, matcher.NearestB, null, _loggersSet.UserFriendlyLogger);

        var dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
        dataToDisplayHolder.Distribution = new ulong[Model01.Constants.DiscreteVectorLength];

        var nearest = matcher.NearestB.Array;
        for (int idx = 0; idx < Model01.Constants.DiscreteVectorLength; idx += 1)
        {
            //dataToDisplayHolder.Distribution[idxB] += 1;
            // Подкрепляем также все пары в 16 ближайших
            foreach (var nearIdx in nearest[idx].Items)
            {
                dataToDisplayHolder.Distribution[nearIdx] += 1;
            }
        }

        _loggersSet.UserFriendlyLogger.LogInformation($"VisualizeData_V1() Done.");
    }

    public sealed record Parameters
    {
        /// <summary>
        /// Количество ближайших для подкрепления
        /// </summary>
        public int NearestCount { get; set; } = 16;
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