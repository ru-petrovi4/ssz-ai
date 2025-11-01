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
using MathNet.Numerics.Providers.LinearAlgebra;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public partial class Model01
{
    #region construction and destruction

    public Model01()
    {
        _loggersSet = new LoggersSet(NullLogger.Instance, new UserFriendlyLogger((l, id, s) => DebugWindow.Instance.AddLine(s)));
    }

    #endregion

    #region public functions       

    public static readonly ModelConstants Constants = new();

    /// <summary>
    ///             
    /// </summary>
    public readonly LanguageInfo LanguageInfo_RU = new();

    /// <summary>
    ///            
    /// </summary>
    public readonly LanguageInfo LanguageInfo_EN = new();

    public Cortex Cortex = null!;

    /// <summary>
    ///     For display.
    ///     Lock CortexCopySyncRoot when read/write.
    /// </summary>
    public Cortex CortexCopy = new (0, 0);

    public readonly object CortexCopySyncRoot = new object();             
    
    public ProjectionOptimization_AlgorithmData? CurrentProjectionOptimization_AlgorithmData_ToDisplay;

    public DiscreteVectorsAndMatrices? CurrentDiscreteVectorsAndMatrices_ToDisplay;

    public WordsNewEmbeddings? CurrentWordsNewEmbeddings;

    public const string FileName_LanguageInfo_WordsDistancesOldMatrix_RU = "AdvancedEmbedding_LanguageInfo_WordsDistancesOldMatrix_RU.bin";
    public const string FileName_LanguageInfo_WordsDistancesOldMatrix_EN = "AdvancedEmbedding_LanguageInfo_WordsDistancesOldMatrix_EN.bin";
    public const string FileName_LanguageDiscreteEmbeddings_WordsDistancesOldMatrix_RU = "AdvancedEmbedding_LanguageDiscreteEmbeddings_WordsDistancesOldMatrix_RU.bin";
    public const string FileName_LanguageDiscreteEmbeddings_WordsDistancesOldMatrix_EN = "AdvancedEmbedding_LanguageDiscreteEmbeddings_WordsDistancesOldMatrix_EN.bin";
    public const string FileName_Clusterization_AlgorithmData_RU = "AdvancedEmbedding_Clusterization_AlgorithmData_RU.bin";
    public const string FileName_Clusterization_AlgorithmData_EN = "AdvancedEmbedding_Clusterization_AlgorithmData_EN.bin";
    public const string FileName_ProjectionOptimization_AlgorithmData_RU = "AdvancedEmbedding_ProjectionOptimization_AlgorithmData_RU.bin";
    public const string FileName_ProjectionOptimization_AlgorithmData_EN = "AdvancedEmbedding_ProjectionOptimization_AlgorithmData_EN.bin";
    public const string FileName_DiscreteVectors_RU = "AdvancedEmbedding_DiscreteVectors_RU.bin";
    public const string FileName_DiscreteVectors_EN = "AdvancedEmbedding_DiscreteVectors_EN.bin";
    public const string FileName_LanguageDiscreteEmbeddings_RU = "AdvancedEmbedding_LanguageDiscreteEmbeddings_RU.bin";
    public const string FileName_LanguageDiscreteEmbeddings_EN = "AdvancedEmbedding_LanguageDiscreteEmbeddings_EN.bin";

    public const int WordsCount = 19800;

    public void FindDiscreteEmbeddings()
    {
        WordsHelper.InitializeWords_RU(LanguageInfo_RU, wordsMaxCount: WordsCount, _loggersSet);
        WordsHelper.InitializeWords_EN(LanguageInfo_EN, wordsMaxCount: WordsCount, _loggersSet);

        bool calculate = true;
        if (calculate)
        {
            LanguageInfo_RU.Temp_WordsDistancesOldMatrix = WordsDistancesOldMatrix_Calculate(LanguageInfo_RU.Words, _loggersSet);
            Helpers.SerializationHelper.SaveToFile(FileName_LanguageInfo_WordsDistancesOldMatrix_RU, LanguageInfo_RU.Temp_WordsDistancesOldMatrix, null, _loggersSet.UserFriendlyLogger);

            LanguageInfo_EN.Temp_WordsDistancesOldMatrix = WordsDistancesOldMatrix_Calculate(LanguageInfo_EN.Words, _loggersSet);
            Helpers.SerializationHelper.SaveToFile(FileName_LanguageInfo_WordsDistancesOldMatrix_EN, LanguageInfo_EN.Temp_WordsDistancesOldMatrix, null, _loggersSet.UserFriendlyLogger);
        }
        else
        {
            LanguageInfo_RU.Temp_WordsDistancesOldMatrix = new MatrixFloat_ColumnMajor();
            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_LanguageInfo_WordsDistancesOldMatrix_RU, LanguageInfo_RU.Temp_WordsDistancesOldMatrix, null, null);
            LanguageInfo_EN.Temp_WordsDistancesOldMatrix = new MatrixFloat_ColumnMajor();
            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_LanguageInfo_WordsDistancesOldMatrix_EN, LanguageInfo_EN.Temp_WordsDistancesOldMatrix, null, null);
        }

        //Calculate_Clusterization_AlgorithmData_Random(_loggersSet);

        calculate = true;
        if (calculate)
        {
            Calculate_Clusterization_AlgorithmData_VonMisesFisherClusterer(LanguageInfo_RU, _loggersSet);
            Helpers.SerializationHelper.SaveToFile(FileName_Clusterization_AlgorithmData_RU, LanguageInfo_RU.Temp_Clusterization_AlgorithmData, null, _loggersSet.UserFriendlyLogger);

            Calculate_Clusterization_AlgorithmData_VonMisesFisherClusterer(LanguageInfo_EN, _loggersSet);
            Helpers.SerializationHelper.SaveToFile(FileName_Clusterization_AlgorithmData_EN, LanguageInfo_EN.Temp_Clusterization_AlgorithmData, null, _loggersSet.UserFriendlyLogger);
        }
        else
        {
            LanguageInfo_RU.Temp_Clusterization_AlgorithmData = new Clusterization_AlgorithmData(LanguageInfo_RU, name: "VonMisesFisherClusterer");
            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_Clusterization_AlgorithmData_RU, LanguageInfo_RU.Temp_Clusterization_AlgorithmData, null, null);
            LanguageInfo_EN.Temp_Clusterization_AlgorithmData = new Clusterization_AlgorithmData(LanguageInfo_EN, name: "VonMisesFisherClusterer");
            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_Clusterization_AlgorithmData_EN, LanguageInfo_EN.Temp_Clusterization_AlgorithmData, null, null);
        }

        calculate = true;
        if (calculate)
        {
            var r = new Random(42);

            LanguageInfo_RU.Temp_ProjectionOptimization_AlgorithmData = Calculate_ProjectionIndices_V4(
                LanguageInfo_RU.Words,
                LanguageInfo_RU.Temp_Clusterization_AlgorithmData.ClusterInfos.ToList(),
                LanguageInfo_RU.Temp_WordsDistancesOldMatrix,
                r, 
                _loggersSet);
            Helpers.SerializationHelper.SaveToFile(FileName_ProjectionOptimization_AlgorithmData_RU, LanguageInfo_RU.Temp_ProjectionOptimization_AlgorithmData, null, _loggersSet.UserFriendlyLogger);
            Helpers.SerializationHelper.SaveToFile(FileName_Clusterization_AlgorithmData_RU, LanguageInfo_RU.Temp_Clusterization_AlgorithmData, null, _loggersSet.UserFriendlyLogger); // Because ClusterInfos changed.

            LanguageInfo_EN.Temp_ProjectionOptimization_AlgorithmData = Calculate_ProjectionIndices_V4(
                LanguageInfo_EN.Words,
                LanguageInfo_EN.Temp_Clusterization_AlgorithmData.ClusterInfos.ToList(),
                LanguageInfo_EN.Temp_WordsDistancesOldMatrix,
                r,
                _loggersSet);
            Helpers.SerializationHelper.SaveToFile(FileName_ProjectionOptimization_AlgorithmData_EN, LanguageInfo_EN.Temp_ProjectionOptimization_AlgorithmData, null, _loggersSet.UserFriendlyLogger);
            Helpers.SerializationHelper.SaveToFile(FileName_Clusterization_AlgorithmData_EN, LanguageInfo_EN.Temp_Clusterization_AlgorithmData, null, _loggersSet.UserFriendlyLogger); // Because ClusterInfos changed.
        }
        else
        {
            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_ProjectionOptimization_AlgorithmData_RU, LanguageInfo_RU.Temp_ProjectionOptimization_AlgorithmData, null, null);
            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_ProjectionOptimization_AlgorithmData_EN, LanguageInfo_EN.Temp_ProjectionOptimization_AlgorithmData, null, null);
        }

        //CalculateDiscreteVectors(Clusterization_AlgorithmData_Random, ProjectionOptimization_AlgorithmData_Random, _loggersSet);
        //SaveToFile_DiscreteVectors(Clusterization_AlgorithmData_Random, _loggersSet);
        //LoadFromFile_DiscreteVectorsAndMatrices(Clusterization_AlgorithmData_Random, _loggersSet);
        //ProxWordsDiscreteMatrix_Calculate(Clusterization_AlgorithmData_Random, _loggersSet);            

        calculate = true;
        if (calculate)
        {
            LanguageInfo_RU.Temp_DiscreteVectorsAndMatrices = Calculate_DiscreteVectorsOnly(
                LanguageInfo_RU.Words,
                LanguageInfo_RU.Temp_Clusterization_AlgorithmData.ClusterInfos.ToList(),
                LanguageInfo_RU.Temp_WordsDistancesOldMatrix,
                LanguageInfo_RU.Temp_ProjectionOptimization_AlgorithmData,
                _loggersSet);
            Helpers.SerializationHelper.SaveToFile(FileName_DiscreteVectors_RU, LanguageInfo_RU.Temp_DiscreteVectorsAndMatrices, null, _loggersSet.UserFriendlyLogger);

            LanguageInfo_EN.Temp_DiscreteVectorsAndMatrices = Calculate_DiscreteVectorsOnly(
                LanguageInfo_EN.Words,
                LanguageInfo_EN.Temp_Clusterization_AlgorithmData.ClusterInfos.ToList(),
                LanguageInfo_EN.Temp_WordsDistancesOldMatrix,
                LanguageInfo_EN.Temp_ProjectionOptimization_AlgorithmData,
                _loggersSet);
            Helpers.SerializationHelper.SaveToFile(FileName_DiscreteVectors_EN, LanguageInfo_EN.Temp_DiscreteVectorsAndMatrices, null, _loggersSet.UserFriendlyLogger);
        }
        else
        {
            LanguageInfo_RU.Temp_DiscreteVectorsAndMatrices = new DiscreteVectorsAndMatrices();
            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_DiscreteVectors_RU, LanguageInfo_RU.Temp_DiscreteVectorsAndMatrices, null, null);

            LanguageInfo_EN.Temp_DiscreteVectorsAndMatrices = new DiscreteVectorsAndMatrices();
            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_DiscreteVectors_EN, LanguageInfo_EN.Temp_DiscreteVectorsAndMatrices, null, null);
        }

        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_RU = new()
        {
            Words = new(),
            ClusterInfos = LanguageInfo_RU.Temp_Clusterization_AlgorithmData.ClusterInfos.ToList(),
            MeanDirections = LanguageInfo_RU.Temp_Clusterization_AlgorithmData.MeanDirections,
            Concentrations = LanguageInfo_RU.Temp_Clusterization_AlgorithmData.Concentrations,
            MixingCoefficients = LanguageInfo_RU.Temp_Clusterization_AlgorithmData.MixingCoefficients,
        };
        for (int i = 0; i < LanguageInfo_RU.Words.Count; i += 1)
        {
            Model03Core.Word word = LanguageInfo_RU.Words[i];
            Model03Core.Word wordClone = new()
            {
                Name = word.Name,
                Index = i,
                ClusterIndex = word.ClusterIndex,
                OldVector = word.OldVector,
                OldVectorNormalized = word.OldVectorNormalized,
                DiscreteVector = LanguageInfo_RU.Temp_DiscreteVectorsAndMatrices.DiscreteVectors[i],
                DiscreteVector_PrimaryBitsOnly = LanguageInfo_RU.Temp_DiscreteVectorsAndMatrices.DiscreteVectors_PrimaryBitsOnly[i],
                DiscreteVector_SecondaryBitsOnly = LanguageInfo_RU.Temp_DiscreteVectorsAndMatrices.DiscreteVectors_SecondaryBitsOnly[i],
            };
#if DEBUG
            var primaryBitsSum = TensorPrimitives.Sum(wordClone.DiscreteVector_PrimaryBitsOnly);
            Debug.Assert(primaryBitsSum > Constants.DiscreteVector_PrimaryBitsCount - 0.00001f && primaryBitsSum < Constants.DiscreteVector_PrimaryBitsCount + 0.00001f);
#endif
            languageDiscreteEmbeddings_RU.Words.Add(wordClone);            
        }
        foreach (int i in Enumerable.Range(0, languageDiscreteEmbeddings_RU.ClusterInfos.Count))
        {
            ModelHelper.SetClusterStatistics(languageDiscreteEmbeddings_RU.ClusterInfos[i], languageDiscreteEmbeddings_RU.Words.Where(w => w.ClusterIndex == i).ToArray());
        }
        Helpers.SerializationHelper.SaveToFile(FileName_LanguageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_RU, null, _loggersSet.UserFriendlyLogger);

        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_EN = new()
        {
            Words = new(),
            ClusterInfos = LanguageInfo_EN.Temp_Clusterization_AlgorithmData.ClusterInfos.ToList(),
            MeanDirections = LanguageInfo_EN.Temp_Clusterization_AlgorithmData.MeanDirections,
            Concentrations = LanguageInfo_EN.Temp_Clusterization_AlgorithmData.Concentrations,
            MixingCoefficients = LanguageInfo_EN.Temp_Clusterization_AlgorithmData.MixingCoefficients,
        };
        for (int i = 0; i < LanguageInfo_EN.Words.Count; i += 1)
        {
            Model03Core.Word word = LanguageInfo_EN.Words[i];
            Model03Core.Word wordClone = new()
            {
                Name = word.Name,
                Index = i,
                ClusterIndex = word.ClusterIndex,
                OldVector = word.OldVector,
                OldVectorNormalized = word.OldVectorNormalized,
                DiscreteVector = LanguageInfo_EN.Temp_DiscreteVectorsAndMatrices.DiscreteVectors[i],
                DiscreteVector_PrimaryBitsOnly = LanguageInfo_EN.Temp_DiscreteVectorsAndMatrices.DiscreteVectors_PrimaryBitsOnly[i],
                DiscreteVector_SecondaryBitsOnly = LanguageInfo_EN.Temp_DiscreteVectorsAndMatrices.DiscreteVectors_SecondaryBitsOnly[i],
            };
#if DEBUG
            var primaryBitsSum = TensorPrimitives.Sum(wordClone.DiscreteVector_PrimaryBitsOnly);
            Debug.Assert(primaryBitsSum > Constants.DiscreteVector_PrimaryBitsCount - 0.00001f && primaryBitsSum < Constants.DiscreteVector_PrimaryBitsCount + 0.00001f);
#endif
            languageDiscreteEmbeddings_EN.Words.Add(wordClone);            
        }
        foreach (int i in Enumerable.Range(0, languageDiscreteEmbeddings_EN.ClusterInfos.Count))
        {
            ModelHelper.SetClusterStatistics(languageDiscreteEmbeddings_EN.ClusterInfos[i], languageDiscreteEmbeddings_EN.Words.Where(w => w.ClusterIndex == i).ToArray());
        }
        Helpers.SerializationHelper.SaveToFile(FileName_LanguageDiscreteEmbeddings_EN, languageDiscreteEmbeddings_EN, null, _loggersSet.UserFriendlyLogger);
    }

    public void FindDiscreteEmbeddings_LanguageDiscreteEmbeddings_Object()
    {
        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_RU = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_RU, null, null);        

        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_EN = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(Model01.FileName_LanguageDiscreteEmbeddings_EN, languageDiscreteEmbeddings_EN, null, null);        

        bool calculate = true;
        if (calculate)
        {
            languageDiscreteEmbeddings_RU.Temp_WordsDistancesOldMatrix = WordsDistancesOldMatrix_Calculate(languageDiscreteEmbeddings_RU.Words, _loggersSet);
            Helpers.SerializationHelper.SaveToFile(FileName_LanguageDiscreteEmbeddings_WordsDistancesOldMatrix_RU, languageDiscreteEmbeddings_RU.Temp_WordsDistancesOldMatrix, null, _loggersSet.UserFriendlyLogger);

            languageDiscreteEmbeddings_EN.Temp_WordsDistancesOldMatrix = WordsDistancesOldMatrix_Calculate(languageDiscreteEmbeddings_EN.Words, _loggersSet);
            Helpers.SerializationHelper.SaveToFile(FileName_LanguageDiscreteEmbeddings_WordsDistancesOldMatrix_EN, languageDiscreteEmbeddings_EN.Temp_WordsDistancesOldMatrix, null, _loggersSet.UserFriendlyLogger);
        }
        else
        {
            languageDiscreteEmbeddings_RU.Temp_WordsDistancesOldMatrix = new MatrixFloat_ColumnMajor();
            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_LanguageDiscreteEmbeddings_WordsDistancesOldMatrix_RU, languageDiscreteEmbeddings_RU.Temp_WordsDistancesOldMatrix, null, null);
            languageDiscreteEmbeddings_EN.Temp_WordsDistancesOldMatrix = new MatrixFloat_ColumnMajor();
            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_LanguageDiscreteEmbeddings_WordsDistancesOldMatrix_EN, languageDiscreteEmbeddings_EN.Temp_WordsDistancesOldMatrix, null, null);
        }       

        calculate = true;
        if (calculate)
        {
            var r = new Random(42);

            languageDiscreteEmbeddings_RU.Temp_ProjectionOptimization_AlgorithmData = Calculate_ProjectionIndices_V4(
                languageDiscreteEmbeddings_RU.Words,
                languageDiscreteEmbeddings_RU.ClusterInfos.ToList(),
                languageDiscreteEmbeddings_RU.Temp_WordsDistancesOldMatrix,
                r,
                _loggersSet);
            Helpers.SerializationHelper.SaveToFile(FileName_ProjectionOptimization_AlgorithmData_RU, languageDiscreteEmbeddings_RU.Temp_ProjectionOptimization_AlgorithmData, null, _loggersSet.UserFriendlyLogger);

            languageDiscreteEmbeddings_EN.Temp_ProjectionOptimization_AlgorithmData = Calculate_ProjectionIndices_V4(
                languageDiscreteEmbeddings_EN.Words,
                languageDiscreteEmbeddings_EN.ClusterInfos.ToList(),
                languageDiscreteEmbeddings_EN.Temp_WordsDistancesOldMatrix,
                r,
                _loggersSet);
            Helpers.SerializationHelper.SaveToFile(FileName_ProjectionOptimization_AlgorithmData_EN, languageDiscreteEmbeddings_EN.Temp_ProjectionOptimization_AlgorithmData, null, _loggersSet.UserFriendlyLogger);            
        }
        else
        {
            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_ProjectionOptimization_AlgorithmData_RU, languageDiscreteEmbeddings_RU.Temp_ProjectionOptimization_AlgorithmData, null, null);
            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_ProjectionOptimization_AlgorithmData_EN, languageDiscreteEmbeddings_EN.Temp_ProjectionOptimization_AlgorithmData, null, null);
        }

        //CalculateDiscreteVectors(Clusterization_AlgorithmData_Random, ProjectionOptimization_AlgorithmData_Random, _loggersSet);
        //SaveToFile_DiscreteVectors(Clusterization_AlgorithmData_Random, _loggersSet);
        //LoadFromFile_DiscreteVectorsAndMatrices(Clusterization_AlgorithmData_Random, _loggersSet);
        //ProxWordsDiscreteMatrix_Calculate(Clusterization_AlgorithmData_Random, _loggersSet);            

        calculate = true;
        if (calculate)
        {
            languageDiscreteEmbeddings_RU.Temp_DiscreteVectorsAndMatrices = Calculate_DiscreteVectorsOnly(
                languageDiscreteEmbeddings_RU.Words,
                languageDiscreteEmbeddings_RU.ClusterInfos,
                languageDiscreteEmbeddings_RU.Temp_WordsDistancesOldMatrix,
                languageDiscreteEmbeddings_RU.Temp_ProjectionOptimization_AlgorithmData,
                _loggersSet);
            Helpers.SerializationHelper.SaveToFile(FileName_DiscreteVectors_RU, languageDiscreteEmbeddings_RU.Temp_DiscreteVectorsAndMatrices, null, _loggersSet.UserFriendlyLogger);

            languageDiscreteEmbeddings_EN.Temp_DiscreteVectorsAndMatrices = Calculate_DiscreteVectorsOnly(
                languageDiscreteEmbeddings_EN.Words,
                languageDiscreteEmbeddings_EN.ClusterInfos,
                languageDiscreteEmbeddings_EN.Temp_WordsDistancesOldMatrix,
                languageDiscreteEmbeddings_EN.Temp_ProjectionOptimization_AlgorithmData,
                _loggersSet);
            Helpers.SerializationHelper.SaveToFile(FileName_DiscreteVectors_EN, languageDiscreteEmbeddings_EN.Temp_DiscreteVectorsAndMatrices, null, _loggersSet.UserFriendlyLogger);
        }
        else
        {
            languageDiscreteEmbeddings_RU.Temp_DiscreteVectorsAndMatrices = new DiscreteVectorsAndMatrices();
            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_DiscreteVectors_RU, languageDiscreteEmbeddings_RU.Temp_DiscreteVectorsAndMatrices, null, null);

            languageDiscreteEmbeddings_EN.Temp_DiscreteVectorsAndMatrices = new DiscreteVectorsAndMatrices();
            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_DiscreteVectors_EN, languageDiscreteEmbeddings_EN.Temp_DiscreteVectorsAndMatrices, null, null);
        }

        for (int i = 0; i < languageDiscreteEmbeddings_RU.Words.Count; i += 1)
        {
            Model03Core.Word word = languageDiscreteEmbeddings_RU.Words[i];

            word.DiscreteVector = languageDiscreteEmbeddings_RU.Temp_DiscreteVectorsAndMatrices.DiscreteVectors[i];
            word.DiscreteVector_PrimaryBitsOnly = languageDiscreteEmbeddings_RU.Temp_DiscreteVectorsAndMatrices.DiscreteVectors_PrimaryBitsOnly[i];
            word.DiscreteVector_SecondaryBitsOnly = languageDiscreteEmbeddings_RU.Temp_DiscreteVectorsAndMatrices.DiscreteVectors_SecondaryBitsOnly[i];            
#if DEBUG
            var primaryBitsSum = TensorPrimitives.Sum(word.DiscreteVector_PrimaryBitsOnly);
            Debug.Assert(primaryBitsSum > Constants.DiscreteVector_PrimaryBitsCount - 0.00001f && primaryBitsSum < Constants.DiscreteVector_PrimaryBitsCount + 0.00001f);
#endif           
        }
        Helpers.SerializationHelper.SaveToFile(FileName_LanguageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_RU, null, _loggersSet.UserFriendlyLogger);

        for (int i = 0; i < languageDiscreteEmbeddings_EN.Words.Count; i += 1)
        {
            Model03Core.Word word = languageDiscreteEmbeddings_EN.Words[i];

            word.DiscreteVector = languageDiscreteEmbeddings_EN.Temp_DiscreteVectorsAndMatrices.DiscreteVectors[i];
            word.DiscreteVector_PrimaryBitsOnly = languageDiscreteEmbeddings_EN.Temp_DiscreteVectorsAndMatrices.DiscreteVectors_PrimaryBitsOnly[i];
            word.DiscreteVector_SecondaryBitsOnly = languageDiscreteEmbeddings_EN.Temp_DiscreteVectorsAndMatrices.DiscreteVectors_SecondaryBitsOnly[i];
#if DEBUG
            var primaryBitsSum = TensorPrimitives.Sum(word.DiscreteVector_PrimaryBitsOnly);
            Debug.Assert(primaryBitsSum > Constants.DiscreteVector_PrimaryBitsCount - 0.00001f && primaryBitsSum < Constants.DiscreteVector_PrimaryBitsCount + 0.00001f);
#endif           
        }
        Helpers.SerializationHelper.SaveToFile(FileName_LanguageDiscreteEmbeddings_EN, languageDiscreteEmbeddings_EN, null, _loggersSet.UserFriendlyLogger);
    }

    /// <summary>
    ///     Для этого для каждого слова брал 10 ближайших соседей в вещественных эмбеддингах и 10 ближайших в дискретных и вычислял пересечение множеств. И вычислял среднее значение отношения <количество общих слов>/10. 
    /// </summary>
    public void GetEmbeddingsQualityInfo1()
    {
        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_RU = new();        
        Helpers.SerializationHelper.LoadFromFileIfExists(FileName_LanguageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_RU, null, null);

        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_EN = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(FileName_LanguageDiscreteEmbeddings_EN, languageDiscreteEmbeddings_RU, null, null);

        var percentage = NeighborStructureComparer.Compare(
            languageDiscreteEmbeddings_RU.Words.Select(w => w.Name).ToArray(),
            languageDiscreteEmbeddings_RU.GetOldEmbeddingsMatrix(),
            languageDiscreteEmbeddings_RU.GetDiscreteEmbeddingsMatrix());

        _loggersSet.UserFriendlyLogger.LogInformation($"RU: {percentage}");
    }

    /// <summary>
    ///     Сравнение с человеческой оценкой
    /// </summary>
    public void GetEmbeddingsQualityInfo2()
    {
        //LanguageDiscreteEmbeddings languageDiscreteEmbeddings_RU = new();
        //Helpers.SerializationHelper.LoadFromFileIfExists(FileName_LanguageDiscreteEmbeddings_RU, languageDiscreteEmbeddings_RU, null);

        LanguageDiscreteEmbeddings languageDiscreteEmbeddings_EN = new();
        Helpers.SerializationHelper.LoadFromFileIfExists(FileName_LanguageDiscreteEmbeddings_EN, languageDiscreteEmbeddings_EN, null, null);

        EmbeddingEvaluator.CompareOnSimilarityDataset(
            Path.Combine("Data", "wordsim353crowd.csv"),
            languageDiscreteEmbeddings_EN.Words.ToDictionary(w => w.Name, w => w.OldVector, StringComparer.InvariantCultureIgnoreCase),
            languageDiscreteEmbeddings_EN.Words.ToDictionary(w => w.Name, w => w.DiscreteVector, StringComparer.InvariantCultureIgnoreCase),
            languageDiscreteEmbeddings_EN.Words[0].OldVector.Length,
            languageDiscreteEmbeddings_EN.Words[0].DiscreteVector.Length,
            _loggersSet);

        _loggersSet.UserFriendlyLogger.LogInformation($"GetEmbeddingsQualityInfo2() Done.");
    }

    #endregion

    #region private functions    

    private MatrixFloat WordsDistancesOldMatrix_Calculate(List<Word> words, ILoggersSet loggersSet)
    {
        var stopwatch = Stopwatch.StartNew();
        
        int wordsCount = words.Count;

        var wordsDistancesOldMatrix = new MatrixFloat(wordsCount, wordsCount);

        Parallel.For(0, wordsCount, index1 =>
        {
            var oldVectrorNormalized = words[index1].OldVectorNormalized;
            for (var index2 = 0; index2 < wordsCount; index2 += 1)
            {
                if (index2 != index1)
                    wordsDistancesOldMatrix[index1, index2] = ModelHelper.GetEnergy(oldVectrorNormalized, words[index2].OldVectorNormalized);
                else
                    wordsDistancesOldMatrix[index1, index2] = 0.0f;
            }
        });        

        stopwatch.Stop();
        loggersSet.UserFriendlyLogger.LogInformation("WordsDistancesOldMatrix_Calculate done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);

        return wordsDistancesOldMatrix;
    }

    //private void SaveToCsvFile_WordsNewEmbeddings(WordsNewEmbeddings wordsNewEmbeddings, string fileName, ILoggersSet loggersSet)
    //{
    //    var totalStopwatch = Stopwatch.StartNew();

    //    string programDataDirectoryFullName = Directory.GetCurrentDirectory();

    //    List<List<string?>> fileData = new();

    //    foreach (var kvp in wordsNewEmbeddings.Words) 
    //    {
    //        fileData.Add(new List<string?> { kvp.Key });
    //    }

    //    CsvHelper.SaveCsvFile(Path.Combine(programDataDirectoryFullName, fileName), fileData);

    //    totalStopwatch.Stop();
    //    loggersSet.UserFriendlyLogger.LogInformation($"{nameof(SaveToCsvFile_WordsNewEmbeddings)} done. Elapsed Milliseconds: {totalStopwatch.ElapsedMilliseconds}");
    //}                  

    #endregion

    #region private fields
   
    private readonly ILoggersSet _loggersSet;                

    //private Clusterization_AlgorithmDataEnum _primaryWordsSelectionMethod;
    private readonly float[] _v1 = new float[2];

    #endregion

    public class ModelConstants
    {
        public int OldVectorLength { get; } = 300;

        public int DiscreteVectorLength { get; } = 300;

        /// <summary>
        ///     For algorithmDatas with fixed primary words count.
        /// </summary>
        public int ClustersCount { get; } = 300;

        public int DiscreteVector_SecondaryBitsCount { get; } = 7;

        public int DiscreteVector_PrimaryBitsCount { get; } = 7;
    }
}


///// <summary>
/////     
///// </summary>
///// <param name="discreteVectorsAndMatrices"></param>
///// <param name="fileName"></param>
///// <param name="loggersSet"></param>
//private void LoadFromFile_DiscreteVectorsAndMatrices(DiscreteVectorsAndMatrices discreteVectorsAndMatrices, string fileName, ILoggersSet loggersSet)
//{
//    var stopwatch = Stopwatch.StartNew();

//    string programDataDirectoryFullName = Directory.GetCurrentDirectory();
//    byte[] bytes = File.ReadAllBytes(Path.Combine(programDataDirectoryFullName, fileName));
//    using (SerializationReader reader = new(bytes))
//    {
//        int discreteVectorsLength = reader.ReadInt32();
//        if (discreteVectorsLength > 0)
//        {
//            var discreteVectors = new float[discreteVectorsLength][];
//            foreach (int i in Enumerable.Range(0, discreteVectorsLength))
//            {
//                discreteVectors[i] = reader.ReadArray<float>()!;
//            }
//            discreteVectorsAndMatrices.DiscreteVectors = discreteVectors;
//        }
//        //algorithmData.ProxWordsDiscreteMatrix = reader.ReadArray<float>();

//        discreteVectorsLength = reader.ReadInt32();
//        if (discreteVectorsLength > 0)
//        {
//            var discreteVectors_PrimaryOnly = new float[discreteVectorsLength][];
//            foreach (int i in Enumerable.Range(0, discreteVectorsLength))
//            {
//                discreteVectors_PrimaryOnly[i] = reader.ReadArray<float>()!;
//            }
//            discreteVectorsAndMatrices.DiscreteVectors_PrimaryOnly = discreteVectors_PrimaryOnly;
//        }
//        //algorithmData.ProxWordsDiscreteMatrix_PrimaryOnly = reader.ReadArray<float>();

//        discreteVectorsLength = reader.ReadInt32();
//        if (discreteVectorsLength > 0)
//        {
//            var discreteVectors_SecondaryOnly = new float[discreteVectorsLength][];
//            foreach (int i in Enumerable.Range(0, discreteVectorsLength))
//            {
//                discreteVectors_SecondaryOnly[i] = reader.ReadArray<float>()!;
//            }
//            discreteVectorsAndMatrices.DiscreteVectors_SecondaryOnly = discreteVectors_SecondaryOnly;
//        }
//        //algorithmData.ProxWordsDiscreteMatrix_SecondaryOnly = reader.ReadArray<float>();
//    }

//    stopwatch.Stop();
//    loggersSet.UserFriendlyLogger.LogInformation("LoadFromFile_DiscreteVectorsAndMatrices done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
//}

///// <summary>
/////     Primary Words Selection AlgorithmData Enum
///// </summary>
//public enum Clusterization_AlgorithmDataEnum
//{
//    None = 0,
//    Random,
//    AlgorithmData_Em,
//    AlgorithmData_KMeans,
//    AlgorithmData_Classes,
//}

//public Clusterization_AlgorithmDataEnum PrimaryWordsSelectionMethod
//{
//    get
//    {
//        return _primaryWordsSelectionMethod;
//    }
//    set
//    {
//        _primaryWordsSelectionMethod = value;

//        //switch (_primaryWordsSelectionMethod)
//        //{
//        //    case Clusterization_AlgorithmDataEnum.Random:
//        //        CurrentClusterization_AlgorithmData_ToDisplay = Clusterization_AlgorithmData_Random;
//        //        break;
//        //    case Clusterization_AlgorithmDataEnum.AlgorithmData_Em:
//        //        CurrentClusterization_AlgorithmData_ToDisplay = Clusterization_AlgorithmData_Em;
//        //        break;
//        //    case Clusterization_AlgorithmDataEnum.AlgorithmData_KMeans:
//        //        CurrentClusterization_AlgorithmData_ToDisplay = Clusterization_AlgorithmData_KMeans;
//        //        break;
//        //    case Clusterization_AlgorithmDataEnum.AlgorithmData_Classes:
//        //        CurrentClusterization_AlgorithmData_ToDisplay = Clusterization_AlgorithmData_Classes;
//        //        break;
//        //    default:
//        //        CurrentClusterization_AlgorithmData_ToDisplay = null;
//        //        break;
//        //}
//    }
//}

//public enum ProjectionOptimization_AlgorithmDataEnum
//{
//    None = 0,
//    Random,
//    Variant3,
//}

//public readonly Clusterization_AlgorithmData Clusterization_AlgorithmData_Em = new Clusterization_AlgorithmData { Name = "Em" };

//public class ProxWords
//{
//    public ProxWords(List<Word> words)
//    {
//        Array = new float[(words.Count - 1) * (words.Count - 1)];
//    }

//    public readonly float[] Array;

//    public ref float this[int id1, int id2]
//    {
//        get { return ref Array[ix + iy * XCount]; }
//    }
//}

//foreach (var point2 in Cortex.Array)
//{
//    if (point2 is null || point2.Id == point.Id)
//        continue;

//    var proxWord = ProxWords[point2.Id + idBias];
//    if (proxWord > 0)
//    {
//        _v2[0] = point2.iX;
//        _v2[1] = point2.iY;
//        double r = Math.Sqrt(Math.Pow(ix - point2.iX, 2) + Math.Pow(iy - point2.iY, 2));

//        pointEnergy += proxWord * r;
//    }
//}

//var primaryWords = new Word[list.Count];
//for (int index = 0; index < list.Count; index += 1)
//{
//    var word = Words[list[index]];
//    primaryWords[index] = word;
//}

//PrimaryWords = primaryWords;