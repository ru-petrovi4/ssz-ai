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
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public partial class Model01
{
    public const string FileName_Mapping = "Mapping.bin";    

    public void FindDiscreteEmbeddings_Mapping()
    {
        WordsHelper.InitializeWords_RU(LanguageInfo_RU, _loggersSet, loadOldVectors: false);
        WordsHelper.InitializeWords_EN(LanguageInfo_EN, _loggersSet, loadOldVectors: false);

        LanguageInfo_RU.Clusterization_AlgorithmData = new Clusterization_AlgorithmData(LanguageInfo_RU, name: "KMeans");
        Helpers.SerializationHelper.LoadFromFileIfExists(FileName_Clusterization_AlgorithmData_KMeans_RU, LanguageInfo_RU.Clusterization_AlgorithmData, null);
        LanguageInfo_EN.Clusterization_AlgorithmData = new Clusterization_AlgorithmData(LanguageInfo_EN, name: "KMeans");
        Helpers.SerializationHelper.LoadFromFileIfExists(FileName_Clusterization_AlgorithmData_KMeans_EN, LanguageInfo_EN.Clusterization_AlgorithmData, null);

        LanguageInfo_RU.ProjectionOptimization_AlgorithmData = new ProjectionOptimization_AlgorithmData(name: "Variant3");
        Helpers.SerializationHelper.LoadFromFileIfExists(FileName_ProjectionOptimization_AlgorithmData_Variant3_RU, LanguageInfo_RU.ProjectionOptimization_AlgorithmData, null);
        LanguageInfo_EN.ProjectionOptimization_AlgorithmData = new ProjectionOptimization_AlgorithmData(name: "Variant3");
        Helpers.SerializationHelper.LoadFromFileIfExists(FileName_ProjectionOptimization_AlgorithmData_Variant3_EN, LanguageInfo_EN.ProjectionOptimization_AlgorithmData, null);

        LanguageInfo_RU.DiscreteVectorsAndMatrices = new DiscreteVectorsAndMatrices();
        Helpers.SerializationHelper.LoadFromFileIfExists(FileName_DiscreteVectors_RU, LanguageInfo_RU.DiscreteVectorsAndMatrices, null);
        LanguageInfo_EN.DiscreteVectorsAndMatrices = new DiscreteVectorsAndMatrices();
        Helpers.SerializationHelper.LoadFromFileIfExists(FileName_DiscreteVectors_EN, LanguageInfo_EN.DiscreteVectorsAndMatrices, null);

        MappingData mappingData = new(_loggersSet, LanguageInfo_RU, LanguageInfo_EN);
        bool calculate = true;
        if (calculate)
        {  
            mappingData.GenerateOwnedData(Constants.DiscreteVectorLength);
            mappingData.Prepare();
            mappingData.CalculateMapping();
            Helpers.SerializationHelper.SaveToFile(FileName_Mapping, mappingData, null);
        }
        else
        {
            Helpers.SerializationHelper.LoadFromFileIfExists(FileName_Mapping, mappingData, null);
            mappingData.Prepare();
            mappingData.CalculateMapping();
        }   
    }
}