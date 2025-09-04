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
    ///     RusVectores        
    /// </summary>
    public readonly LanguageInfo LanguageInfo_RU = new();

    /// <summary>
    ///     GloVe (Stanford)        
    /// </summary>
    public readonly LanguageInfo LanguageInfo_EN = new();

    public Cortex Cortex = null!;

    /// <summary>
    ///     For display.
    ///     Lock CortexCopySyncRoot when read/write.
    /// </summary>
    public Cortex CortexCopy = new (0, 0);

    public readonly object CortexCopySyncRoot = new object();             
    
    public ProjectionOptimization_Algorithm? CurrentProjectionOptimization_Algorithm_ToDisplay;

    public DiscreteVectorsAndMatrices? CurrentDiscreteVectorsAndMatrices_ToDisplay;

    public WordsNewEmbeddings? CurrentWordsNewEmbeddings;    

    public void Initialize()
    {
        Task.Run(async () =>
        {
            #region RU Words Initialization
            WordsHelper.InitializeWords_RU(LanguageInfo_RU, _loggersSet);
            LanguageInfo_RU.Clusterization_Algorithm = new Clusterization_Algorithm(LanguageInfo_RU.Words) { Name = "KMeans" };
            #endregion

            #region EN Words Initialization
            WordsHelper.InitializeWords_EN(LanguageInfo_EN, _loggersSet);
            LanguageInfo_EN.Clusterization_Algorithm = new Clusterization_Algorithm(LanguageInfo_EN.Words) { Name = "KMeans" };
            #endregion

            ProxWordsOldMatrix_Calculate(LanguageInfo_RU, _loggersSet);
            string fileName = "AdvancedEmbedding_LanguageInfo_ProxWordsOldMatrix_RU.bin";
            Helpers.SerializationHelper.SaveToFile(fileName, LanguageInfo_RU.ProxWordsOldMatrix, null);
            //Helpers.SerializationHelper.LoadFromFileIfExists(fileName, LanguageInfo_RU.ProxWordsOldMatrix, null);

            ProxWordsOldMatrix_Calculate(LanguageInfo_EN, _loggersSet);
            fileName = "AdvancedEmbedding_LanguageInfo_ProxWordsOldMatrix_EN.bin";
            Helpers.SerializationHelper.SaveToFile(fileName, LanguageInfo_EN.ProxWordsOldMatrix, null);
            //Helpers.SerializationHelper.LoadFromFileIfExists(fileName, LanguageInfo_EN.ProxWordsOldMatrix, null);

            //Calculate_Clusterization_Algorithm_Random(_loggersSet);

            Calculate_Clusterization_Algorithm_KMeans(LanguageInfo_RU, _loggersSet);
            fileName = "AdvancedEmbedding_Clusterization_Algorithm_KMeans_RU.bin";
            Helpers.SerializationHelper.SaveToFile(fileName, LanguageInfo_RU.Clusterization_Algorithm, null);
            //Helpers.SerializationHelper.LoadFromFileIfExists(fileName, LanguageInfo_RU.Clusterization_Algorithm_KMeans, null);                

            Calculate_Clusterization_Algorithm_KMeans(LanguageInfo_EN, _loggersSet);
            fileName = "AdvancedEmbedding_Clusterization_Algorithm_KMeans_EN.bin";
            Helpers.SerializationHelper.SaveToFile(fileName, LanguageInfo_EN.Clusterization_Algorithm, null);
            //Helpers.SerializationHelper.LoadFromFileIfExists(fileName, LanguageInfo_EN.Clusterization_Algorithm_KMeans, null);    
                            
            Calculate_ProjectionIndices_Variant3(LanguageInfo_RU, _loggersSet);
            fileName = "AdvancedEmbedding_ProjectionOptimization_Algorithm_Variant3_RU.bin";
            Helpers.SerializationHelper.SaveToFile(fileName, LanguageInfo_RU.ProjectionOptimization_Algorithm, null);
            //Helpers.SerializationHelper.LoadFromFileIfExists(fileName, LanguageInfo_RU.ProjectionOptimization_Algorithm_Variant3, null);                                

            Calculate_ProjectionIndices_Variant3(LanguageInfo_EN, _loggersSet);
            fileName = "AdvancedEmbedding_ProjectionOptimization_Algorithm_Variant3_EN.bin";
            Helpers.SerializationHelper.SaveToFile(fileName, LanguageInfo_EN.ProjectionOptimization_Algorithm, null);
            //Helpers.SerializationHelper.LoadFromFileIfExists(fileName, LanguageInfo_EN.ProjectionOptimization_Algorithm_Variant3, null);    

            //CalculateDiscreteVectors(Clusterization_Algorithm_Random, ProjectionOptimization_Algorithm_Random, _loggersSet);
            //SaveToFile_DiscreteVectors(Clusterization_Algorithm_Random, _loggersSet);
            //LoadFromFile_DiscreteVectorsAndMatrices(Clusterization_Algorithm_Random, _loggersSet);
            //ProxWordsNewMatrix_Calculate(Clusterization_Algorithm_Random, _loggersSet);

            DiscreteVectorsAndMatrices discreteVectorsAndMatrices = Calculate_DiscreteVectors(LanguageInfo_RU, _loggersSet);
            fileName = "AdvancedEmbedding_DiscreteVectors_RU.bin";
            Helpers.SerializationHelper.SaveToFile(fileName, discreteVectorsAndMatrices, null);
            //Helpers.SerializationHelper.LoadFromFileIfExists(fileName, discreteVectorsAndMatrices, null);    

            discreteVectorsAndMatrices = Calculate_DiscreteVectors(LanguageInfo_EN, _loggersSet);
            fileName = "AdvancedEmbedding_DiscreteVectors_EN.bin";
            Helpers.SerializationHelper.SaveToFile(fileName, discreteVectorsAndMatrices, null);
            //Helpers.SerializationHelper.LoadFromFileIfExists(fileName, discreteVectorsAndMatrices, null);  
            
            CurrentDiscreteVectorsAndMatrices_ToDisplay = discreteVectorsAndMatrices;
            //CurrentClusterization_Algorithm_ToDisplay = Clusterization_Algorithm_KMeans;
            //CurrentProjectionOptimization_Algorithm_ToDisplay = ProjectionOptimization_Algorithm_Variant3;
            
            //LoadFromFile_DiscreteVectorsAndMatrices(Clusterization_Algorithm_KMeans, _loggersSet);
            //ProxWordsNewMatrix_Calculate(Clusterization_Algorithm_KMeans, _loggersSet);

            //DiscreteVectorsAndMatrices discreteVectorsAndMatrices = Calculate_DiscreteVectors(Clusterization_Algorithm_Classes, ProjectionOptimization_Algorithm_Random, _loggersSet);
            //CurrentDiscreteVectorsAndMatrices_ToDisplay = discreteVectorsAndMatrices;
            //SaveToFile_DiscreteVectorsAndMatrices(Algorithm_Classes, _loggersSet);
            //LoadFromFile_DiscreteVectorsAndMatrices(Clusterization_Algorithm_Classes, _loggersSet);
            //ProxWordsNewMatrix_Calculate(Clusterization_Algorithm_Classes, _loggersSet);

            //CurrentWordsNewEmbeddings = Calculate_WordsNewEmbeddings(_loggersSet);
            //SaveToFile_WordsNewEmbeddings(CurrentWordsNewEmbeddings, "NewWordsEmbeddings.csv", _loggersSet);

            //CompareOldAndNewPhraseEmbeddings(_loggersSet);
        });            
    }        

    public void Close()
    {            
    }            

    #endregion

    #region private functions
    
    private void CreateCortexCopy()
    {
        lock (CortexCopySyncRoot)
        {
            for (int i = 0; i < Cortex.Array.Length; i += 1)
            {
                CortexCopy.Array[i].CopyData(Cortex.Array[i]);
            }
        }
    }

    private void ProxWordsOldMatrix_Calculate(LanguageInfo languageInfo, ILoggersSet loggersSet)
    {
        var stopwatch = Stopwatch.StartNew();

        var words = languageInfo.Words;
        int wordsCount = words.Count;

        var proxWordsOldMatrix = new MatrixFloat(wordsCount, wordsCount);
        Parallel.For(0, wordsCount, index1 =>
        {
            int indexBias = index1 * wordsCount;
            var oldVectror = words[index1].OldVectorNormalized;
            for (var index2 = 0; index2 < wordsCount; index2 += 1)
            {
                if (index2 != index1)
                    proxWordsOldMatrix.Data[indexBias + index2] = TensorPrimitives.Dot(oldVectror, words[index2].OldVectorNormalized);
                else
                    proxWordsOldMatrix.Data[indexBias + index2] = 1.0f;
            }
        });
        languageInfo.ProxWordsOldMatrix = proxWordsOldMatrix;

        stopwatch.Stop();
        loggersSet.UserFriendlyLogger.LogInformation("ProxWordsMatrixCalculate done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);   
    }

    private void SaveToFile_WordsNewEmbeddings(WordsNewEmbeddings wordsNewEmbeddings, string fileName, ILoggersSet loggersSet)
    {
        var totalStopwatch = Stopwatch.StartNew();

        string programDataDirectoryFullName = Directory.GetCurrentDirectory();

        List<List<string?>> fileData = new();

        foreach (var kvp in wordsNewEmbeddings.Words) 
        {
            fileData.Add(new List<string?> { kvp.Key });
        }

        CsvHelper.SaveCsvFile(Path.Combine(programDataDirectoryFullName, fileName), fileData);

        totalStopwatch.Stop();
        loggersSet.UserFriendlyLogger.LogInformation($"{nameof(SaveToFile_WordsNewEmbeddings)} done. Elapsed Milliseconds: {totalStopwatch.ElapsedMilliseconds}");
    }        

    private void CortexSaveToFile(ILoggersSet loggersSet)
    {
        string programDataDirectoryFullName = Directory.GetCurrentDirectory();

        using (MemoryStream memoryStream = new())
        using (SerializationWriter serializationWriter = new(memoryStream))
        {
            Cortex.SerializeOwnedData(serializationWriter, null);
            byte[] bytes = memoryStream.ToArray();
            File.WriteAllBytes(Path.Combine(programDataDirectoryFullName, "Cortex.bin"), bytes);
        }
    }

    private void CortexLoadFromFile(LanguageInfo languageInfo, ILoggersSet loggersSet)
    {
        var stopwatch = Stopwatch.StartNew();

        string programDataDirectoryFullName = Directory.GetCurrentDirectory();
        byte[] bytes = File.ReadAllBytes(Path.Combine(programDataDirectoryFullName, "Cortex.bin"));
        using (SerializationReader serializationReader = new(bytes))
        {
            Cortex = new Cortex();
            Cortex.DeserializeOwnedData(serializationReader, null);
        }

        int ix, iy;
        for (ix = 0; ix < Cortex.XCount; ix += 1)
        {
            for (iy = 0; iy < Cortex.YCount; iy += 1)
            {
                ref var pointRef = ref Cortex[ix, iy];
                if (pointRef.WordIndex >= 0)
                    languageInfo.Words[pointRef.WordIndex].Point = pointRef;
            }
        }

        #region CortexCopy Initialization

        lock (CortexCopySyncRoot)
        {
            CortexCopy = new Cortex(Cortex.XCount, Cortex.YCount);
            for (int i = 0; i < Cortex.Array.Length; i += 1)
            {
                Point point = new()
                {
                    V = new float[2]
                };
                point.CopyData(Cortex.Array[i]);
                CortexCopy.Array[i] = point;
            }
        }

        #endregion

        stopwatch.Stop();
        loggersSet.UserFriendlyLogger.LogInformation("CortexLoadFromFile done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
    }        

    #endregion

    #region private fields
   
    private readonly ILoggersSet _loggersSet;                

    //private Clusterization_AlgorithmEnum _primaryWordsSelectionMethod;
    private readonly float[] _v1 = new float[2];

    #endregion

    public class ModelConstants
    {
        public int OldVectorLength { get; } = 300;

        public int DiscreteVectorLength { get; } = 200;

        /// <summary>
        ///     For algorithms with fixed primary words count.
        /// </summary>
        public int PrimaryWordsCount { get; } = 300;

        public int PrimaryWords_DiscreteVector_BitsCount { get; } = 8;

        public int SecondaryWords_DiscreteVector_BitsCount { get; } = 8;
    }
}

public class Word
{
    public Word()
    {
        OldVector = new float[Model01.Constants.OldVectorLength];
        OldVectorNormalized = new float[Model01.Constants.OldVectorLength];
        DiscreteVector_ToDisplay = new float[Model01.Constants.DiscreteVectorLength];
    }

    /// <summary>
    ///     Index in Words Array.
    ///     Index == 0: Empty word
    /// </summary>
    public int Index;

    public string Name = null!;

    public double Freq;

    /// <summary>
    ///     Initialized when Cortex is initialized.
    /// </summary>
    public Point Point = null!;

    /// <summary>
    ///     Original normalized vector (module 1).
    /// </summary>
    public readonly float[] OldVector;

    /// <summary>
    ///     Original normalized vector (module 1).
    /// </summary>
    public readonly float[] OldVectorNormalized;

    public float[]? DiscreteVector_ToDisplay;        

    public bool Temp_Flag;        
}    

public class Cortex : IOwnedDataSerializable
{
    public Cortex()
    {
    }

    public Cortex(int xCount, int yCount)
    {
        XCount = xCount;
        YCount = yCount;            
        Array = new Point[xCount * yCount];
    }

    public int XCount;

    public int YCount;

    public Point[] Array = null!;

    public ref Point this[int ix, int iy]
    {
        get { return ref Array[iy * XCount + ix]; }            
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.Write(XCount);
        writer.Write(YCount);
        writer.Write(Array.Length);
        for (int i = 0; i < Array.Length; i += 1)
        {
            var point = Array[i];
            writer.WriteOptimized(point.WordIndex);
            writer.WriteOptimized(point.GroupId_ToDisplay);                
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        XCount = reader.ReadInt32();
        YCount = reader.ReadInt32();
        int arrayLength = reader.ReadInt32();
        Array = new Point[arrayLength];
        int ix = 0;
        int iy = 0;
        for (int i = 0; i < arrayLength; i += 1)
        {                
            Point point = new Point();
            Array[i] = point;
            point.WordIndex = reader.ReadOptimizedInt32();                
            point.GroupId_ToDisplay = reader.ReadOptimizedInt32();                
            point.V = new float[2];
            point.V[0] = ix;
            point.V[1] = iy;
            ix += 1;
            if (ix == XCount)
            {
                ix = 0;
                iy += 1;
            }
        }
    }
}

public class Point
{
    /// <summary>
    ///     Index in Words
    /// </summary>
    public int WordIndex;

    public int GroupId_ToDisplay = (int)PointGroupId_ToDisplay.None;        

    /// <summary>
    ///     |iX, iY| vector
    /// </summary>
    /// <remarks>Otimized for calculation</remarks>
    public float[] V = null!;        

    /// <summary>
    ///     Top N point refs (ordered by proximity)
    ///     (Proximity, Point)
    /// </summary>
    public (float, Point)[]? Temp_TopProxPoints;

    /// <summary>
    ///     Top N primary point refs (ordered by proximity)
    ///     (Proximity, Point)
    ///     Not null only for primary words.
    /// </summary>
    public (float, Point)[]? Temp_TopProxPrimaryPoints;

    public void CopyData(Point that)
    {
        WordIndex = that.WordIndex;
        GroupId_ToDisplay = that.GroupId_ToDisplay;            
        V[0] = that.V[0];
        V[1] = that.V[1];
        Temp_TopProxPoints = that.Temp_TopProxPoints;
        Temp_TopProxPrimaryPoints = that.Temp_TopProxPrimaryPoints;
    }
}

public class WordCluster
{        
    public float[] CentroidOldVector = null!;

    public int PrimaryWordIndex;

    public int WordsCount;        
}

public enum CortexDisplayType
{
    GroupId_ToDisplay = 0,
    Spot,        
}

public enum DotProductVariant
{
    All = 0,
    PrimaryOnly,
    SecondaryOnly,
}

public enum PointGroupId_ToDisplay
{
    None = 0,
    // 1-9 reserved for different colored groups
    PrimaryPoint = 10,
    MainPoint1 = 12,
    PrimaryPoint_Selected1 = 13,
    SecondaryPoint_Selected1 = 14,
    PrimaryAndSecondaryPoint_Selected1 = 15,
    //MainPoint2 = 15,
    //PrimaryPoint_Selected2 = 16,
    //SecondaryPoint_Selected2 = 17,
}

public class WordsNewEmbeddings
{
    /// <summary>
    ///     [Словоформа, DiscreteVector Index]
    /// </summary>
    public CaseInsensitiveDictionary<int> Words = new();
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
//    using (SerializationReader serializationReader = new(bytes))
//    {
//        int discreteVectorsLength = serializationReader.ReadInt32();
//        if (discreteVectorsLength > 0)
//        {
//            var discreteVectors = new float[discreteVectorsLength][];
//            foreach (int i in Enumerable.Range(0, discreteVectorsLength))
//            {
//                discreteVectors[i] = serializationReader.ReadArray<float>()!;
//            }
//            discreteVectorsAndMatrices.DiscreteVectors = discreteVectors;
//        }
//        //algorithm.ProxWordsNewMatrix = serializationReader.ReadArray<float>();

//        discreteVectorsLength = serializationReader.ReadInt32();
//        if (discreteVectorsLength > 0)
//        {
//            var discreteVectors_PrimaryOnly = new float[discreteVectorsLength][];
//            foreach (int i in Enumerable.Range(0, discreteVectorsLength))
//            {
//                discreteVectors_PrimaryOnly[i] = serializationReader.ReadArray<float>()!;
//            }
//            discreteVectorsAndMatrices.DiscreteVectors_PrimaryOnly = discreteVectors_PrimaryOnly;
//        }
//        //algorithm.ProxWordsNewMatrix_PrimaryOnly = serializationReader.ReadArray<float>();

//        discreteVectorsLength = serializationReader.ReadInt32();
//        if (discreteVectorsLength > 0)
//        {
//            var discreteVectors_SecondaryOnly = new float[discreteVectorsLength][];
//            foreach (int i in Enumerable.Range(0, discreteVectorsLength))
//            {
//                discreteVectors_SecondaryOnly[i] = serializationReader.ReadArray<float>()!;
//            }
//            discreteVectorsAndMatrices.DiscreteVectors_SecondaryOnly = discreteVectors_SecondaryOnly;
//        }
//        //algorithm.ProxWordsNewMatrix_SecondaryOnly = serializationReader.ReadArray<float>();
//    }

//    stopwatch.Stop();
//    loggersSet.UserFriendlyLogger.LogInformation("LoadFromFile_DiscreteVectorsAndMatrices done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
//}

///// <summary>
/////     Primary Words Selection Algorithm Enum
///// </summary>
//public enum Clusterization_AlgorithmEnum
//{
//    None = 0,
//    Random,
//    Algorithm_Em,
//    Algorithm_KMeans,
//    Algorithm_Classes,
//}

//public Clusterization_AlgorithmEnum PrimaryWordsSelectionMethod
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
//        //    case Clusterization_AlgorithmEnum.Random:
//        //        CurrentClusterization_Algorithm_ToDisplay = Clusterization_Algorithm_Random;
//        //        break;
//        //    case Clusterization_AlgorithmEnum.Algorithm_Em:
//        //        CurrentClusterization_Algorithm_ToDisplay = Clusterization_Algorithm_Em;
//        //        break;
//        //    case Clusterization_AlgorithmEnum.Algorithm_KMeans:
//        //        CurrentClusterization_Algorithm_ToDisplay = Clusterization_Algorithm_KMeans;
//        //        break;
//        //    case Clusterization_AlgorithmEnum.Algorithm_Classes:
//        //        CurrentClusterization_Algorithm_ToDisplay = Clusterization_Algorithm_Classes;
//        //        break;
//        //    default:
//        //        CurrentClusterization_Algorithm_ToDisplay = null;
//        //        break;
//        //}
//    }
//}

//public enum ProjectionOptimization_AlgorithmEnum
//{
//    None = 0,
//    Random,
//    Variant3,
//}

//public readonly Clusterization_Algorithm Clusterization_Algorithm_Em = new Clusterization_Algorithm { Name = "Em" };

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