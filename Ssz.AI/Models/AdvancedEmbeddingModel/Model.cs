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

namespace Ssz.AI.Models.AdvancedEmbeddingModel
{    
    public partial class Model
    {
        #region construction and destruction

        public Model()
        {
            _loggersSet = new LoggersSet(NullLogger.Instance, new UserFriendlyLogger((l, id, s) => DebugWindow.Instance.AddLine(s)));
        }            

        #endregion

        #region public functions

        public const int OldVectorLength = 300;

        public const int NewVectorLength = 200;

        /// <summary>
        ///     For algorithms with fixed primary words count.
        /// </summary>
        public int PrimaryWordsCount = 300;

        public int PrimaryWords_NewVector_BitsCount = 8;

        public int SecondaryWords_NewVector_BitsCount = 8;

        /// <summary>
        ///     RusVectores
        ///     <para>Ordered Descending by Freq</para>      
        /// </summary>
        public List<Word> Words_RU = null!;

        /// <summary>
        ///     GloVe (Stanford)
        ///     <para>Ordered Descending by Freq</para>        
        /// </summary>
        public List<Word> Words_EN = null!;

        public Cortex Cortex = null!;

        /// <summary>
        ///     For display.
        ///     Lock CortexCopySyncRoot when read/write.
        /// </summary>
        public Cortex CortexCopy = new (0, 0);

        public readonly object CortexCopySyncRoot = new object();

        /// <summary>
        ///     [WordIndex1, WordIndex2] Words correlation matrix.
        /// </summary>
        /// <remarks>
        ///     Normalized vectors scalar product.
        ///     int indexBias = word1.Index * Words.Count;
        ///     index = indexBias + word2.Index
        /// </remarks>
        public float[] ProxWordsOldMatrix = null!;

        public readonly Clusterization_Algorithm Clusterization_Algorithm_Random = new Clusterization_Algorithm { Name = "Random" };
        public readonly Clusterization_Algorithm Clusterization_Algorithm_Em = new Clusterization_Algorithm { Name = "Em" };
        public readonly Clusterization_Algorithm Clusterization_Algorithm_KMeans = new Clusterization_Algorithm { Name = "KMeans" };
        public readonly Clusterization_Algorithm Clusterization_Algorithm_Classes = new Clusterization_Algorithm { Name = "Classes" };
        public Clusterization_Algorithm? CurrentClusterization_Algorithm_ToDisplay;

        public readonly ProjectionOptimization_Algorithm ProjectionOptimization_Algorithm_Random = new ProjectionOptimization_Algorithm { Name = "Random" };        
        public readonly ProjectionOptimization_Algorithm ProjectionOptimization_Algorithm_Variant3 = new ProjectionOptimization_Algorithm { Name = "Variant3" };
        public ProjectionOptimization_Algorithm? CurrentProjectionOptimization_Algorithm_ToDisplay;

        public NewVectorsAndMatrices? CurrentNewVectorsAndMatrices_ToDisplay;

        public WordsNewEmbeddings? CurrentWordsNewEmbeddings;

        public Clusterization_AlgorithmEnum PrimaryWordsSelectionMethod
        {
            get
            {
                return _primaryWordsSelectionMethod;
            }
            set
            {
                _primaryWordsSelectionMethod = value;

                //switch (_primaryWordsSelectionMethod)
                //{
                //    case Clusterization_AlgorithmEnum.Random:
                //        CurrentClusterization_Algorithm_ToDisplay = Clusterization_Algorithm_Random;
                //        break;
                //    case Clusterization_AlgorithmEnum.Algorithm_Em:
                //        CurrentClusterization_Algorithm_ToDisplay = Clusterization_Algorithm_Em;
                //        break;
                //    case Clusterization_AlgorithmEnum.Algorithm_KMeans:
                //        CurrentClusterization_Algorithm_ToDisplay = Clusterization_Algorithm_KMeans;
                //        break;
                //    case Clusterization_AlgorithmEnum.Algorithm_Classes:
                //        CurrentClusterization_Algorithm_ToDisplay = Clusterization_Algorithm_Classes;
                //        break;
                //    default:
                //        CurrentClusterization_Algorithm_ToDisplay = null;
                //        break;
                //}
            }
        }

        public void Initialize()
        {
            Task.Run(async () =>
            {
                InitializeWords(_loggersSet);

                //ProxWordsOldMatrix_Calculate(_loggersSet);
                //ProxWordsOldMatrix_SaveToFile(_loggersSet);
                ProxWordsOldMatrix_LoadFromFile(_loggersSet);

                //Calculate_Clusterization_Algorithm_Random(_loggersSet);

                //Calculate_Clusterization_Algorithm_KMeans(_loggersSet);
                //Clusterization_Algorithm_SaveToFile(Clusterization_Algorithm_KMeans, _loggersSet);
                Clusterization_Algorithm_LoadFromFile(Clusterization_Algorithm_KMeans, _loggersSet);

                //Calculate_Clusterization_Algorithm_Classes(_loggersSet);
                //Clusterization_Algorithm_SaveToFile(Clusterization_Algorithm_Classes, _loggersSet);
                //Clusterization_Algorithm_LoadFromFile(Clusterization_Algorithm_Classes, _loggersSet);                

                //OptimizeCortex2(_loggersSet, Clusterization_Algorithm_KMeans);
                //CortexSaveToFile(_loggersSet);
                //CortexLoadFromFile(_loggersSet);                

                //Calculate_ProjectionIndices_Random(_loggersSet);

                //Calculate_ProjectionIndices_Variant3(Clusterization_Algorithm_KMeans, _loggersSet);
                //SaveToFile_ProjectionIndices(ProjectionOptimization_Algorithm_Variant3, "ProjectionOptimization.bin", _loggersSet);
                LoadFromFile_ProjectionIndices(ProjectionOptimization_Algorithm_Variant3, "ProjectionOptimization.bin", _loggersSet);

                //CalculateNewVectors(Clusterization_Algorithm_Random, ProjectionOptimization_Algorithm_Random, _loggersSet);
                //SaveToFile_NewVectors(Clusterization_Algorithm_Random, _loggersSet);
                //LoadFromFile_NewVectorsAndMatrices(Clusterization_Algorithm_Random, _loggersSet);
                //ProxWordsNewMatrix_Calculate(Clusterization_Algorithm_Random, _loggersSet);

                //NewVectorsAndMatrices newVectorsAndMatrices = Calculate_NewVectors(Clusterization_Algorithm_KMeans, ProjectionOptimization_Algorithm_Variant3, _loggersSet);
                //SaveToFile_NewVectors(newVectorsAndMatrices, "NewVectors.bin", _loggersSet);
                NewVectorsAndMatrices newVectorsAndMatrices = new();
                LoadFromFile_NewVectors(newVectorsAndMatrices, "NewVectors.bin", _loggersSet);
                CurrentNewVectorsAndMatrices_ToDisplay = newVectorsAndMatrices;
                //CurrentClusterization_Algorithm_ToDisplay = Clusterization_Algorithm_KMeans;
                //CurrentProjectionOptimization_Algorithm_ToDisplay = ProjectionOptimization_Algorithm_Variant3;
                
                //LoadFromFile_NewVectorsAndMatrices(Clusterization_Algorithm_KMeans, _loggersSet);
                //ProxWordsNewMatrix_Calculate(Clusterization_Algorithm_KMeans, _loggersSet);

                //NewVectorsAndMatrices newVectorsAndMatrices = Calculate_NewVectors(Clusterization_Algorithm_Classes, ProjectionOptimization_Algorithm_Random, _loggersSet);
                //CurrentNewVectorsAndMatrices_ToDisplay = newVectorsAndMatrices;
                //SaveToFile_NewVectorsAndMatrices(Algorithm_Classes, _loggersSet);
                //LoadFromFile_NewVectorsAndMatrices(Clusterization_Algorithm_Classes, _loggersSet);
                //ProxWordsNewMatrix_Calculate(Clusterization_Algorithm_Classes, _loggersSet);

                CurrentWordsNewEmbeddings = Calculate_WordsNewEmbeddings(_loggersSet);
                SaveToFile_WordsNewEmbeddings(CurrentWordsNewEmbeddings, "NewWordsEmbeddings.csv", _loggersSet);

                CompareOldAndNewPhraseEmbeddings(_loggersSet);
            });            
        }        

        public void Close()
        {            
        }            

        #endregion

        #region private functions

        private void InitializeWords(ILoggersSet loggersSet)
        {            
            Control.UseNativeMKL();

            string programDataDirectoryFullName = Directory.GetCurrentDirectory();
            loggersSet.UserFriendlyLogger.LogInformation(LinearAlgebraControl.Provider.ToString());

            #region RU Words Initialization

            Words_RU = new(20000); // Initial reserved capacity                        
            
            foreach (var line in File.ReadAllLines(Path.Combine(programDataDirectoryFullName, @"Data\Ssz.AI.AdvancedEmbedding\RU\model_20000.csv")))
            {
                var parts = CsvHelper.ParseCsvLine(",", line);
                if (parts.Length < 300 || String.IsNullOrEmpty(parts[0]))
                    continue;
                Word word = new Word
                {
                    Index = Words_RU.Count,
                    Name = parts[0]!,
                };                
                if (parts.Length - 2 != OldVectorLength)
                {
                    loggersSet.UserFriendlyLogger.LogError("Incorrect vector length in input = " + (parts.Length - 2));
                    return;
                }
                var oldVectror = word.OldVector;
                foreach (int i in Enumerable.Range(0, parts.Length - 2))
                {
                    oldVectror[i] = Single.Parse(parts[i + 1] ?? @"", CultureInfo.InvariantCulture);
                }
                //word.Vector = Vector<float>.Build.Dense(vector);
                float norm = TensorPrimitives.Norm(oldVectror);
                TensorPrimitives.Divide(oldVectror, norm, oldVectror);
                word.Freq = new Any(parts[^1]).ValueAsDouble(false);

                Words_RU.Add(word);
            }

            #endregion                       

            #region EN Words Initialization

            Words_EN = new(20100); // Initial reserved capacity                        

            foreach (var line in File.ReadAllLines(Path.Combine(programDataDirectoryFullName, @"Data\Ssz.AI.AdvancedEmbedding\EN\glove.42B.300d_20000.txt")))
            {
                var parts = CsvHelper.ParseCsvLine(" ", line);
                if (parts.Length < 300 || String.IsNullOrEmpty(parts[0]))
                    continue;
                Word word = new Word
                {
                    Index = Words_EN.Count,
                    Name = parts[0]!,
                };
                if (parts.Length - 1 != OldVectorLength)
                {
                    loggersSet.UserFriendlyLogger.LogError("Incorrect vector length in input = " + (parts.Length - 2));
                    return;
                }
                var oldVectror = word.OldVector;
                foreach (int i in Enumerable.Range(0, parts.Length - 2))
                {
                    oldVectror[i] = Single.Parse(parts[i + 1] ?? @"", CultureInfo.InvariantCulture);
                }
                //word.Vector = Vector<float>.Build.Dense(vector);
                float norm = TensorPrimitives.Norm(oldVectror);
                TensorPrimitives.Divide(oldVectror, norm, oldVectror);
                word.Freq = new Any(parts[^1]).ValueAsDouble(false);

                Words_EN.Add(word);
            }

            #endregion   
        }

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

        private void ProxWordsOldMatrix_Calculate(ILoggersSet loggersSet)
        {
            var stopwatch = Stopwatch.StartNew();

            var proxWordsOldMatrix = new float[Words_RU.Count * Words_RU.Count];
            Parallel.For(0, Words_RU.Count, index1 =>
            {
                int indexBias = index1 * Words_RU.Count;
                var oldVectror = Words_RU[index1].OldVector;
                for (var index2 = 0; index2 < Words_RU.Count; index2 += 1)
                {
                    if (index2 != index1)
                        proxWordsOldMatrix[indexBias + index2] = TensorPrimitives.Dot(oldVectror, Words_RU[index2].OldVector);
                    else
                        proxWordsOldMatrix[indexBias + index2] = 1.0f;
                }
            });
            ProxWordsOldMatrix = proxWordsOldMatrix;

            stopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation("ProxWordsMatrixCalculate done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);   
        }

        private void ProxWordsOldMatrix_SaveToFile(ILoggersSet loggersSet)
        {
            string programDataDirectoryFullName = Directory.GetCurrentDirectory();

            using (MemoryStream memoryStream = new())
            using (SerializationWriter serializationWriter = new(memoryStream))
            {
                serializationWriter.Write(ProxWordsOldMatrix.Length);
                foreach (var proxWords in ProxWordsOldMatrix)
                {
                    serializationWriter.Write(proxWords);
                }
                byte[] bytes = memoryStream.ToArray();
                File.WriteAllBytes(Path.Combine(programDataDirectoryFullName, "ProxWordsOldMatrix.bin"), bytes);
            }
        }

        private void ProxWordsOldMatrix_LoadFromFile(ILoggersSet loggersSet)
        {
            var stopwatch = Stopwatch.StartNew();

            string programDataDirectoryFullName = Directory.GetCurrentDirectory();
            byte[] bytes = File.ReadAllBytes(Path.Combine(programDataDirectoryFullName, "ProxWordsOldMatrix.bin"));
            // using (var stream = File.OpenRead(Path.Combine(programDataDirectoryFullName, "ProxWordsOldMatrix.bin"))) // Slow
            using (SerializationReader serializationReader = new(bytes))
            {
                int proxWordsMatrixLength = serializationReader.ReadInt32();
                ProxWordsOldMatrix = new float[proxWordsMatrixLength];                
                foreach (int i in Enumerable.Range(0, proxWordsMatrixLength))
                {
                    ProxWordsOldMatrix[i] = serializationReader.ReadSingle();                    
                }                
            }

            stopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation("ProxWordsMatrixLoadFromFile done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
        }

        /// <summary>
        ///     Writes PrimaryWords, ClusterIndices
        /// </summary>
        /// <param name="algorithm"></param>
        /// <param name="loggersSet"></param>
        private void Clusterization_Algorithm_SaveToFile(Clusterization_Algorithm algorithm, ILoggersSet loggersSet)
        {
            var stopwatch = Stopwatch.StartNew();

            string programDataDirectoryFullName = Directory.GetCurrentDirectory();

            using (MemoryStream memoryStream = new())
            using (SerializationWriter serializationWriter = new(memoryStream))
            {
                var list = algorithm.PrimaryWords!.Select(w => w.Index).ToList();
                serializationWriter.WriteList(list);
                serializationWriter.WriteArray(algorithm.ClusterIndices);

                byte[] bytes = memoryStream.ToArray();
                File.WriteAllBytes(Path.Combine(programDataDirectoryFullName, "Clusterization_Algorithm_" + algorithm.Name + ".bin"), bytes);
            }

            stopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation(algorithm.Name + " AAlgorithm_SaveToFile1 done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
        }

        /// <summary>
        ///     Reads PrimaryWords, ClusterIndices
        /// </summary>
        /// <param name="algorithm"></param>
        /// <param name="loggersSet"></param>
        private void Clusterization_Algorithm_LoadFromFile(Clusterization_Algorithm algorithm, ILoggersSet loggersSet)
        {
            var stopwatch = Stopwatch.StartNew();

            string programDataDirectoryFullName = Directory.GetCurrentDirectory();
            byte[] bytes = File.ReadAllBytes(Path.Combine(programDataDirectoryFullName, "Clusterization_Algorithm_" + algorithm.Name + ".bin"));
            using (SerializationReader serializationReader = new(bytes))
            {
                var list = serializationReader.ReadList<int>()!;

                //if (list.Count != PrimaryWordsCount)
                //    throw new InvalidOperationException();

                algorithm.PrimaryWords = list.Select(i => Words_RU[i]).ToArray();
                algorithm.ClusterIndices = serializationReader.ReadArray<int>();
            }

            stopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation(algorithm.Name + " Algorithm_LoadFromFile1 done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
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

        /// <summary>
        ///     
        /// </summary>
        /// <param name="newVectorsAndMatrices"></param>
        /// <param name="fileName"></param>
        /// <param name="loggersSet"></param>
        private void SaveToFile_NewVectors(NewVectorsAndMatrices newVectorsAndMatrices, string fileName, ILoggersSet loggersSet)
        {
            var totalStopwatch = Stopwatch.StartNew();

            string programDataDirectoryFullName = Directory.GetCurrentDirectory();

            using (MemoryStream memoryStream = new())
            using (SerializationWriter serializationWriter = new(memoryStream))
            {
                serializationWriter.Write(newVectorsAndMatrices.NewVectors.Length);
                if (newVectorsAndMatrices.NewVectors is not null)
                    foreach (var vectorNew in newVectorsAndMatrices.NewVectors)
                    {
                        serializationWriter.WriteArray(vectorNew);
                    }                
                //serializationWriter.WriteArray(algorithm.ProxWordsNewMatrix);

                serializationWriter.Write(newVectorsAndMatrices.NewVectors_PrimaryOnly.Length);
                if (newVectorsAndMatrices.NewVectors_PrimaryOnly is not null)
                    foreach (var vectorNew in newVectorsAndMatrices.NewVectors_PrimaryOnly)
                    {
                        serializationWriter.WriteArray(vectorNew);
                    }
                //serializationWriter.WriteArray(algorithm.ProxWordsNewMatrix_PrimaryOnly);

                serializationWriter.Write(newVectorsAndMatrices.NewVectors_SecondaryOnly.Length);
                if (newVectorsAndMatrices.NewVectors_SecondaryOnly is not null)
                    foreach (var vectorNew in newVectorsAndMatrices.NewVectors_SecondaryOnly)
                    {
                        serializationWriter.WriteArray(vectorNew);
                    }
                //serializationWriter.WriteArray(algorithm.ProxWordsNewMatrix_SecondaryOnly);

                byte[] bytes = memoryStream.ToArray();
                File.WriteAllBytes(Path.Combine(programDataDirectoryFullName, fileName), bytes);
            }

            totalStopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation($"{nameof(SaveToFile_NewVectors)} done. Elapsed Milliseconds: {totalStopwatch.ElapsedMilliseconds}");
        }

        /// <summary>
        ///     
        /// </summary>
        /// <param name="newVectorsAndMatrices"></param>
        /// <param name="fileName"></param>
        /// <param name="loggersSet"></param>
        private void LoadFromFile_NewVectors(NewVectorsAndMatrices newVectorsAndMatrices, string fileName, ILoggersSet loggersSet)
        {
            var totalStopwatch = Stopwatch.StartNew();

            string programDataDirectoryFullName = Directory.GetCurrentDirectory();
            byte[] bytes = File.ReadAllBytes(Path.Combine(programDataDirectoryFullName, fileName));
            using (SerializationReader serializationReader = new(bytes))
            {
                int newVectorsLength = serializationReader.ReadInt32();
                if (newVectorsLength > 0)
                {
                    var newVectors = new float[newVectorsLength][];
                    foreach (int i in Enumerable.Range(0, newVectorsLength))
                    {
                        newVectors[i] = serializationReader.ReadArray<float>()!;
                    }
                    newVectorsAndMatrices.NewVectors = newVectors;
                }
                //algorithm.ProxWordsNewMatrix = serializationReader.ReadArray<float>();

                newVectorsLength = serializationReader.ReadInt32();
                if (newVectorsLength > 0)
                {
                    var newVectors_PrimaryOnly = new float[newVectorsLength][];
                    foreach (int i in Enumerable.Range(0, newVectorsLength))
                    {
                        newVectors_PrimaryOnly[i] = serializationReader.ReadArray<float>()!;
                    }
                    newVectorsAndMatrices.NewVectors_PrimaryOnly = newVectors_PrimaryOnly;
                }
                //algorithm.ProxWordsNewMatrix_PrimaryOnly = serializationReader.ReadArray<float>();

                newVectorsLength = serializationReader.ReadInt32();
                if (newVectorsLength > 0)
                {
                    var newVectors_SecondaryOnly = new float[newVectorsLength][];
                    foreach (int i in Enumerable.Range(0, newVectorsLength))
                    {
                        newVectors_SecondaryOnly[i] = serializationReader.ReadArray<float>()!;
                    }
                    newVectorsAndMatrices.NewVectors_SecondaryOnly = newVectors_SecondaryOnly;
                }
                //algorithm.ProxWordsNewMatrix_SecondaryOnly = serializationReader.ReadArray<float>();
            }

            totalStopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation($"{nameof(LoadFromFile_NewVectors)} done. Elapsed Milliseconds: {totalStopwatch.ElapsedMilliseconds}");
        }

        /// <summary>
        ///     
        /// </summary>
        /// <param name="newVectorsAndMatrices"></param>
        /// <param name="fileName"></param>
        /// <param name="loggersSet"></param>
        private void LoadFromFile_NewVectorsAndMatrices(NewVectorsAndMatrices newVectorsAndMatrices, string fileName, ILoggersSet loggersSet)
        {
            var stopwatch = Stopwatch.StartNew();

            string programDataDirectoryFullName = Directory.GetCurrentDirectory();
            byte[] bytes = File.ReadAllBytes(Path.Combine(programDataDirectoryFullName, fileName));
            using (SerializationReader serializationReader = new(bytes))
            {
                int newVectorsLength = serializationReader.ReadInt32();
                if (newVectorsLength > 0)
                {
                    var newVectors = new float[newVectorsLength][];
                    foreach (int i in Enumerable.Range(0, newVectorsLength))
                    {
                        newVectors[i] = serializationReader.ReadArray<float>()!;
                    }
                    newVectorsAndMatrices.NewVectors = newVectors;
                }                
                //algorithm.ProxWordsNewMatrix = serializationReader.ReadArray<float>();

                newVectorsLength = serializationReader.ReadInt32();
                if (newVectorsLength > 0)
                {
                    var newVectors_PrimaryOnly = new float[newVectorsLength][];
                    foreach (int i in Enumerable.Range(0, newVectorsLength))
                    {
                        newVectors_PrimaryOnly[i] = serializationReader.ReadArray<float>()!;
                    }
                    newVectorsAndMatrices.NewVectors_PrimaryOnly = newVectors_PrimaryOnly;
                }
                //algorithm.ProxWordsNewMatrix_PrimaryOnly = serializationReader.ReadArray<float>();

                newVectorsLength = serializationReader.ReadInt32();
                if (newVectorsLength > 0)
                {
                    var newVectors_SecondaryOnly = new float[newVectorsLength][];
                    foreach (int i in Enumerable.Range(0, newVectorsLength))
                    {
                        newVectors_SecondaryOnly[i] = serializationReader.ReadArray<float>()!;
                    }
                    newVectorsAndMatrices.NewVectors_SecondaryOnly = newVectors_SecondaryOnly;
                }
                //algorithm.ProxWordsNewMatrix_SecondaryOnly = serializationReader.ReadArray<float>();
            }

            stopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation("LoadFromFile_NewVectorsAndMatrices done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
        }
        
        private void SaveToFile_ProjectionIndices(ProjectionOptimization_Algorithm projectionOptimization_Algorithm, string fileName, ILoggersSet loggersSet)
        {
            var stopwatch = Stopwatch.StartNew();

            string programDataDirectoryFullName = Directory.GetCurrentDirectory();

            using (MemoryStream memoryStream = new())
            using (SerializationWriter serializationWriter = new(memoryStream))
            {
                serializationWriter.WriteArray(projectionOptimization_Algorithm.WordsProjectionIndices);

                byte[] bytes = memoryStream.ToArray();
                File.WriteAllBytes(Path.Combine(programDataDirectoryFullName, fileName), bytes);
            }

            stopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation("SaveToFile_ProjectionIndices done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
        }

        private void LoadFromFile_ProjectionIndices(ProjectionOptimization_Algorithm projectionOptimization_Algorithm, string fileName, ILoggersSet loggersSet)
        {
            var stopwatch = Stopwatch.StartNew();

            string programDataDirectoryFullName = Directory.GetCurrentDirectory();
            byte[] bytes = File.ReadAllBytes(Path.Combine(programDataDirectoryFullName, fileName));
            using (SerializationReader serializationReader = new(bytes))
            {
                projectionOptimization_Algorithm.WordsProjectionIndices = serializationReader.ReadArray<int>()!;
            }

            stopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation("LoadFromFile_ProjectionIndices done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
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

        private void CortexLoadFromFile(ILoggersSet loggersSet)
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
                        Words_RU[pointRef.WordIndex].Point = pointRef;
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

        private Clusterization_AlgorithmEnum _primaryWordsSelectionMethod;
        private readonly float[] _v1 = new float[2];

        #endregion
    }

    public class Word
    {
        public Word()
        {
            OldVector = new float[Model.OldVectorLength];
            NewVector_ToDisplay = new float[Model.NewVectorLength];
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

        public float[]? NewVector_ToDisplay;        

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

    /// <summary>
    ///     Primary Words Selection Algorithm Enum
    /// </summary>
    public enum Clusterization_AlgorithmEnum
    {
        None = 0,
        Random,
        Algorithm_Em,
        Algorithm_KMeans,
        Algorithm_Classes,
    }

    /// <summary>
    ///     Primary Words Selection Algorithm
    /// </summary>
    public class Clusterization_Algorithm
    {
        public string Name = null!;

        public Word[]? PrimaryWords;

        /// <summary>
        ///    For each Word. ClusterIndices.Length == Words.Length
        /// </summary>
        public int[]? ClusterIndices;        
    }    

    public enum ProjectionOptimization_AlgorithmEnum
    {
        None = 0,
        Random,
        Variant3,        
    }
    
    public class ProjectionOptimization_Algorithm
    {
        public string Name = null!;

        public int[] WordsProjectionIndices = null!;
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
        ///     [Словоформа, NewVector Index]
        /// </summary>
        public CaseInsensitiveDictionary<int> Words = new();
    }
}


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