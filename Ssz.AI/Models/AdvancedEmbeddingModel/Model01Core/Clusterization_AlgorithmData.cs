using Ssz.AI.Helpers;
using Ssz.Utils.Serialization;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core
{
    /// <summary>
    ///     Primary Words Selection AlgorithmData
    /// </summary>
    public class Clusterization_AlgorithmData : ISerializableModelObject
    {
        #region construction and destruction

        public Clusterization_AlgorithmData(LanguageInfo languageInfo, string name)
        {
            LanguageInfo = languageInfo;
            Name = name;
        }

        #endregion
        
        public readonly LanguageInfo LanguageInfo;

        public readonly string Name = null!;

        /// <summary>
        ///    For each Word. ClusterIndices.Length == Words.Length
        /// </summary>
        public int[] ClusterIndices = null!;

        public ClusterInfo[] ClusterInfos = null!;

        public Word[] PrimaryWords = null!;        

        public bool[] IsPrimaryWord = null!;

        // Параметры модели - изучаются в процессе обучения
        /// <summary>
        /// μ_k - направления средних для каждого кластера [K x D]
        /// </summary>
        /// <remarks>device: CPU</remarks>
        public Tensor MeanDirections { get; set; } = null!;

        /// <summary>
        /// κ_k - параметры концентрации для каждого кластера [K]
        /// </summary>
        /// <remarks>device: CPU</remarks>
        public Tensor Concentrations { get; set; } = null!;

        /// <summary>
        /// α_k - коэффициенты смешивания [K]
        /// </summary>
        /// <remarks>device: CPU</remarks>
        public Tensor MixingCoefficients { get; set; } = null!;

        public void GenerateOwnedData(int clustersCount)
        {
            ClusterIndices = new int[LanguageInfo.Words.Count];
            int d = LanguageInfo.Words[0].OldVectorNormalized.Length;
            ClusterInfos = Enumerable.Range(0, clustersCount).Select(_ => new ClusterInfo
            { 
                CentroidOldVectorNormalized = new float[d] 
            }).ToArray();
            PrimaryWords = new Word[clustersCount];
            IsPrimaryWord = new bool[LanguageInfo.Words.Count];            
        }

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(1))
            {
                writer.WriteArray(ClusterIndices);
                writer.WriteArrayOfOwnedDataSerializable(ClusterInfos, null);
                var primaryWordIndices = PrimaryWords!.Select(w => w.Index).ToList();
                writer.WriteList(primaryWordIndices);

                TorchSharpHelper.WriteTensor(MeanDirections, writer);
                TorchSharpHelper.WriteTensor(Concentrations, writer);
                TorchSharpHelper.WriteTensor(MixingCoefficients, writer);
            }
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {   
            using (Block block = reader.EnterBlock())
            {
                switch (block.Version)
                {
                    case 1:
                        ClusterIndices = reader.ReadArray<int>()!;
                        ClusterInfos = reader.ReadArrayOfOwnedDataSerializable(() => new ClusterInfo(), null);
                        var primaryWordIndices = reader.ReadList<int>()!;
                        //if (list.Count != PrimaryWordsCount)
                        //    throw new InvalidOperationException();                        
                        PrimaryWords = primaryWordIndices.Select(i => LanguageInfo.Words[i]).ToArray();
                        IsPrimaryWord = new bool[LanguageInfo.Words.Count];                                              
                        foreach (int primaryWordIndex in primaryWordIndices)
                        {
                            IsPrimaryWord[primaryWordIndex] = true;
                        }

                        MeanDirections = TorchSharpHelper.ReadTensor(reader);
                        Concentrations = TorchSharpHelper.ReadTensor(reader);
                        MixingCoefficients = TorchSharpHelper.ReadTensor(reader);
#if DEBUG
                        //var sum = IsPrimaryWord.Sum(b => b ? 1 : 0);
#endif
                        break;
                }
            }
        }        
    }
}
