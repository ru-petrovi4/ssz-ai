using Ssz.Utils.Serialization;
using System.Collections.Generic;
using System.Linq;

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

        public float[][] ClusterCenters = null!;

        public Word[] PrimaryWords = null!;        

        public bool[] IsPrimaryWord = null!;        

        public void GenerateOwnedData(int clustersCount)
        {
            ClusterIndices = new int[LanguageInfo.Words.Count];
            ClusterCenters = new float[clustersCount][];
            PrimaryWords = new Word[clustersCount];
            IsPrimaryWord = new bool[LanguageInfo.Words.Count];            
        }

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(1))
            {
                writer.WriteArray(ClusterIndices);
                writer.Write(ClusterCenters.Length);
                foreach (int clusterIndex in Enumerable.Range(0, ClusterCenters.Length))
                {
                    writer.WriteArray(ClusterCenters[clusterIndex]);
                }
                var primaryWordIndices = PrimaryWords!.Select(w => w.Index).ToList();
                writer.WriteList(primaryWordIndices);                
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
                        ClusterCenters = new float[reader.ReadInt32()][];
                        foreach (int clusterIndex in Enumerable.Range(0, ClusterCenters.Length))
                        {
                            ClusterCenters[clusterIndex] = reader.ReadArray<float>()!;
                        }

                        var primaryWordIndices = reader.ReadList<int>()!;
                        //if (list.Count != PrimaryWordsCount)
                        //    throw new InvalidOperationException();                        
                        PrimaryWords = primaryWordIndices.Select(i => LanguageInfo.Words[i]).ToArray();
                        IsPrimaryWord = new bool[LanguageInfo.Words.Count];                                              
                        foreach (int primaryWordIndex in primaryWordIndices)
                        {
                            IsPrimaryWord[primaryWordIndex] = true;
                        }
#if DEBUG
                        //var sum = IsPrimaryWord.Sum(b => b ? 1 : 0);
#endif
                        break;
                }
            }
        }        
    }
}
