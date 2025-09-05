using Ssz.Utils.Serialization;
using System.Collections.Generic;
using System.Linq;

namespace Ssz.AI.Models.AdvancedEmbeddingModel
{
    /// <summary>
    ///     Primary Words Selection AlgorithmData
    /// </summary>
    public class Clusterization_AlgorithmData : ISerializableModelObject
    {
        #region construction and destruction

        public Clusterization_AlgorithmData(LanguageInfo languageInfo)
        {
            LanguageInfo = languageInfo;
        }

        #endregion
        
        public readonly LanguageInfo LanguageInfo;

        public string Name = null!;

        public Word[] PrimaryWords = null!;

        public bool[] IsPrimaryWord = null!;

        /// <summary>
        ///    For each Word. ClusterIndices.Length == Words.Length
        /// </summary>
        public int[] ClusterIndices = null!;

        public void GenerateOwnedData(int primaryWordsCount)
        {
            PrimaryWords = new Word[primaryWordsCount];
            IsPrimaryWord = new bool[LanguageInfo.Words.Count];
            ClusterIndices = new int[LanguageInfo.Words.Count];
        }

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(1))
            {
                var primaryWordIndices = PrimaryWords!.Select(w => w.Index).ToList();
                writer.WriteList(primaryWordIndices);
                writer.WriteArray(ClusterIndices);
            }
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {   
            using (Block block = reader.EnterBlock())
            {
                switch (block.Version)
                {
                    case 1:
                        var primaryWordIndices = reader.ReadList<int>()!;

                        //if (list.Count != PrimaryWordsCount)
                        //    throw new InvalidOperationException();

                        PrimaryWords = primaryWordIndices.Select(i => LanguageInfo.Words[i]).ToArray();
                        IsPrimaryWord = new bool[LanguageInfo.Words.Count];
                        ClusterIndices = reader.ReadArray<int>()!;                        
                        foreach (int primaryWordIndex in primaryWordIndices)
                        {
                            IsPrimaryWord[primaryWordIndex] = true;
                        }
                        break;
                }
            }
        }        
    }
}
