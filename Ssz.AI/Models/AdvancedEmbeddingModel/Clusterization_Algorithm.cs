using Ssz.Utils.Serialization;
using System.Collections.Generic;
using System.Linq;

namespace Ssz.AI.Models.AdvancedEmbeddingModel
{
    /// <summary>
    ///     Primary Words Selection Algorithm
    /// </summary>
    public class Clusterization_Algorithm : IOwnedDataSerializable
    {
        #region construction and destruction

        public Clusterization_Algorithm(List<Word> words)
        {
            Words = words;
        }

        #endregion

        public readonly List<Word> Words;

        public string Name = null!;

        public Word[]? PrimaryWords;

        /// <summary>
        ///    For each Word. ClusterIndices.Length == Words.Length
        /// </summary>
        public int[]? ClusterIndices;

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(1))
            {
                var list = PrimaryWords!.Select(w => w.Index).ToList();
                writer.WriteList(list);
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
                        var list = reader.ReadList<int>()!;

                        //if (list.Count != PrimaryWordsCount)
                        //    throw new InvalidOperationException();

                        PrimaryWords = list.Select(i => Words[i]).ToArray();
                        ClusterIndices = reader.ReadArray<int>();
                        break;
                }
            }
        }        
    }
}
