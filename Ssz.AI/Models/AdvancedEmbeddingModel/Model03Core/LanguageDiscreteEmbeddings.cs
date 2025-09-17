using Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;
using Ssz.Utils.Serialization;
using System.Collections.Generic;
using System.Linq;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model03Core;

public class LanguageDiscreteEmbeddings : IOwnedDataSerializable
{
    /// <summary>        
    ///     <para>Ordered Descending by Freq</para>      
    /// </summary>
    public List<WordWithDiscreteEmbedding> Words = null!;

    /// <summary>        
    ///     <para>Ordered Descending by Freq</para>      
    /// </summary>
    public List<WordWithDiscreteEmbedding> PrimaryWords = null!;

    public void SerializeOwnedData(SerializationWriter serializationWriter, object? context)
    {
        using (serializationWriter.EnterBlock(1))
        {
            serializationWriter.WriteListOfOwnedDataSerializable(Words, null);
            serializationWriter.WriteList(PrimaryWords.Select(w => w.Index).ToList());            
        }
    }

    public void DeserializeOwnedData(SerializationReader serializationReader, object? context)
    {   
        using (Block block = serializationReader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    Words = serializationReader.ReadListOfOwnedDataSerializable(() => new WordWithDiscreteEmbedding(), null);

                    var primaryWordIndices = serializationReader.ReadList<int>()!;
                    PrimaryWords = primaryWordIndices.Select(i => Words[i]).ToList();
                    break;
            }
        }
    }
}
