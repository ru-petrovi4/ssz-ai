using System.Collections.Generic;

namespace Ssz.AI.Models.AdvancedEmbeddingModel2;

public class InputCorpusData
{
    public Dictionary<string, Word> Dictionary = new(10000);

    public List<Word> Words = new(10000);

    public int CurrentWordIndex = -1;

    public List<Cortex.Memory> CortexMemories = new(10000);

    public int CurrentCortexMemoryIndex = -1;
}
