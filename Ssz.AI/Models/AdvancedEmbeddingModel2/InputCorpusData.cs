using MathNet.Numerics.Distributions;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;

namespace Ssz.AI.Models.AdvancedEmbeddingModel2;

public class InputCorpusData
{
    public Dictionary<string, Word> Dictionary = new(10000);

    public List<Word> Words = new(10000);

    public List<Word> OrderedWords = null!;

    public int Current_OrderedWords_Index = -1;

    public List<Cortex.Memory> CortexMemories = new(10000);

    public int CurrentCortexMemoryIndex = -1;
}
