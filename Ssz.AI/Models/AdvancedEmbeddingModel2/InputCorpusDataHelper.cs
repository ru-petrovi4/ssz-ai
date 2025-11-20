using Ssz.AI.Core;
using Ssz.Utils.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Models.AdvancedEmbeddingModel2;

public static class InputCorpusDataHelper
{
    public static InputCorpusData GetInputCorpusData(Random r, int discreteVectorLength)
    {        
        var sequences = MorphologicalTextParser.LoadFromFile(
            Path.Combine(AIConstants.DataDirectory, Model01.AdvancedEmbedding2_Directory, "input_sequences.txt")
            );
        InputCorpusData inputCorpusData = new();
        Dictionary<string, Word> dictionary = inputCorpusData.Dictionary;
        var words = inputCorpusData.Words;
        int[] indices = new int[discreteVectorLength];
        List<Word> sequenceWords = new();
        List<Cortex.Memory> cortexMemories = inputCorpusData.CortexMemories;
        int corpus_WordsCount = 0;
        foreach (var s in sequences)
        {
            sequenceWords.Clear();
            foreach (var wordName in s)
            {
                corpus_WordsCount += 1;
                if (!dictionary.TryGetValue(wordName, out Word? word))
                {
                    word = new()
                    {
                        Name = wordName,
                        Index = words.Count,
                        DiscreteRandomVector = new float[discreteVectorLength],
                        DiscreteOptimizedVector = new float[discreteVectorLength],
                        DiscreteOptimizedVector_PrimaryBitsOnly = new float[discreteVectorLength],
                        DiscreteOptimizedVector_SecondaryBitsOnly = new float[discreteVectorLength],
                    };
                    for (int i = 0; i < discreteVectorLength; i += 1)
                    {
                        indices[i] = i;
                    }
                    r.Shuffle(indices);
                    for (int i = 0; i < 7; i += 1)
                    {
                        word.DiscreteRandomVector[indices[i]] = 1.0f;
                    }
                    dictionary.Add(wordName, word);
                    words.Add(word);
                }
                word.Temp_InCorpusCount += 1;
                sequenceWords.Add(word);
            }
            if (sequenceWords.Count > 2)
            {
                Cortex.Memory cortexMemory = new Cortex.Memory()
                {
                    DiscreteRandomVector = new float[discreteVectorLength],
                    WordIndices = sequenceWords.Select(w => w.Index).ToArray()
                };
                for (int i = 0; i < sequenceWords.Count; i += 1)
                {
                    TensorPrimitives.Add(cortexMemory.DiscreteRandomVector, sequenceWords[i].DiscreteRandomVector, cortexMemory.DiscreteRandomVector);
                }
                TensorPrimitives.Min(cortexMemory.DiscreteRandomVector, 1.0f, cortexMemory.DiscreteRandomVector);
                cortexMemory.DiscreteRandomVector_Color = Visualisation.GetColorFromDiscreteVector(cortexMemory.DiscreteRandomVector);
                cortexMemories.Add(cortexMemory);
            }
        }
        foreach (var kvp in dictionary)
        {
            kvp.Value.CorpusFreq = (float)kvp.Value.Temp_InCorpusCount / corpus_WordsCount;
        }
        inputCorpusData.OrderedWords = words.OrderByDescending(w => w.CorpusFreq).ToList();
        return inputCorpusData;
    }
}
