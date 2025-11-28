using Avalonia.Media;
using Microsoft.Extensions.Logging;
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
    public static InputCorpusData GetInputCorpusData(
        List<Word> words,
        Random random, 
        int discreteVectorLength, 
        int discreteOptimizedVector_PrimaryBitsCount,
        ILogger logger)
    {
        string fileFullName = Path.Combine(AIConstants.DataDirectory, Model01.AdvancedEmbedding2_Directory, "input_sequences.txt");
        var sequences = MorphologicalTextParser.LoadFromFile(fileFullName);
        InputCorpusData inputCorpusData = new();
        Dictionary<string, Word> wordsDictionary = inputCorpusData.WordsDictionary;   
        foreach (var word in words)
        {
            wordsDictionary[word.Name] = word;
        }
        int[] indices = new int[discreteVectorLength];
        List<Word> sequenceWords = new();
        List<Word> sequencesWords = new(10000);
        List<Cortex.Memory> cortexMemories = inputCorpusData.CortexMemories;        
        foreach (var s in sequences)
        {
            sequenceWords.Clear();
            foreach (var wordName in s)
            {                
                if (!wordsDictionary.TryGetValue(wordName, out Word? word))
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
                    random.Shuffle(indices);
                    for (int i = 0; i < discreteOptimizedVector_PrimaryBitsCount; i += 1)
                    {
                        word.DiscreteRandomVector[indices[i]] = 1.0f;
                    }
                    wordsDictionary.Add(wordName, word);
                    words.Add(word);
                }
                word.Temp_InCorpusCount += 1;
                sequenceWords.Add(word);
                sequencesWords.Add(word);
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
                cortexMemory.DiscreteRandomVector_Color = System.Drawing.Color.Black;  //Visualisation.GetColorFromDiscreteVector(cortexMemory.DiscreteRandomVector);
                cortexMemories.Add(cortexMemory);
            }
        }
        foreach (var word in sequencesWords)
        {
            word.CorpusFreq = (float)word.Temp_InCorpusCount / sequencesWords.Count;
        }
        inputCorpusData.OrderedWords = words.OrderByDescending(w => w.CorpusFreq).ToList();

        for (int ci = 0; ci < 300; ci += 1)
        {
            var cortexMemory = cortexMemories[random.Next(cortexMemories.Count)];
            cortexMemory.DiscreteRandomVector_Color = Visualisation.GetColorFromDiscreteVector(cortexMemory.DiscreteRandomVector);
        }

        logger.LogInformation($"InputCorpusData loaded from file: {fileFullName}.");

        return inputCorpusData;
    }
}
