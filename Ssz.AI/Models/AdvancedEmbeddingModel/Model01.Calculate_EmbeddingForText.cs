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
using Microsoft.Extensions.Logging;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Logging;

namespace Ssz.AI.Models.AdvancedEmbeddingModel
{
    public partial class Model01
    {
        public float[] GetEmbeddingForPhrase(string phrase)
        {
            if (CurrentWordsNewEmbeddings is null || CurrentDiscreteVectorsAndMatrices_ToDisplay is null)
                return new float[0];

            var wordsDictionary = CurrentWordsNewEmbeddings.Words;
            var discreteVectors = CurrentDiscreteVectorsAndMatrices_ToDisplay.DiscreteVectors;

            float[] embedding = new float[Constants.DiscreteVectorLength];

            foreach (var wordString in phrase.Split())
            {
                if (wordsDictionary.TryGetValue(NormalizeWord2(wordString), out int index))
                {
                    TensorPrimitives.Add(embedding, discreteVectors[index], embedding);
                }
                else
                {
                }
            }

            return embedding;
        }

        private string NormalizeWord2(string wordString)
        {
            wordString = NormalizeWord(wordString);
            wordString = new string(wordString.Where(c => Char.IsLetter(c)).ToArray());
            return wordString;
        }

        private string NormalizeWord(string wordString)
        {
            return wordString.ToLowerInvariant().Replace('ё', 'е');
        }

        private WordsNewEmbeddings Calculate_WordsNewEmbeddings(LanguageInfo languageInfo, ILoggersSet loggersSet)
        {
            var totalStopwatch = Stopwatch.StartNew();

            WordsNewEmbeddings embeddingForText = new();

            string programDataDirectoryFullName = Directory.GetCurrentDirectory();

            CaseInsensitiveDictionary<Word> wordsDictionary = new();
            foreach (var word in languageInfo.Words)
            {
                wordsDictionary[word.Name] = word;
            }
            
            WordGroup? wordGroup = null;
            foreach (var line in File.ReadAllLines(Path.Combine(programDataDirectoryFullName, @"dict.opcorpora.txt")))
            {                
                if (String.IsNullOrWhiteSpace(line))
                {                    
                    if (wordGroup is not null && wordGroup.Word is not null)
                    {
                        foreach (var wordString in wordGroup.Words) 
                        {
                            embeddingForText.Words[wordString] = wordGroup.Word.Index;
                        }

                        wordGroup = null;
                    }
                    continue;
                }
                if (new Any(line).ValueAsInt32(false) > 0)
                {   
                    continue;
                }                                       
                var indexOfTab = line.IndexOf('\t');
                if (indexOfTab > 0)
                {
                    if (wordGroup is null)
                        wordGroup = new WordGroup();

                    var wordString = NormalizeWord(line.Substring(0, indexOfTab));
                    var word = FindWord(wordsDictionary, wordString, line.Substring(indexOfTab + 1));
                    if (word != null)
                        wordGroup.Word = word;
                    wordGroup.Words.Add(wordString);
                }                
            }

            totalStopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation($"{nameof(Calculate_WordsNewEmbeddings)} done. Elapsed Milliseconds: {totalStopwatch.ElapsedMilliseconds}");

            return embeddingForText;
        }

        private Word? FindWord(CaseInsensitiveDictionary<Word> wordsDictionary, string v0, string v1)
        {
            string postfixedWord = @"";
            if (v1.StartsWith(@"ADJF"))
                postfixedWord = v0 + "_ADJ";
            else if (v1.StartsWith(@"ADVB"))
                postfixedWord = v0 + "_ADV";
            else if (v1.StartsWith(@"NOUN"))
                postfixedWord = v0 + "_NOUN";
            else if (v1.StartsWith(@"VERB"))
                postfixedWord = v0 + "_VERB";
            
            if (postfixedWord != @"")
            {
                return wordsDictionary.TryGetValue(postfixedWord);
            }

            return null;
        }

        private class WordGroup
        {
            public List<string> Words = new();

            /// <summary>
            ///     Primary or secondary word
            /// </summary>
            public Word? Word;
        }
    }
}