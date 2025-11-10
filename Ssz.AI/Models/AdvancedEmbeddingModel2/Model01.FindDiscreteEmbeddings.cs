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
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.Providers.LinearAlgebra;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Ssz.AI.Core;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel2;

public class Model01
{
    public const string AdvancedEmbedding2_Directory = "Ssz.AI.AdvancedEmbedding2";

    #region construction and destruction

    public Model01()
    {
        _loggersSet = new LoggersSet(NullLogger.Instance, new UserFriendlyLogger((l, id, s) => DebugWindow.Instance.AddLine(s)));
    }

    #endregion

    #region public functions       

    public static readonly ModelConstants Constants = new();

    public void StemInputText()
    {
        // Настройка информации о процессе
        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            WorkingDirectory = Path.Combine(AIConstants.DataDirectory, AdvancedEmbedding2_Directory),
            FileName = Path.Combine(AIConstants.DataDirectory, AdvancedEmbedding2_Directory, "mystem.exe"),  // Или путь к исполняемому файлу, например, @"C:\path\to\program.exe"
            Arguments = "-c -d -i input.fb2 input_stemmed.txt",  // Аргументы, если нужны, например, "file.txt"
            UseShellExecute = false,  // Не использовать shell для запуска (рекомендуется для контроля)
            RedirectStandardOutput = true,  // Перенаправить вывод, если нужно захватывать
            RedirectStandardError = true,
            CreateNoWindow = true  // Не создавать окно (для консольных приложений)
        };

        using (Process process = new Process())
        {
            process.StartInfo = startInfo;
            process.Start();  // Запуск процесса

            // Ожидание завершения (блокирует поток)
            process.WaitForExit();  // Или process.WaitForExit(timeoutInMs) с таймаутом

            // Получение кода выхода (0 обычно значит успех)
            int exitCode = process.ExitCode;
            _loggersSet.UserFriendlyLogger.LogInformation($"Процесс mystem.exe завершён с кодом: {exitCode}");

            // Если захватывали вывод
            string output = process.StandardOutput.ReadToEnd();
            string error = process.StandardError.ReadToEnd();
            if (!string.IsNullOrEmpty(output))
                _loggersSet.UserFriendlyLogger.LogInformation($"Вывод: {output}");
            if (!string.IsNullOrEmpty(error))
                _loggersSet.UserFriendlyLogger.LogInformation($"Ошибка: {error}");
        }
    }

    public void ExportSequences()
    {
        var sequences = MorphologicalTextParser.ParseTextToSequences(Path.Combine(AIConstants.DataDirectory, AdvancedEmbedding2_Directory, "input_stemmed_extended.txt"));
        MorphologicalTextParser.WriteToFile(
            sequences,
            Path.Combine(AIConstants.DataDirectory, AdvancedEmbedding2_Directory, "input_sequences.txt"),
            _loggersSet.UserFriendlyLogger);
    }

    public void Calculate()
    {
        Random r = new(41);

        InputCorpusData inputCorpusData = GetInputCorpusData(r);

        Cortex cortex = new Cortex(Constants);
        cortex.GenerateOwnedData();
        cortex.Prepare();

        cortex.CalculateCortexMemories(inputCorpusData, r);        
    }

    private InputCorpusData GetInputCorpusData(Random r)
    {        

        var sequences = MorphologicalTextParser.LoadFromFile(
            Path.Combine(AIConstants.DataDirectory, AdvancedEmbedding2_Directory, "input_sequences.txt"),
            _loggersSet.UserFriendlyLogger
            );
        InputCorpusData inputCorpusData = new();
        Dictionary<string, Word> dictionary = inputCorpusData.Dictionary;
        int[] indices = new int[Constants.DiscreteVectorLength];
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
                    };
                    for (int i = 0; i < Constants.DiscreteVectorLength; i += 1)
                    {
                        indices[i] = i;
                    }
                    r.Shuffle(indices);
                    for (int i = 0; i < 7; i += 1)
                    {
                        word.DiscreteRandomVector[indices[i]] = 1.0f;                        
                    }
                }
                word.Temp_InCorpusCount += 1;
                sequenceWords.Add(word);
            }
            if (sequenceWords.Count > 2)
            {
                Cortex.Memory cortexMemory = new Cortex.Memory(Constants)
                {
                    Words = sequenceWords.ToArray()
                };
                for (int i = 0; i < sequenceWords.Count; i += 1)
                {
                    TensorPrimitives.Add(cortexMemory.DiscreteRandomVector, sequenceWords[i].DiscreteRandomVector, cortexMemory.DiscreteRandomVector);
                }
                TensorPrimitives.Min(cortexMemory.DiscreteRandomVector, 1.0f, cortexMemory.DiscreteRandomVector);
                cortexMemories.Add(cortexMemory);
            }
        }
        foreach (var kvp in dictionary)
        {
            kvp.Value.CorpusFreq = (float)kvp.Value.Temp_InCorpusCount / corpus_WordsCount;
        }
        return inputCorpusData;
    }

    #endregion

    #region private fields

    private readonly ILoggersSet _loggersSet;

    #endregion    

    public class ModelConstants
    {
        public int DiscreteVectorLength => 300;

        /// <summary>
        ///     Количество миниколонок в зоне коры по оси X
        /// </summary>
        public int CortexWidth_MiniColumns => 17;

        /// <summary>
        ///     Количество миниколонок в зоне коры по оси Y
        /// </summary>
        public int CortexHeight_MiniColumns => 17;

        public double SuperActivityRadius_MiniColumns => 5;

        /// <summary>
        ///     Нулевой уровень косинусного расстояния
        /// </summary>
        public float K0 { get; set; } = 0.2f;

        public float K1 { get; set; } = 0.2f;

        /// <summary>
        ///     Косинусное расстояние для пустой колонки
        /// </summary>
        public float K2 { get; set; } = 0.96f;

        public float K3 { get; set; } = 0.2f;

        public float K4 { get; set; } = 0.2f;

        public float[] PositiveK { get; set; } = [1.00f, 0.13f, 0.065f, 0.00f];

        public float[] NegativeK { get; set; } = [1.00f, 0.13f, 0.08f, 0.00f];
    }
}


public class InputCorpusData
{
    public Dictionary<string, Word> Dictionary = new(10000);

    public List<Cortex.Memory> CortexMemories = new(10000);
}