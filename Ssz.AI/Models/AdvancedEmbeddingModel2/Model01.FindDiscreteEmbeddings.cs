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
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Ssz.AI.Core;
using Ssz.AI.Helpers;
using Ssz.AI.ViewModels;
using Ssz.Utils;
using Ssz.Utils.Addons;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;

namespace Ssz.AI.Models.AdvancedEmbeddingModel2;

public class Model01
{
    public const string AdvancedEmbedding2_Directory = "Ssz.AI.AdvancedEmbedding2";

    public const string Input_Directory = "input";

    public const string FileName_Cortex = "AdvancedEmbedding2_Cortex.bin";

    #region construction and destruction

    public Model01()
    {
        LoggersSet = new LoggersSet(
            NullLogger.Instance,
            new WrapperUserFriendlyLogger(
                new SszLogger("Ssz.AI.Models.AdvancedEmbeddingModel2.Model01", "Ssz.AI.Models.AdvancedEmbeddingModel2.Model01", new SszLoggerOptions()
                {
                    LogsDirectory = "Data",
#if DEBUG
                    LogFileName = "AdvancedEmbeddingModel2_Model01_Logs_Debug.txt"
#else
                    LogFileName = "AdvancedEmbeddingModel2_Model01_Logs.txt"
#endif
                }),
            new UserFriendlyLogger((l, id, s) => DebugWindow.Instance.AddLine(s))));
    }

    #endregion

    #region public functions       

    public ILoggersSet LoggersSet { get; }

    public static readonly ModelConstants Constants = new();

    public InputCorpusData InputCorpusData = null!;    

    public Cortex Cortex = null!;    

    public void StemInputText()
    {
        List<List<string>> allSequences = new(10000);

        int i = 0;
        foreach (var fb2FileFullName in Directory.GetFiles(Path.Combine(AIConstants.DataDirectory, AdvancedEmbedding2_Directory, Input_Directory), @"*.fb2"))
        {
            i += 1;            
            var fb2FileNewFullName = Path.Combine(AIConstants.DataDirectory, AdvancedEmbedding2_Directory, Input_Directory, $"{i}.fb2");
            File.Copy(fb2FileFullName, fb2FileNewFullName, true);
            var fb2FileName = Path.GetFileName(fb2FileNewFullName);
            var fb2_Stemmed_FileName = fb2FileName + "_stemmed.txt";
            // Настройка информации о процессе
            ProcessStartInfo startInfo = new ProcessStartInfo
            {
                WorkingDirectory = Path.Combine(AIConstants.DataDirectory, AdvancedEmbedding2_Directory, Input_Directory),
                FileName = Path.Combine(AIConstants.DataDirectory, AdvancedEmbedding2_Directory, "mystem.exe"),  // Или путь к исполняемому файлу, например, @"C:\path\to\program.exe"
                Arguments = $"-c -d -i {fb2FileName} {fb2_Stemmed_FileName}",  // Аргументы, если нужны, например, "file.txt"
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
                LoggersSet.LoggerAndUserFriendlyLogger.LogInformation($"Процесс mystem.exe завершён с кодом: {exitCode}");

                // Если захватывали вывод
                string output = process.StandardOutput.ReadToEnd();
                string error = process.StandardError.ReadToEnd();
                if (!string.IsNullOrEmpty(output))
                    LoggersSet.LoggerAndUserFriendlyLogger.LogInformation($"Вывод: {output}");
                if (!string.IsNullOrEmpty(error))
                    LoggersSet.LoggerAndUserFriendlyLogger.LogInformation($"Ошибка: {error}");
            }

            var sequences = MorphologicalTextParser.ParseTextToSequences(Path.Combine(AIConstants.DataDirectory, AdvancedEmbedding2_Directory, Input_Directory, fb2_Stemmed_FileName));
            allSequences.AddRange(sequences);
        }

        MorphologicalTextParser.WriteToFile(
            allSequences,
            Path.Combine(AIConstants.DataDirectory, AdvancedEmbedding2_Directory, "input_sequences.txt"),
            LoggersSet.LoggerAndUserFriendlyLogger);
    }

    public bool Calculate_PutPhrases_BasedOnSuperActivity(int cortexMemoriesCount, Random random)
    {
        return Cortex.Calculate_PutPhrases_BasedOnSuperActivity(InputCorpusData, cortexMemoriesCount, random);
    }

    public async Task ReorderPhrases1Epoch_BasedOnSuperActivityAsync(int epochCount, Random random, CancellationToken cancellationToken, Func<Task>? epochRefreshAction = null)
    {
        await Cortex.Calculate_ReorderCortexMemories_BasedOnSuperActivityAsync(epochCount, random, cancellationToken, epochRefreshAction);
    }

    public bool Calculate_PutPhrases_Randomly(int cortexMemoriesCount, Random random)
    {
        return Cortex.Calculate_PutPhrases_Randomly(InputCorpusData, cortexMemoriesCount, random);
    }    

    public VisualizationWithDesc[] GetImageWithDescs()
    {
        var bitmapFromMiniColums_ActivityColor = Visualisation.GetBitmapFromMiniColums_ActivityColor(Cortex);
        var bitmapFromMiniColums_SuperActivityColor = Visualisation.GetBitmapFromMiniColums_SuperActivityColor(Cortex, null);
        
        var bitmapFromMiniColums_ActivityColor_WordCode = Visualisation.GetBitmapFromMiniColums_Activity_Code(Cortex);
        var bitmapFromMiniColums_SuperActivityColor_WordCode = Visualisation.GetBitmapFromMiniColums_SuperActivity_Code(Cortex);

        return [                      
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(bitmapFromMiniColums_ActivityColor),
                    Desc = @"Активность миниколонок (белый - максимум)" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(bitmapFromMiniColums_ActivityColor_WordCode),
                    Desc = @"Активность миниколонок, код слова" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(bitmapFromMiniColums_SuperActivityColor),
                    Desc = @"Суперактивность миниколонок (белый - максимум)" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(bitmapFromMiniColums_SuperActivityColor_WordCode),
                    Desc = @"Суперактивность миниколонок, код слова" },
                new Model3DWithDesc { Data = Visualization3D.Get_MiniColumnsMemories_Model3DScene(Cortex),
                    Desc = $"Накопленные воспоминания в миниколонках." },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Visualisation.GetBitmapFromMiniColumsMemoriesColor(Cortex)),
                    Desc = @"Средний цвет накопленных воспоминаний в миниколонках" },
                new ImageWithDesc { Image = BitmapHelper.ConvertImageToAvaloniaBitmap(Visualisation.GetBitmapFromMiniColumsMemoriesCount(Cortex)),
                    Desc = @"Количество воспоминаний в миниколонках" }
            ];
    }

    #endregion

    #region private functions    



    #endregion

    #region private fields    

    #endregion

    public class ModelConstants : IMiniColumnsActivityConstants
    {
        public int DiscreteVectorLength => 300;

        public int DiscreteOptimizedVector_PrimaryBitsCount => 7;

        /// <summary>
        ///     Количество миниколонок в зоне коры по оси X
        /// </summary>
        public int CortexWidth_MiniColumns => 17;

        /// <summary>
        ///     Количество миниколонок в зоне коры по оси Y
        /// </summary>
        public int CortexHeight_MiniColumns => 17;

        public double SuperActivityRadius_MiniColumns => 3;

        /// <summary>
        ///     Нулевой уровень косинусного подобия
        /// </summary>
        public float K0 { get; set; } = 0.13f; // 0.12

        public float K1 { get; set; } = 0.2f;

        /// <summary>
        ///     Косинусное подобие с пустой миниколонкой
        /// </summary>
        public float K2 { get; set; } = 0.96f;

        public float K3 { get; set; } = 0.2f;

        /// <summary>
        ///     Порог суперактивности
        /// </summary>
        public float K4 { get; set; } = 0.2f;

        public float[] PositiveK { get; set; } = [1.00f, 0.13f, 0.065f];

        public float[] NegativeK { get; set; } = [1.00f, 0.13f, 0.08f];

        /// <summary>
        ///     Включен ли порог на суперактивность при накоплении воспоминаний
        /// </summary>
        public bool SuperactivityThreshold { get; set; }
    }
}