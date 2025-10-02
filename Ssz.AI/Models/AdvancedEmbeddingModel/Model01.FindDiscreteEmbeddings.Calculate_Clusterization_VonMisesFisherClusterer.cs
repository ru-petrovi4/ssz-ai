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
using Microsoft.Extensions.Logging;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using TorchSharp;
using static TorchSharp.torch;

namespace Ssz.AI.Models.AdvancedEmbeddingModel
{    
    public partial class Model01
    {
        public void Calculate_Clusterization_AlgorithmData_VonMisesFisherClusterer(LanguageInfo languageInfo, ILoggersSet loggersSet)
        {
            var words = languageInfo.Words;

            var clusterization_AlgorithmData = new Clusterization_AlgorithmData(languageInfo, name: "VonMisesFisherClusterer");
            clusterization_AlgorithmData.GenerateOwnedData(Constants.PrimaryWordsCount);
            languageInfo.Clusterization_AlgorithmData = clusterization_AlgorithmData;

            var totalStopwatch = Stopwatch.StartNew();

            int rows = words.Count;
            int cols = words[0].OldVectorNormalized.Length;

            // Объединить все данные в один float[] (в строковом порядке)
            float[] flat = new float[rows * cols];
            for (int i = 0; i < rows; ++i)
            {
                Array.Copy(words[i].OldVectorNormalized, 0, flat, i * cols, cols);
            }            
            var oldVectorsTensor = torch.tensor(flat, new long[] { rows, cols }, ScalarType.Float32);

            loggersSet.UserFriendlyLogger.LogInformation("=== von Mises-Fisher Кластеризация ===");

            // Устанавливаем случайное семя для воспроизводимости
            torch.manual_seed(42);

            loggersSet.UserFriendlyLogger.LogInformation($"Сгенерированы тестовые данные: {oldVectorsTensor.shape[0]} точек, размерность {oldVectorsTensor.shape[1]}");

            // Создаём и обучаем кластеризатор
            var clusterer = new VonMisesFisherClusterer(
                loggersSet.UserFriendlyLogger,
                numClusters: 300,
                maxIterations: 50,
                tolerance: 1e-6,
                useHardAssignment: false
            );

            loggersSet.UserFriendlyLogger.LogInformation("\nНачинаем обучение...");
            clusterer.Fit(oldVectorsTensor);

            // Выводим результаты
            clusterer.PrintModelSummary();

            // Демонстрируем предсказание
            loggersSet.UserFriendlyLogger.LogInformation("\n=== Тестирование предсказаний ===");
            var predictions = clusterer.Predict(oldVectorsTensor.slice(0, 0, 10, 1)); // Первые 10 точек
            var probabilities = clusterer.PredictProba(oldVectorsTensor.slice(0, 0, 10, 1));

            loggersSet.UserFriendlyLogger.LogInformation("Предсказания для первых 10 точек:");
            for (int i = 0; i < 10; i++)
            {
                var pred = predictions[i].item<long>();
                var prob = probabilities[i, pred].item<double>();
                loggersSet.UserFriendlyLogger.LogInformation($"Точка {i}: Кластер {pred} (вероятность: {prob:F3})");
            }

            totalStopwatch.Stop();
            loggersSet.UserFriendlyLogger.LogInformation("Calculate Clusterization VonMisesFisherClusterer totally done. Elapsed Milliseconds = " + totalStopwatch.ElapsedMilliseconds);
        }
    }    
}