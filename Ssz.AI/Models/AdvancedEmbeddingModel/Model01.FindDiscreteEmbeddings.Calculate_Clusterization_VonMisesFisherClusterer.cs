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

            loggersSet.UserFriendlyLogger.LogInformation("=== Демонстрация von Mises-Fisher Кластеризации ===");

            // Устанавливаем случайное семя для воспроизводимости
            torch.manual_seed(42);

            // Генерируем тестовые данные (нормированные эмбеддинги слов)
            var testData = GenerateTestData();

            loggersSet.UserFriendlyLogger.LogInformation($"Сгенерированы тестовые данные: {testData.shape[0]} точек, размерность {testData.shape[1]}");

            // Создаём и обучаем кластеризатор
            var clusterer = new VonMisesFisherClusterer(
                loggersSet.UserFriendlyLogger,
                numClusters: 300,
                maxIterations: 50,
                tolerance: 1e-6,
                useHardAssignment: false
            );

            loggersSet.UserFriendlyLogger.LogInformation("\nНачинаем обучение...");
            clusterer.Fit(testData);

            // Выводим результаты
            clusterer.PrintModelSummary();

            // Демонстрируем предсказание
            loggersSet.UserFriendlyLogger.LogInformation("\n=== Тестирование предсказаний ===");
            var predictions = clusterer.Predict(testData.slice(0, 0, 10, 1)); // Первые 10 точек
            var probabilities = clusterer.PredictProba(testData.slice(0, 0, 10, 1));

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