using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using System.Threading.Tasks;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core;
using Microsoft.Extensions.Logging;
using Ssz.AI.Models;
using TorchSharp;
using static TorchSharp.torch;

namespace CrossLingualEmbeddings.Utils
{
    /// <summary>
    /// Утилитарные функции для работы с эмбеддингами и тензорами
    /// Аналог utils.py из Python проекта с использованием высокопроизводительных операций
    /// </summary>
    public static class EmbeddingUtils
    {
        #region Constants
        
        /// <summary>
        /// Размер батча по умолчанию для операций с большими тензорами
        /// </summary>
        private const int DefaultBatchSize = 4096;
        
        /// <summary>
        /// Минимальная норма вектора для избежания деления на ноль
        /// </summary>
        private const float MinNorm = 1e-8f;
        
        #endregion

        #region Embedding Loading and Processing

        /// <summary>
        /// Загружает предобученные эмбеддинги из текстового файла
        /// Использует высокопроизводительную обработку с System.Numerics.Tensors
        /// </summary>
        /// <param name="filePath">Путь к файлу с эмбеддингами</param>
        /// <param name="maxVocab">Максимальный размер словаря (-1 для неограниченного)</param>
        /// <param name="embeddingDim">Размерность эмбеддингов</param>
        /// <param name="toLowerCase">Приводить ли слова к нижнему регистру</param>
        /// <param name="logger">Логгер для отслеживания процесса</param>
        /// <returns>Кортеж из словаря и матрицы эмбеддингов</returns>
        /// <exception cref="FileNotFoundException">Если файл не найден</exception>
        /// <exception cref="ArgumentException">Если параметры некорректны</exception>
        public static async Task<(Dictionary dictionary, MatrixFloat embeddings)> LoadTextEmbeddingsAsync(
            string filePath, 
            int maxVocab = -1, 
            int embeddingDim = 300, 
            bool toLowerCase = true,
            ILogger? logger = null)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"Файл эмбеддингов не найден: {filePath}");

            if (embeddingDim <= 0)
                throw new ArgumentException("Размерность эмбеддингов должна быть положительной", nameof(embeddingDim));

            logger?.LogInformation($"Загрузка эмбеддингов из файла: {filePath}");
            
            var word2Id = new System.Collections.Generic.Dictionary<string, int>();
            var embeddings = new List<float[]>();
            var startTime = DateTime.UtcNow;

            using var reader = new StreamReader(filePath, Encoding.UTF8);
            
            // Читаем заголовок файла (если есть)
            var firstLine = await reader.ReadLineAsync();
            var isFirstLineHeader = false;
            
            if (firstLine != null && firstLine.Split(' ').Length == 2)
            {
                var parts = firstLine.Split(' ');
                if (int.TryParse(parts[0], out _) && int.TryParse(parts[1], out int dimFromFile))
                {
                    if (dimFromFile != embeddingDim)
                    {
                        logger?.LogWarning($"Размерность в файле ({dimFromFile}) не совпадает с ожидаемой ({embeddingDim})");
                    }
                    isFirstLineHeader = true;
                }
            }

            // Если первая строка не заголовок, обрабатываем её как эмбеддинг
            if (!isFirstLineHeader && firstLine != null)
            {
                ProcessEmbeddingLine(firstLine, word2Id, embeddings, embeddingDim, toLowerCase, logger);
            }

            // Обрабатываем остальные строки
            string? line;
            int processedLines = 0;
            
            while ((line = await reader.ReadLineAsync()) != null && 
                   (maxVocab == -1 || word2Id.Count < maxVocab))
            {
                ProcessEmbeddingLine(line, word2Id, embeddings, embeddingDim, toLowerCase, logger);
                processedLines++;
                
                // Логируем прогресс каждые 10000 строк
                if (processedLines % 10000 == 0)
                {
                    logger?.LogProgress(processedLines, maxVocab == -1 ? -1 : maxVocab, "Загрузка эмбеддингов");
                }
            }

            logger?.LogInformation($"Загружено {word2Id.Count} предобученных эмбеддингов слов");
            
            // Создаем словарь
            var id2Word = word2Id.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
            var sortedId2Word = new SortedDictionary<int, string>(id2Word);
            var dictionary = new Dictionary(sortedId2Word, word2Id, "unknown");
            
            // Конвертируем эмбеддинги в MatrixFloat для оптимальной производительности
            var embeddingMatrix = CreateMatrixFromEmbeddings(embeddings, embeddingDim);
            
            var elapsedTime = DateTime.UtcNow - startTime;
            logger?.LogPerformance(elapsedTime, word2Id.Count, "Загрузка эмбеддингов");
            
            return (dictionary, embeddingMatrix);
        }

        /// <summary>
        /// Обрабатывает одну строку файла эмбеддингов
        /// </summary>
        /// <param name="line">Строка для обработки</param>
        /// <param name="word2Id">Словарь слово -> индекс</param>
        /// <param name="embeddings">Список эмбеддингов</param>
        /// <param name="expectedDim">Ожидаемая размерность</param>
        /// <param name="toLowerCase">Приводить к нижнему регистру</param>
        /// <param name="logger">Логгер</param>
        private static void ProcessEmbeddingLine(
            string line, 
            System.Collections.Generic.Dictionary<string, int> word2Id, 
            List<float[]> embeddings, 
            int expectedDim, 
            bool toLowerCase, 
            ILogger? logger)
        {
            if (string.IsNullOrWhiteSpace(line)) return;

            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < expectedDim + 1) return;

            var word = parts[0];
            if (toLowerCase) word = word.ToLowerInvariant();

            // Проверяем, не встречалось ли это слово раньше
            if (word2Id.ContainsKey(word))
            {
                logger?.LogWarning($"Слово '{word}' найдено дважды в файле эмбеддингов");
                return;
            }

            // Парсим числовые значения эмбеддинга
            var embedding = new float[expectedDim];
            bool isValidEmbedding = true;
            
            for (int i = 0; i < expectedDim; i++)
            {
                if (!float.TryParse(parts[i + 1], out embedding[i]))
                {
                    isValidEmbedding = false;
                    break;
                }
            }

            if (!isValidEmbedding)
            {
                logger?.LogWarning($"Некорректные числовые значения для слова '{word}'");
                return;
            }

            // Проверяем на нулевой вектор и заменяем при необходимости
            if (IsZeroVector(embedding))
            {
                embedding[0] = 0.01f; // Устанавливаем небольшое значение для избежания нулевого вектора
            }

            word2Id[word] = word2Id.Count;
            embeddings.Add(embedding);
        }

        /// <summary>
        /// Проверяет, является ли вектор нулевым
        /// </summary>
        /// <param name="vector">Вектор для проверки</param>
        /// <returns>true, если вектор нулевой</returns>
        private static bool IsZeroVector(ReadOnlySpan<float> vector)
        {
            return TensorPrimitives.Norm(vector) < MinNorm;
        }

        /// <summary>
        /// Создает MatrixFloat из списка эмбеддингов для оптимальной производительности
        /// </summary>
        /// <param name="embeddings">Список эмбеддингов</param>
        /// <param name="embeddingDim">Размерность эмбеддингов</param>
        /// <returns>Матрица эмбеддингов</returns>
        private static MatrixFloat CreateMatrixFromEmbeddings(List<float[]> embeddings, int embeddingDim)
        {
            var matrix = new MatrixFloat_RowMajor(embeddings.Count, embeddingDim);
            
            // Используем параллельное копирование для лучшей производительности
            Parallel.For(0, embeddings.Count, i =>
            {
                var sourceSpan = new ReadOnlySpan<float>(embeddings[i]);
                var targetSpan = matrix.GetRow(i);
                sourceSpan.CopyTo(targetSpan);
            });
            
            return matrix;
        }

        #endregion

        #region Normalization

        /// <summary>
        /// Нормализует эмбеддинги различными методами с использованием высокопроизводительных операций
        /// </summary>
        /// <param name="embeddings">Матрица эмбеддингов</param>
        /// <param name="normalizationTypes">Типы нормализации, разделенные запятыми (center, renorm)</param>
        /// <param name="mean">Предыдущее среднее значение (для центрирования)</param>
        /// <returns>Среднее значение после нормализации (если применялось центрирование)</returns>
        public static MatrixFloat? NormalizeEmbeddings(MatrixFloat embeddings, string normalizationTypes, MatrixFloat? mean = null)
        {
            if (string.IsNullOrWhiteSpace(normalizationTypes))
                return mean;

            var types = normalizationTypes.Split(',', StringSplitOptions.RemoveEmptyEntries)
                                        .Select(t => t.Trim().ToLowerInvariant())
                                        .ToArray();

            MatrixFloat? resultMean = mean;

            foreach (var type in types)
            {
                switch (type)
                {
                    case "center":
                        resultMean = CenterEmbeddings(embeddings, mean);
                        break;
                    case "renorm":
                        RenormalizeEmbeddings(embeddings);
                        break;
                    default:
                        throw new ArgumentException($"Неизвестный тип нормализации: {type}");
                }
            }

            return resultMean;
        }

        /// <summary>
        /// Центрирует эмбеддинги (вычитает среднее) с использованием TensorPrimitives
        /// </summary>
        /// <param name="embeddings">Матрица эмбеддингов</param>
        /// <param name="mean">Предыдущее среднее (если есть)</param>
        /// <returns>Среднее значение</returns>
        private static MatrixFloat CenterEmbeddings(MatrixFloat embeddings, MatrixFloat? mean = null)
        {
            var embeddingDim = embeddings.ColumnsCount;
            var vocabSize = embeddings.RowsCount;
            
            MatrixFloat meanVector;
            
            if (mean == null)
            {
                // Вычисляем среднее значение для каждой размерности
                meanVector = new MatrixFloat_RowMajor(1, embeddingDim);
                
                // Используем параллельные вычисления для каждой размерности
                Parallel.For(0, embeddingDim, dim =>
                {
                    var column = embeddings.GetColumn(dim);
                    var sum = TensorPrimitives.Sum(column);
                    meanVector[0, dim] = sum / vocabSize;
                });
            }
            else
            {
                meanVector = mean;
            }
            
            // Вычитаем среднее из каждого эмбеддинга
            Parallel.For(0, vocabSize, i =>
            {
                var row = embeddings.GetRow(i);
                var meanRow = meanVector.GetRow(0);
                
                // Используем высокопроизводительное вычитание векторов
                TensorPrimitives.Subtract(row, meanRow, row);
            });
            
            return meanVector;
        }

        /// <summary>
        /// Ренормализует эмбеддинги (приводит норму к единице) с использованием TensorPrimitives
        /// </summary>
        /// <param name="embeddings">Матрица эмбеддингов</param>
        private static void RenormalizeEmbeddings(MatrixFloat embeddings)
        {
            var vocabSize = embeddings.RowsCount;
            
            // Нормализуем каждый эмбеддинг параллельно
            Parallel.For(0, vocabSize, i =>
            {
                var row = embeddings.GetRow(i);
                var norm = TensorPrimitives.Norm(row);
                
                // Избегаем деления на ноль
                if (norm > MinNorm)
                {
                    // Нормализуем вектор
                    TensorPrimitives.Divide(row, norm, row);
                }
                else
                {
                    // Если вектор нулевой, заменяем его малым значением
                    row.Fill(0.01f / (float)Math.Sqrt(row.Length));
                }
            });
        }

        #endregion

        #region Nearest Neighbors and Distance Calculations

        /// <summary>
        /// Вычисляет среднее расстояние до k ближайших соседей с использованием высокопроизводительных операций
        /// </summary>
        /// <param name="embeddings">Матрица эмбеддингов для поиска</param>
        /// <param name="queries">Матрица запросов</param>
        /// <param name="k">Количество ближайших соседей</param>
        /// <returns>Массив средних расстояний</returns>
        public static float[] GetNearestNeighborAverageDistances(MatrixFloat embeddings, MatrixFloat queries, int k)
        {
            var queryCount = queries.RowsCount;
            var embeddingCount = embeddings.RowsCount;
            var dimension = embeddings.ColumnsCount;
            
            if (queries.ColumnsCount != dimension)
                throw new ArgumentException("Размерности эмбеддингов и запросов должны совпадать");

            var averageDistances = new float[queryCount];
            
            // Обрабатываем запросы батчами для оптимизации памяти
            var batchSize = Math.Min(DefaultBatchSize, queryCount);
            
            Parallel.For(0, (queryCount + batchSize - 1) / batchSize, batchIdx =>
            {
                var startIdx = batchIdx * batchSize;
                var endIdx = Math.Min(startIdx + batchSize, queryCount);
                
                for (int queryIdx = startIdx; queryIdx < endIdx; queryIdx++)
                {
                    var queryVector = queries.GetRow(queryIdx);
                    var distances = new float[embeddingCount];
                    
                    // Вычисляем косинусные расстояния ко всем эмбеддингам
                    for (int embIdx = 0; embIdx < embeddingCount; embIdx++)
                    {
                        var embeddingVector = embeddings.GetRow(embIdx);
                        distances[embIdx] = TensorPrimitives.CosineSimilarity(queryVector, embeddingVector);
                    }
                    
                    // Находим k лучших расстояний
                    Array.Sort(distances);
                    Array.Reverse(distances); // Сортируем по убыванию для косинусного сходства
                    
                    // Вычисляем среднее расстояние для k ближайших соседей
                    var topK = distances.AsSpan(0, Math.Min(k, distances.Length));
                    averageDistances[queryIdx] = TensorPrimitives.Sum(topK) / topK.Length;
                }
            });
            
            return averageDistances;
        }

        /// <summary>
        /// Вычисляет матрицу косинусных сходств между двумя наборами эмбеддингов
        /// </summary>
        /// <param name="embeddings1">Первый набор эмбеддингов</param>
        /// <param name="embeddings2">Второй набор эмбеддингов</param>
        /// <returns>Матрица сходств</returns>
        public static MatrixFloat ComputeCosineSimilarityMatrix(MatrixFloat embeddings1, MatrixFloat embeddings2)
        {
            var count1 = embeddings1.RowsCount;
            var count2 = embeddings2.RowsCount;
            var dimension = embeddings1.ColumnsCount;
            
            if (embeddings2.ColumnsCount != dimension)
                throw new ArgumentException("Размерности эмбеддингов должны совпадать");
            
            var similarityMatrix = new MatrixFloat_RowMajor(count1, count2);
            
            // Параллельно вычисляем сходства
            Parallel.For(0, count1, i =>
            {
                var vector1 = embeddings1.GetRow(i);
                var similarities = similarityMatrix.GetRow(i);
                
                for (int j = 0; j < count2; j++)
                {
                    var vector2 = embeddings2.GetRow(j);
                    similarities[j] = TensorPrimitives.CosineSimilarity(vector1, vector2);
                }
            });
            
            return similarityMatrix;
        }

        #endregion

        #region Export and Import

        /// <summary>
        /// Экспортирует эмбеддинги в текстовый файл
        /// </summary>
        /// <param name="dictionary">Словарь</param>
        /// <param name="embeddings">Матрица эмбеддингов</param>
        /// <param name="filePath">Путь к файлу для экспорта</param>
        /// <param name="logger">Логгер</param>
        public static async Task ExportEmbeddingsAsync(Dictionary dictionary, MatrixFloat embeddings, string filePath, ILogger? logger = null)
        {
            logger?.LogInformation($"Экспорт эмбеддингов в файл: {filePath}");
            
            using var writer = new StreamWriter(filePath, false, Encoding.UTF8);
            
            // Записываем заголовок
            await writer.WriteLineAsync($"{dictionary.Count} {embeddings.ColumnsCount}");
            
            // Записываем каждый эмбеддинг
            for (int i = 0; i < dictionary.Count; i++)
            {
                var word = dictionary[i];
                var embedding = embeddings.GetRow(i);
                
                var line = new StringBuilder(word);
                foreach (var value in embedding)
                {
                    line.Append($" {value:F5}");
                }
                
                await writer.WriteLineAsync(line.ToString());
            }
            
            logger?.LogInformation($"Экспорт завершен: {dictionary.Count} эмбеддингов");
        }

        #endregion

        #region Utility Functions

        /// <summary>
        /// Создает случайную ортогональную матрицу для инициализации
        /// </summary>
        /// <param name="size">Размер матрицы</param>
        /// <param name="seed">Seed для генератора случайных чисел</param>
        /// <returns>Ортогональная матрица</returns>
        public static MatrixFloat CreateOrthogonalMatrix(int size, int? seed = null)
        {
            var random = seed.HasValue ? new Random(seed.Value) : new Random();
            var matrix = new MatrixFloat_RowMajor(size, size);
            
            // Генерируем случайную матрицу
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    matrix[i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
                }
            }
            
            // Применяем QR разложение для получения ортогональной матрицы
            // Для простоты используем процедуру Грама-Шмидта
            OrthogonalizeMatrix(matrix);
            
            return matrix;
        }

        /// <summary>
        /// Ортогонализует матрицу с использованием процедуры Грама-Шмидта
        /// </summary>
        /// <param name="matrix">Матрица для ортогонализации</param>
        private static void OrthogonalizeMatrix(MatrixFloat matrix)
        {
            var size = matrix.RowsCount;
            
            for (int i = 0; i < size; i++)
            {
                var currentRow = matrix.GetRow(i);
                
                // Ортогонализуем относительно всех предыдущих векторов
                for (int j = 0; j < i; j++)
                {
                    var previousRow = matrix.GetRow(j);
                    var projection = TensorPrimitives.Dot(currentRow, previousRow);
                    
                    // Вычитаем проекцию: current = current - projection * previous
                    var scaledPrevious = new float[size];
                    TensorPrimitives.Multiply(previousRow, projection, scaledPrevious);
                    TensorPrimitives.Subtract(currentRow, scaledPrevious, currentRow);
                }
                
                // Нормализуем вектор
                var norm = TensorPrimitives.Norm(currentRow);
                if (norm > MinNorm)
                {
                    TensorPrimitives.Divide(currentRow, norm, currentRow);
                }
            }
        }

        /// <summary>
        /// Проверяет валидность матрицы эмбеддингов
        /// </summary>
        /// <param name="embeddings">Матрица эмбеддингов</param>
        /// <param name="checkForNaN">Проверять ли на NaN значения</param>
        /// <param name="checkForInfinity">Проверять ли на бесконечные значения</param>
        /// <returns>true, если матрица валидна</returns>
        public static bool ValidateEmbeddings(MatrixFloat embeddings, bool checkForNaN = true, bool checkForInfinity = true)
        {
            var data = embeddings.Data;
            
            for (int i = 0; i < data.Length; i++)
            {
                var value = data[i];
                
                if (checkForNaN && float.IsNaN(value))
                    return false;
                
                if (checkForInfinity && float.IsInfinity(value))
                    return false;
            }
            
            return true;
        }

        #endregion
    }
}