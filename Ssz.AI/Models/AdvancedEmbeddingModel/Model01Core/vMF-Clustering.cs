using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;

/// <summary>
/// Реализация кластеризации von Mises-Fisher (vMF) на единичной сфере
/// Использует EM-алгоритм для оценки параметров смеси vMF распределений
/// </summary>
public class VonMisesFisherClusterer
{
    // Поля класса для хранения параметров модели
    private readonly int _numClusters;         // Количество кластеров K
    private readonly int _maxIterations;       // Максимальное количество итераций EM
    private readonly double _tolerance;        // Порог сходимости по логарифмической вероятности
    private readonly bool _useHardAssignment;  // Использовать жёсткое (hard) или мягкое (soft) назначение
        
    // Параметры модели - изучаются в процессе обучения
    public Tensor MeanDirections { get; private set; }    // μ_k - направления средних для каждого кластера [K x D]
    public Tensor Concentrations { get; private set; }    // κ_k - параметры концентрации для каждого кластера [K]
    public Tensor MixingCoefficients { get; private set; } // α_k - коэффициенты смешивания [K]
        
    // Логи процесса обучения
    public List<double> LogLikelihoodHistory { get; private set; }

    /// <summary>
    /// Конструктор кластеризатора vMF
    /// </summary>
    /// <param name="numClusters">Количество кластеров K</param>
    /// <param name="maxIterations">Максимальное количество итераций EM алгоритма</param>
    /// <param name="tolerance">Порог сходимости для логарифмической вероятности</param>
    /// <param name="useHardAssignment">Использовать жёсткое назначение (true) или мягкое (false)</param>
    public VonMisesFisherClusterer(int numClusters, int maxIterations = 100, double tolerance = 1e-6, bool useHardAssignment = false)
    {
        _numClusters = numClusters;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _useHardAssignment = useHardAssignment;
        LogLikelihoodHistory = new List<double>();
    }

    /// <summary>
    /// Основной метод обучения кластеризатора на нормированных данных
    /// </summary>
    /// <param name="data">Нормированные данные [N x D], где N - количество точек, D - размерность</param>
    public void Fit(Tensor data)
    {
        // Проверяем, что данные корректно нормированы
        ValidateNormalizedData(data);
            
        var (numSamples, dimension) = (data.shape[0], data.shape[1]);
            
        // Инициализируем параметры модели
        InitializeParameters(data, numSamples, dimension);
            
        double prevLogLikelihood = double.NegativeInfinity;
            
        // Основной цикл EM алгоритма
        for (int iteration = 0; iteration < _maxIterations; iteration++)
        {
            // E-шаг: вычисляем posterior probabilities p(k|x_i)
            var posteriors = ComputePosteriors(data);
                
            // M-шаг: обновляем параметры модели
            UpdateParameters(data, posteriors);
                
            // Вычисляем логарифмическую вероятность для проверки сходимости
            var logLikelihood = ComputeLogLikelihood(data);
            LogLikelihoodHistory.Add(logLikelihood);
                
            // Проверяем сходимость
            if (Math.Abs(logLikelihood - prevLogLikelihood) < _tolerance)
            {
                Console.WriteLine($"Сходимость достигнута на итерации {iteration + 1}");
                break;
            }
                
            prevLogLikelihood = logLikelihood;
                
            // Выводим прогресс каждые 10 итераций
            if ((iteration + 1) % 10 == 0)
            {
                Console.WriteLine($"Итерация {iteration + 1}: Log-Likelihood = {logLikelihood:F6}");
            }
        }
            
        Console.WriteLine($"Обучение завершено. Финальная Log-Likelihood: {prevLogLikelihood:F6}");
    }

    /// <summary>
    /// Проверяет, что входные данные правильно нормированы (лежат на единичной сфере)
    /// </summary>
    /// <param name="data">Данные для проверки</param>
    private void ValidateNormalizedData(Tensor data)
    {
        // Вычисляем нормы всех векторов
        var norms = torch.norm(data, dimension: 1);
        var minNorm = torch.min(norms).item<double>();
        var maxNorm = torch.max(norms).item<double>();
            
        // Проверяем, что все нормы приблизительно равны 1.0
        if (Math.Abs(minNorm - 1.0) > 1e-6 || Math.Abs(maxNorm - 1.0) > 1e-6)
        {
            throw new ArgumentException($"Данные должны быть нормированы на единичную сферу. " +
                $"Найдены нормы в диапазоне [{minNorm:F8}, {maxNorm:F8}]");
        }
    }

    /// <summary>
    /// Инициализирует параметры модели перед началом EM алгоритма
    /// Использует kmeans++ подобную инициализацию для выбора начальных центров
    /// </summary>
    /// <param name="data">Входные данные</param>
    /// <param name="numSamples">Количество образцов</param>
    /// <param name="dimension">Размерность данных</param>
    private void InitializeParameters(Tensor data, long numSamples, long dimension)
    {
        // Инициализация коэффициентов смешивания (равномерное распределение)
        MixingCoefficients = torch.ones(_numClusters, dtype: torch.float32) / _numClusters;
            
        // Инициализация направлений средних с помощью kmeans++ подобного метода
        MeanDirections = torch.zeros(new long[] { _numClusters, dimension }, dtype: torch.float32);
            
        // Выбираем первый центр случайно
        var randomIndex = torch.randint(0, numSamples, new long[] { 1 }).item<long>();
        MeanDirections[0] = data[randomIndex];
            
        // Выбираем остальные центры с вероятностью пропорциональной расстоянию до ближайшего центра
        for (int k = 1; k < _numClusters; k++)
        {
            var distances = torch.zeros(numSamples);
                
            // Для каждой точки данных находим расстояние до ближайшего уже выбранного центра
            for (long i = 0; i < numSamples; i++)
            {
                var minDistance = double.MaxValue;
                    
                // Вычисляем минимальное расстояние до уже выбранных центров
                for (int j = 0; j < k; j++)
                {
                    // Используем угловое расстояние: 1 - cosine_similarity
                    var cosineSimilarity = torch.dot(data[i], MeanDirections[j]).item<double>();
                    var angularDistance = 1.0 - cosineSimilarity;
                    minDistance = Math.Min(minDistance, angularDistance);
                }
                    
                distances[i] = minDistance;
            }
                
            // Выбираем следующий центр с вероятностью пропорциональной квадрату расстояния
            var probabilities = torch.pow(distances, 2);
            probabilities = probabilities / torch.sum(probabilities);
                
            // Используем multinomial sampling для выбора индекса
            var selectedIndex = torch.multinomial(probabilities, 1).item<long>();
            MeanDirections[k] = data[selectedIndex];
        }
            
        // Инициализация параметров концентрации
        // Начинаем с умеренных значений концентрации
        Concentrations = torch.ones(_numClusters, dtype: torch.float32) * 1.0f;
            
        Console.WriteLine("Параметры инициализированы:");
        Console.WriteLine($"Количество кластеров: {_numClusters}");
        Console.WriteLine($"Размерность данных: {dimension}");
        Console.WriteLine($"Количество образцов: {numSamples}");
    }

    /// <summary>
    /// E-шаг EM алгоритма: вычисляет posterior probabilities p(k|x_i)
    /// </summary>
    /// <param name="data">Входные данные [N x D]</param>
    /// <returns>Матрица posterior probabilities [N x K]</returns>
    private Tensor ComputePosteriors(Tensor data)
    {
        var numSamples = data.shape[0];
        var posteriors = torch.zeros(new long[] { numSamples, _numClusters });
            
        // Вычисляем логарифмические вероятности для численной стабильности
        var logProbabilities = torch.zeros(new long[] { numSamples, _numClusters });
            
        for (int k = 0; k < _numClusters; k++)
        {
            // Вычисляем cosine similarity между данными и k-м центром
            var cosineSimilarities = torch.matmul(data, MeanDirections[k]);
                
            // Вычисляем логарифм нормализующей константы c_d(κ)
            var logNormalizingConstant = ComputeLogNormalizingConstant(
                Concentrations[k].item<double>(), 
                data.shape[1]
            );
                
            // Логарифм vMF плотности: log c_d(κ) + κ * μ^T * x
            logProbabilities[:, k] = logNormalizingConstant + 
                Concentrations[k] * cosineSimilarities + 
                torch.log(MixingCoefficients[k]);
        }
            
        if (_useHardAssignment)
        {
            // Жёсткое назначение: назначаем каждую точку кластеру с максимальной вероятностью
            var assignments = torch.argmax(logProbabilities, dim: 1);
            for (long i = 0; i < numSamples; i++)
            {
                var assignedCluster = assignments[i].item<long>();
                posteriors[i, assignedCluster] = 1.0;
            }
        }
        else
        {
            // Мягкое назначение: используем softmax для получения вероятностей
            posteriors = torch.softmax(logProbabilities, dim: 1);
        }
            
        return posteriors;
    }

    /// <summary>
    /// M-шаг EM алгоритма: обновляет параметры модели на основе posterior probabilities
    /// </summary>
    /// <param name="data">Входные данные [N x D]</param>
    /// <param name="posteriors">Posterior probabilities [N x K]</param>
    private void UpdateParameters(Tensor data, Tensor posteriors)
    {
        var numSamples = data.shape[0];
            
        for (int k = 0; k < _numClusters; k++)
        {
            // Обновляем коэффициент смешивания α_k
            var effectiveSampleSize = torch.sum(posteriors[:, k]);
            MixingCoefficients[k] = effectiveSampleSize / numSamples;
                
            // Вычисляем взвешенную сумму точек для кластера k
            var weightedSum = torch.zeros_like(MeanDirections[k]);
            for (long i = 0; i < numSamples; i++)
            {
                weightedSum += posteriors[i, k] * data[i];
            }
                
            // Обновляем направление среднего μ_k
            var resultantLength = torch.norm(weightedSum);
            MeanDirections[k] = weightedSum / resultantLength;
                
            // Вычисляем средний resultant length для оценки κ_k
            var meanResultantLength = (resultantLength / effectiveSampleSize).item<double>();
                
            // Обновляем параметр концентрации κ_k используя аппроксимацию из статьи
            var newConcentration = ApproximateConcentration(meanResultantLength, data.shape[1]);
            Concentrations[k] = (float)newConcentration;
        }
            
        // Нормализуем коэффициенты смешивания для обеспечения того, что их сумма равна 1
        MixingCoefficients = MixingCoefficients / torch.sum(MixingCoefficients);
    }

    /// <summary>
    /// Вычисляет приближённое значение параметра концентрации κ
    /// Использует аппроксимацию из статьи: κ ≈ r̄d - r̄³ / (1 - r̄²)
    /// где r̄ - средний resultant length
    /// </summary>
    /// <param name="meanResultantLength">Средний resultant length r̄</param>
    /// <param name="dimension">Размерность данных d</param>
    /// <returns>Приближённое значение κ</returns>
    private double ApproximateConcentration(double meanResultantLength, long dimension)
    {
        // Ограничиваем r̄ чтобы избежать деления на ноль
        var r = Math.Max(0.0, Math.Min(0.999999, meanResultantLength));
        var d = (double)dimension;
            
        // Применяем аппроксимацию из статьи Banerjee et al.
        // κ ≈ r̄d - r̄³ / (1 - r̄²)
        var numerator = r * d - Math.Pow(r, 3);
        var denominator = 1.0 - Math.Pow(r, 2);
            
        var kappa = numerator / denominator;
            
        // Ограничиваем κ положительными разумными значениями
        return Math.Max(0.01, Math.Min(1000.0, kappa));
    }

    /// <summary>
    /// Вычисляет логарифм нормализующей константы c_d(κ) для vMF распределения
    /// Использует аппроксимацию для высоких размерностей чтобы избежать переполнения
    /// </summary>
    /// <param name="kappa">Параметр концентрации κ</param>
    /// <param name="dimension">Размерность d</param>
    /// <returns>Логарифм нормализующей константы</returns>
    private double ComputeLogNormalizingConstant(double kappa, long dimension)
    {
        var d = (double)dimension;
            
        // Для больших d используем аппроксимацию Стирлинга для избежания переполнения
        // log c_d(κ) ≈ (d/2 - 1) * log(κ) - (d/2) * log(2π) - log I_{d/2-1}(κ)
            
        if (dimension > 50 || kappa > 100)
        {
            // Используем асимптотическую аппроксимацию для больших κ или d
            // I_ν(x) ≈ exp(x) / sqrt(2πx) для больших x
            var nu = d / 2.0 - 1.0;
                
            var logBessel = kappa - 0.5 * Math.Log(2.0 * Math.PI * kappa);
            var logNormalizer = (d / 2.0 - 1.0) * Math.Log(kappa) - 
                                (d / 2.0) * Math.Log(2.0 * Math.PI) - logBessel;
                
            return logNormalizer;
        }
        else
        {
            // Для малых размерностей используем более точные вычисления
            // Это упрощённая версия, в production коде следует использовать
            // специализированные библиотеки для функций Бесселя
            var logGamma = LogGamma(d / 2.0);
            var logNormalizer = Math.Log(kappa / 2.0) * (d / 2.0 - 1.0) - 
                                logGamma - (d / 2.0) * Math.Log(Math.PI);
                
            return logNormalizer;
        }
    }

    /// <summary>
    /// Простая аппроксимация логарифма гамма-функции
    /// В production следует использовать более точную реализацию
    /// </summary>
    /// <param name="x">Аргумент</param>
    /// <returns>Приближённое значение log Γ(x)</returns>
    private double LogGamma(double x)
    {
        // Используем аппроксимацию Стирлинга: log Γ(x) ≈ (x-0.5)*log(x) - x + 0.5*log(2π)
        if (x < 1.0) return LogGamma(x + 1.0) - Math.Log(x);
            
        return (x - 0.5) * Math.Log(x) - x + 0.5 * Math.Log(2.0 * Math.PI);
    }

    /// <summary>
    /// Вычисляет полную логарифмическую вероятность данных для текущих параметров модели
    /// </summary>
    /// <param name="data">Входные данные</param>
    /// <returns>Логарифмическая вероятность</returns>
    private double ComputeLogLikelihood(Tensor data)
    {
        var numSamples = data.shape[0];
        var totalLogLikelihood = 0.0;
            
        for (long i = 0; i < numSamples; i++)
        {
            var sampleLogLikelihood = double.NegativeInfinity;
                
            for (int k = 0; k < _numClusters; k++)
            {
                // Вычисляем cosine similarity
                var cosineSimilarity = torch.dot(data[i], MeanDirections[k]).item<double>();
                    
                // Вычисляем логарифм компоненты смеси
                var logNormalizingConstant = ComputeLogNormalizingConstant(
                    Concentrations[k].item<double>(), 
                    data.shape[1]
                );
                    
                var logComponent = Math.Log(MixingCoefficients[k].item<double>()) + 
                                    logNormalizingConstant + 
                                    Concentrations[k].item<double>() * cosineSimilarity;
                    
                // Используем log-sum-exp trick для численной стабильности
                if (sampleLogLikelihood == double.NegativeInfinity)
                {
                    sampleLogLikelihood = logComponent;
                }
                else
                {
                    var maxLog = Math.Max(sampleLogLikelihood, logComponent);
                    sampleLogLikelihood = maxLog + Math.Log(
                        Math.Exp(sampleLogLikelihood - maxLog) + Math.Exp(logComponent - maxLog)
                    );
                }
            }
                
            totalLogLikelihood += sampleLogLikelihood;
        }
            
        return totalLogLikelihood;
    }

    /// <summary>
    /// Предсказывает кластерные назначения для новых данных
    /// </summary>
    /// <param name="data">Новые нормированные данные [N x D]</param>
    /// <returns>Назначения кластеров [N] (индексы от 0 до K-1)</returns>
    public Tensor Predict(Tensor data)
    {
        ValidateNormalizedData(data);
            
        var posteriors = ComputePosteriors(data);
        return torch.argmax(posteriors, dim: 1);
    }

    /// <summary>
    /// Возвращает мягкие назначения (вероятности принадлежности) для данных
    /// </summary>
    /// <param name="data">Нормированные данные [N x D]</param>
    /// <returns>Матрица вероятностей [N x K]</returns>
    public Tensor PredictProba(Tensor data)
    {
        ValidateNormalizedData(data);
        return ComputePosteriors(data);
    }

    /// <summary>
    /// Выводит информацию о изученной модели
    /// </summary>
    public void PrintModelSummary()
    {
        Console.WriteLine("=== Результаты vMF Кластеризации ===");
        Console.WriteLine($"Количество кластеров: {_numClusters}");
        Console.WriteLine($"Алгоритм назначения: {(_useHardAssignment ? "Hard (жёсткое)" : "Soft (мягкое)")}");
            
        Console.WriteLine("\nПараметры кластеров:");
        for (int k = 0; k < _numClusters; k++)
        {
            Console.WriteLine($"\nКластер {k}:");
            Console.WriteLine($"  Коэффициент смешивания α_{k}: {MixingCoefficients[k].item<double>():F4}");
            Console.WriteLine($"  Концентрация κ_{k}: {Concentrations[k].item<double>():F4}");
            Console.WriteLine($"  Направление μ_{k}: [{string.Join(", ", MeanDirections[k].data<float>().Take(5).Select(x => x.ToString("F3")))}...]");
        }
            
        if (LogLikelihoodHistory.Any())
        {
            Console.WriteLine($"\nФинальная log-likelihood: {LogLikelihoodHistory.Last():F6}");
            Console.WriteLine($"Количество итераций: {LogLikelihoodHistory.Count}");
        }
    }
}

/// <summary>
/// Демонстрационная программа использования vMF кластеризации
/// </summary>
public class Program
{
    public static void Main()
    {
        Console.WriteLine("=== Демонстрация von Mises-Fisher Кластеризации ===");
            
        // Устанавливаем случайное семя для воспроизводимости
        torch.manual_seed(42);
            
        // Генерируем тестовые данные (нормированные эмбеддинги слов)
        var testData = GenerateTestData();
            
        Console.WriteLine($"Сгенерированы тестовые данные: {testData.shape[0]} точек, размерность {testData.shape[1]}");
            
        // Создаём и обучаем кластеризатор
        var clusterer = new VonMisesFisherClusterer(
            numClusters: 3, 
            maxIterations: 50, 
            tolerance: 1e-6, 
            useHardAssignment: false
        );
            
        Console.WriteLine("\nНачинаем обучение...");
        clusterer.Fit(testData);
            
        // Выводим результаты
        clusterer.PrintModelSummary();
            
        // Демонстрируем предсказание
        Console.WriteLine("\n=== Тестирование предсказаний ===");
        var predictions = clusterer.Predict(testData.slice(0, 0, 10, 1)); // Первые 10 точек
        var probabilities = clusterer.PredictProba(testData.slice(0, 0, 10, 1));
            
        Console.WriteLine("Предсказания для первых 10 точек:");
        for (int i = 0; i < 10; i++)
        {
            var pred = predictions[i].item<long>();
            var prob = probabilities[i, pred].item<double>();
            Console.WriteLine($"Точка {i}: Кластер {pred} (вероятность: {prob:F3})");
        }
    }

    /// <summary>
    /// Генерирует тестовые данные для демонстрации
    /// Создаёт 3 кластера нормированных векторов на единичной сфере
    /// </summary>
    /// <returns>Тензор с тестовыми данными [N x D]</returns>
    private static Tensor GenerateTestData()
    {
        const int numSamplesPerCluster = 30;
        const int numClusters = 3;
        const int dimension = 10;
            
        var allData = new List<Tensor>();
            
        // Создаём истинные центры кластеров
        var trueCenters = new[]
        {
            torch.tensor(new float[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }), // Центр 1
            torch.tensor(new float[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 }), // Центр 2  
            torch.tensor(new float[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 })  // Центр 3
        };
            
        // Генерируем данные для каждого кластера
        for (int cluster = 0; cluster < numClusters; cluster++)
        {
            var center = trueCenters[cluster];
            var concentration = new[] { 10.0, 15.0, 5.0 }[cluster]; // Разные концентрации
                
            for (int sample = 0; sample < numSamplesPerCluster; sample++)
            {
                // Генерируем точку около центра с заданной концентрацией
                var noise = torch.randn(dimension) * (1.0 / Math.Sqrt(concentration));
                var point = center + noise;
                    
                // Нормализуем на единичную сферу
                point = point / torch.norm(point);
                allData.Add(point);
            }
        }
            
        // Объединяем все данные и перемешиваем
        var combinedData = torch.stack(allData.ToArray());
        var shuffleIndices = torch.randperm(combinedData.shape[0]);
            
        return combinedData[shuffleIndices];
    }
}