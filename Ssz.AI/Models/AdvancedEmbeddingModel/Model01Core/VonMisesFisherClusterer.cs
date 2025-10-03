using Microsoft.Extensions.Logging;
using Ssz.Utils.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model01Core;

/// <summary>
/// Реализация кластеризации von Mises-Fisher (vMF) на единичной сфере
/// Использует EM-алгоритм для оценки параметров смеси vMF распределений
/// </summary>
public class VonMisesFisherClusterer
{
    private readonly Device _device;
    private readonly IUserFriendlyLogger _userFriendlyLogger;

    // Поля класса для хранения параметров модели
    /// <summary>
    /// Количество кластеров K
    /// </summary>
    private readonly int _numClusters;

    /// <summary>
    /// Максимальное количество итераций EM
    /// </summary>
    private readonly int _maxIterations;

    /// <summary>
    /// Порог сходимости по логарифмической вероятности
    /// </summary>
    private readonly float _tolerance;

    /// <summary>
    /// Использовать жёсткое (hard) или мягкое (soft) назначение
    /// </summary>
    private readonly bool _useHardAssignment;

    // Параметры модели - изучаются в процессе обучения
    /// <summary>
    /// μ_k - направления средних для каждого кластера [K x D]
    /// </summary>
    /// <remarks>device: CPU</remarks>
    public Tensor MeanDirections { get; private set; } = null!;

    /// <summary>
    /// κ_k - параметры концентрации для каждого кластера [K]
    /// </summary>
    /// <remarks>device: CPU</remarks>
    public Tensor Concentrations { get; private set; } = null!;

    /// <summary>
    /// α_k - коэффициенты смешивания [K]
    /// </summary>
    /// <remarks>device: CPU</remarks>
    public Tensor MixingCoefficients { get; private set; } = null!;

    // Логи процесса обучения
    public List<float> LogLikelihoodHistory { get; private set; }

    /// <summary>
    /// Конструктор кластеризатора vMF
    /// </summary>
    /// <param name="device"></param>
    /// <param name="userFriendlyLogger"></param>
    /// <param name="numClusters">Количество кластеров K</param>
    /// <param name="maxIterations">Максимальное количество итераций EM алгоритма</param>
    /// <param name="tolerance">Порог сходимости для логарифмической вероятности</param>
    /// <param name="useHardAssignment">Использовать жёсткое назначение (true) или мягкое (false)</param>
    public VonMisesFisherClusterer(
        Device device,
        IUserFriendlyLogger userFriendlyLogger,
        int numClusters, 
        int maxIterations, 
        float tolerance, 
        bool useHardAssignment)
    {
        _device = device;
        _userFriendlyLogger = userFriendlyLogger;
        _numClusters = numClusters;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _useHardAssignment = useHardAssignment;
        LogLikelihoodHistory = new List<float>();
    }

    /// <summary>
    /// Основной метод обучения кластеризатора на нормированных данных
    /// </summary>
    /// <param name="oldVectorsTensor">Нормированные данные [N x D], где N - количество точек, D - размерность. Device: CPU</param>
    public void Fit(Tensor oldVectorsTensor)
    {
        // Проверяем, что данные корректно нормированы
        ValidateNormalizedData(oldVectorsTensor);
            
        var (numSamples, dimension) = (oldVectorsTensor.shape[0], oldVectorsTensor.shape[1]);
            
        // Инициализируем параметры модели
        InitializeParameters(oldVectorsTensor, Math.Min(10000, numSamples), dimension);
            
        float prevLogLikelihood = float.NegativeInfinity;
            
        // Основной цикл EM алгоритма
        for (int iteration = 0; iteration < _maxIterations; iteration += 1)
        {
            // E-шаг: вычисляем posterior probabilities p(k|x_i)
            var posteriors = ComputePosteriors(oldVectorsTensor);
                
            // M-шаг: обновляем параметры модели
            UpdateParameters(oldVectorsTensor, posteriors);
                
            // Вычисляем логарифмическую вероятность для проверки сходимости
            var logLikelihood = ComputeLogLikelihood(oldVectorsTensor[..1000, ..]);
            LogLikelihoodHistory.Add(logLikelihood);
                
            // Проверяем сходимость
            if (Math.Abs(logLikelihood - prevLogLikelihood) < _tolerance)
            {
                _userFriendlyLogger.LogInformation($"Сходимость достигнута на итерации {iteration + 1}");
                break;
            }
                
            prevLogLikelihood = logLikelihood;
                
            // Выводим прогресс каждые 10 итераций
            //if ((iteration + 1) % 10 == 0)
            {
                _userFriendlyLogger.LogInformation($"Итерация {iteration + 1}: Log-Likelihood = {logLikelihood:F6}");
            }
        }
            
        _userFriendlyLogger.LogInformation($"Обучение завершено. Финальная Log-Likelihood: {prevLogLikelihood:F6}");
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="oldVectorsTensor">Device: CPU</param>
    /// <param name="clusterization_AlgorithmData"></param>
    public void GetResult(Tensor oldVectorsTensor, Clusterization_AlgorithmData clusterization_AlgorithmData)
    {
        var numSamples = oldVectorsTensor.shape[0];        

        // Вычисляем логарифмические вероятности для численной стабильности
        using var logProbabilities = torch.zeros(size: new long[] { numSamples, _numClusters });

        for (int k = 0; k < _numClusters; k++)
        {
            // Вычисляем cosine similarity между данными и k-м центром
            var cosineSimilarities = torch.matmul(oldVectorsTensor, MeanDirections[k]);

            // Вычисляем логарифм нормализующей константы c_d(κ)
            var logNormalizingConstant = ComputeLogNormalizingConstant(
                Concentrations[k].item<float>(),
                oldVectorsTensor.shape[1]
            );

            // Логарифм vMF плотности: log c_d(κ) + κ * μ^T * x
            logProbabilities[.., k] = logNormalizingConstant +
                Concentrations[k] * cosineSimilarities +
                torch.log(MixingCoefficients[k]);

            MeanDirections.select(dim: 0, index: k).data<float>().CopyTo(clusterization_AlgorithmData.ClusterInfos[k].CentroidOldVectorNormalized);
        }

        // Жёсткое назначение: назначаем каждую точку кластеру с максимальной вероятностью
        var assignments = torch.argmax(logProbabilities, dim: 1);
        for (long i = 0; i < numSamples; i++)
        {
            var assignedCluster = assignments[i].item<long>();
            clusterization_AlgorithmData.ClusterIndices[i] = (int)assignedCluster;
        }

        var clusterInfos = clusterization_AlgorithmData.ClusterInfos;
        Word[] primaryWords = clusterization_AlgorithmData.PrimaryWords;
        var words = clusterization_AlgorithmData.LanguageInfo.Words;

        Parallel.For(0, clusterInfos.Length, clusterIndex =>
        {
            var centroidOldVectorNormalized = clusterInfos[clusterIndex].CentroidOldVectorNormalized;

            int nearestWordIndex = -1;
            float nearestDotProduct = 0.0f;
            for (int wordIndex = 0; wordIndex < words.Count; wordIndex += 1)
            {
                Word word = words[wordIndex];
                var oldVectror = word.OldVectorNormalized;

                float dotProduct = System.Numerics.Tensors.TensorPrimitives.Dot(oldVectror, centroidOldVectorNormalized);
                if (dotProduct > nearestDotProduct)
                {
                    nearestDotProduct = dotProduct;
                    nearestWordIndex = wordIndex;
                }
            }

            primaryWords[clusterIndex] = words[nearestWordIndex];
        });

        Array.Clear(clusterization_AlgorithmData.IsPrimaryWord);        
        foreach (var primaryWord in primaryWords)
        {
            clusterization_AlgorithmData.IsPrimaryWord[primaryWord.Index] = true;
        }
    }

    /// <summary>
    /// Проверяет, что входные данные правильно нормированы (лежат на единичной сфере)
    /// </summary>
    /// <param name="data">Данные для проверки</param>
    private void ValidateNormalizedData(Tensor data)
    {
        // Вычисляем нормы всех векторов
        var norms = torch.norm(data, dimension: 1);
        var minNorm = torch.min(norms).item<float>();
        var maxNorm = torch.max(norms).item<float>();
            
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
    /// <param name="oldVectorsTensor">Входные данные</param>
    /// <param name="numSamples">Количество образцов</param>
    /// <param name="dimension">Размерность данных</param>
    private void InitializeParameters(Tensor oldVectorsTensor, long numSamples, long dimension)
    {
        using (var disposeScope = torch.NewDisposeScope())
        {
            var oldVectorsTensor_device = oldVectorsTensor.to(_device);

            // Инициализация направлений средних с помощью kmeans++ подобного метода
            var meanDirections_device = torch.zeros(new long[] { _numClusters, dimension }, dtype: torch.float32, device: _device);

            // Выбираем первый центр случайно
            var randomIndex = torch.randint(low: 0, high: numSamples, size: new long[] { 1 }).item<long>();
            meanDirections_device[0] = oldVectorsTensor_device[randomIndex];

            // Выбираем остальные центры с вероятностью пропорциональной расстоянию до ближайшего центра
            for (int k = 1; k < _numClusters; k += 1)
            {
                // Для каждой точки данных находим расстояние до ближайшего уже выбранного центра

                var similarity = torch.mm(oldVectorsTensor_device, meanDirections_device[..k, ..].t());
                var (probabilities, indices) = torch.topk(similarity, k: 1, dim: 1, largest: true);
                probabilities.neg_().add_(1.0f).pow_(2);

                // UNOPTIMIZED
                //var distances = torch.zeros(numSamples);
                //for (long i = 0; i < numSamples; i += 1)
                //{
                //    var minDistance = float.MaxValue;

                //    // Вычисляем минимальное расстояние до уже выбранных центров
                //    for (int j = 0; j < k; j += 1)
                //    {
                //        // Используем угловое расстояние: 1 - cosine_similarity
                //        var cosineSimilarity = torch.dot(oldVectorsTensor[i], MeanDirections[j]).item<float>();
                //        var angularDistance = 1.0 - cosineSimilarity;
                //        minDistance = Math.Min(minDistance, angularDistance);
                //    }

                //    distances[i] = minDistance;
                //}

                //// Выбираем следующий центр с вероятностью пропорциональной квадрату расстояния
                //var probabilities = torch.pow(distances, 2);
                //var s = torch.sum(probabilities);
                //probabilities = probabilities / s;        

                // Используем multinomial sampling для выбора индекса
                var selectedIndex = torch.multinomial(input: probabilities.t(), num_samples: 1).item<long>();
                meanDirections_device[k] = oldVectorsTensor_device[selectedIndex];

                if ((k + 1) % 10 == 0)
                {
                    _userFriendlyLogger.LogInformation($"Инициализирован кластер {k}/{_numClusters}.");
                }
            }

            MeanDirections = meanDirections_device.to(CPU).DetachFromDisposeScope();
        }            

        // Инициализация параметров концентрации
        // Начинаем с умеренных значений концентрации
        Concentrations = torch.ones(size: _numClusters, dtype: torch.float32) * 1.0f;

        // Инициализация коэффициентов смешивания (равномерное распределение)
        MixingCoefficients = torch.ones(_numClusters, dtype: torch.float32) / _numClusters;

        _userFriendlyLogger.LogInformation("Параметры инициализированы:");
        _userFriendlyLogger.LogInformation($"Количество кластеров: {_numClusters}");
        _userFriendlyLogger.LogInformation($"Размерность данных: {dimension}");
        _userFriendlyLogger.LogInformation($"Количество образцов: {numSamples}");
    }

    /// <summary>
    /// E-шаг EM алгоритма: вычисляет posterior probabilities p(k|x_i)
    /// </summary>
    /// <param name="oldVectorsTensor">Входные данные [N x D]</param>
    /// <returns>Матрица posterior probabilities [N x K]</returns>
    private Tensor ComputePosteriors(Tensor oldVectorsTensor)
    {
        var numSamples = oldVectorsTensor.shape[0];
        var posteriors = torch.zeros(new long[] { numSamples, _numClusters });
            
        // Вычисляем логарифмические вероятности для численной стабильности
        using var logProbabilities = torch.zeros(size: new long[] { numSamples, _numClusters });
            
        for (int k = 0; k < _numClusters; k++)
        {
            // Вычисляем cosine similarity между данными и k-м центром
            var cosineSimilarities = torch.matmul(oldVectorsTensor, MeanDirections[k, ..].t());
                
            // Вычисляем логарифм нормализующей константы c_d(κ)
            var logNormalizingConstant = ComputeLogNormalizingConstant(
                Concentrations[k].item<float>(), 
                oldVectorsTensor.shape[1]
            );
                
            // Логарифм vMF плотности: log c_d(κ) + κ * μ^T * x
            logProbabilities[.., k] = logNormalizingConstant + 
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
                posteriors[i, assignedCluster] = 1.0f;
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
    /// <param name="oldVectorsTensor">Входные данные [N x D]. Device: CPU</param>
    /// <param name="posteriors">Posterior probabilities [N x K]. Device: CPU</param>
    private void UpdateParameters(Tensor oldVectorsTensor, Tensor posteriors)
    {
        var numSamples = oldVectorsTensor.shape[0];
            
        for (int k = 0; k < _numClusters; k += 1)
        {
            // Обновляем коэффициент смешивания α_k
            var effectiveSampleSize = torch.sum(posteriors[.., k]);
            MixingCoefficients[k] = effectiveSampleSize / numSamples;
                
            // Вычисляем взвешенную сумму точек для кластера k
            var weightedSum = torch.zeros_like(MeanDirections[k]);
            for (long i = 0; i < numSamples; i++)
            {
                weightedSum += posteriors[i, k] * oldVectorsTensor[i];
            }
                
            // Обновляем направление среднего μ_k
            var resultantLength = torch.norm(weightedSum);
            MeanDirections[k] = weightedSum / resultantLength;
                
            // Вычисляем средний resultant length для оценки κ_k
            var meanResultantLength = (resultantLength / effectiveSampleSize).item<float>();
                
            // Обновляем параметр концентрации κ_k используя аппроксимацию из статьи
            var newConcentration = ApproximateConcentration(meanResultantLength, oldVectorsTensor.shape[1]);
            Concentrations[k] = newConcentration;
        }
            
        // Нормализуем коэффициенты смешивания для обеспечения того, что их сумма равна 1
        MixingCoefficients = MixingCoefficients / torch.sum(MixingCoefficients);
        float threshold = 1e-10f;
        // Создаём булеву маску: где tensor < порог
        var mask = MixingCoefficients.lt(threshold); // mask — логический тензор Nx1
        // Применяем маску и присваиваем новое значение тем элементам, где mask == true
        MixingCoefficients[mask] = threshold;
    }

    /// <summary>
    /// Вычисляет приближённое значение параметра концентрации κ
    /// Использует аппроксимацию из статьи: κ ≈ r̄d - r̄³ / (1 - r̄²)
    /// где r̄ - средний resultant length
    /// </summary>
    /// <param name="meanResultantLength">Средний resultant length r̄</param>
    /// <param name="dimension">Размерность данных d</param>
    /// <returns>Приближённое значение κ</returns>
    private float ApproximateConcentration(float meanResultantLength, long dimension)
    {
        // Ограничиваем r̄ чтобы избежать деления на ноль
        var r = MathF.Max(0.0f, MathF.Min(0.999999f, meanResultantLength));
        var d = (float)dimension;
            
        // Применяем аппроксимацию из статьи Banerjee et al.
        // κ ≈ r̄d - r̄³ / (1 - r̄²)
        var numerator = r * d - MathF.Pow(r, 3);
        var denominator = 1.0f - MathF.Pow(r, 2);
            
        var kappa = numerator / denominator;
            
        // Ограничиваем κ положительными разумными значениями
        return Math.Max(0.01f, MathF.Min(1000.0f, kappa));
    }

    /// <summary>
    /// Вычисляет логарифм нормализующей константы c_d(κ) для vMF распределения
    /// Использует аппроксимацию для высоких размерностей чтобы избежать переполнения
    /// </summary>
    /// <param name="kappa">Параметр концентрации κ</param>
    /// <param name="dimension">Размерность d</param>
    /// <returns>Логарифм нормализующей константы</returns>
    private float ComputeLogNormalizingConstant(float kappa, long dimension)
    {
        var d = (float)dimension;
            
        // Для больших d используем аппроксимацию Стирлинга для избежания переполнения
        // log c_d(κ) ≈ (d/2 - 1) * log(κ) - (d/2) * log(2π) - log I_{d/2-1}(κ)
            
        if (dimension > 50 || kappa > 100)
        {
            // Используем асимптотическую аппроксимацию для больших κ или d
            // I_ν(x) ≈ exp(x) / sqrt(2πx) для больших x
            //var nu = d / 2.0f - 1.0f;
                
            var logBessel = kappa - 0.5f * MathF.Log(2.0f * MathF.PI * kappa);
            var logNormalizer = (d / 2.0f - 1.0f) * MathF.Log(kappa) - 
                                (d / 2.0f) * MathF.Log(2.0f * MathF.PI) - logBessel;
                
            return logNormalizer;
        }
        else
        {
            // Для малых размерностей используем более точные вычисления
            // Это упрощённая версия, в production коде следует использовать
            // специализированные библиотеки для функций Бесселя
            var logGamma = LogGamma(d / 2.0f);
            var logNormalizer = MathF.Log(kappa / 2.0f) * (d / 2.0f - 1.0f) - 
                                logGamma - (d / 2.0f) * MathF.Log(MathF.PI);
                
            return logNormalizer;
        }
    }

    /// <summary>
    /// Простая аппроксимация логарифма гамма-функции
    /// В production следует использовать более точную реализацию
    /// </summary>
    /// <param name="x">Аргумент</param>
    /// <returns>Приближённое значение log Γ(x)</returns>
    private float LogGamma(float x)
    {
        // Используем аппроксимацию Стирлинга: log Γ(x) ≈ (x-0.5)*log(x) - x + 0.5*log(2π)
        if (x < 1.0f) return LogGamma(x + 1.0f) - MathF.Log(x);
            
        return (x - 0.5f) * MathF.Log(x) - x + 0.5f * MathF.Log(2.0f * MathF.PI);
    }

    /// <summary>
    /// Вычисляет полную логарифмическую вероятность данных для текущих параметров модели
    /// </summary>
    /// <param name="oldVectorsTensor">Входные данные</param>
    /// <returns>Логарифмическая вероятность</returns>
    private float ComputeLogLikelihood(Tensor oldVectorsTensor)
    {
        var numSamples = oldVectorsTensor.shape[0];
        var totalLogLikelihood = 0.0f;

        //using (var disposeScope = torch.NewDisposeScope())
        {
            var oldVectorsTensor_device = oldVectorsTensor; //.to(_device);
            // [K x D]
            var meanDirectionsr_device = MeanDirections; //.to(_device);

            // Вычисляем cosine similarity
            var cosineSimilarity = torch.mm(oldVectorsTensor_device, meanDirectionsr_device.t());

            for (long i = 0; i < numSamples; i++)
            {
                var sampleLogLikelihood = float.NegativeInfinity;

                for (int k = 0; k < _numClusters; k += 1)
                {
                    // Вычисляем логарифм компоненты смеси
                    var logNormalizingConstant = ComputeLogNormalizingConstant(
                        Concentrations[k].item<float>(),
                        oldVectorsTensor.shape[1]
                    );

                    var logComponent = MathF.Log(MixingCoefficients[k].item<float>()) +
                                        logNormalizingConstant +
                                        Concentrations[k].item<float>() * cosineSimilarity[i, k].item<float>();

                    // Используем log-sum-exp trick для численной стабильности
                    if (sampleLogLikelihood == float.NegativeInfinity)
                    {
                        sampleLogLikelihood = logComponent;
                    }
                    else
                    {
                        var maxLog = Math.Max(sampleLogLikelihood, logComponent);
                        sampleLogLikelihood = maxLog + MathF.Log(
                            MathF.Exp(sampleLogLikelihood - maxLog) + MathF.Exp(logComponent - maxLog)
                        );
                    }
                }

                totalLogLikelihood += sampleLogLikelihood;
            }
        }            
            
        return totalLogLikelihood;
    }

    /// <summary>
    /// Предсказывает кластерные назначения для новых данных
    /// </summary>
    /// <param name="oldVectorsTensor">Новые нормированные данные [N x D]. Device: CPU</param>
    /// <returns>Назначения кластеров [N] (индексы от 0 до K-1)</returns>
    public Tensor Predict(Tensor oldVectorsTensor)
    {
        ValidateNormalizedData(oldVectorsTensor);
            
        var posteriors = ComputePosteriors(oldVectorsTensor);
        return torch.argmax(posteriors, dim: 1);
    }

    /// <summary>
    /// Возвращает мягкие назначения (вероятности принадлежности) для данных
    /// </summary>
    /// <param name="oldVectorsTensor">Нормированные данные [N x D]. Device: CPU</param>
    /// <returns>Матрица вероятностей [N x K]</returns>
    public Tensor PredictProba(Tensor oldVectorsTensor)
    {
        ValidateNormalizedData(oldVectorsTensor);
        return ComputePosteriors(oldVectorsTensor);
    }

    /// <summary>
    /// Выводит информацию о изученной модели
    /// </summary>
    public void PrintModelSummary()
    {
        _userFriendlyLogger.LogInformation("=== Результаты vMF Кластеризации ===");
        _userFriendlyLogger.LogInformation($"Количество кластеров: {_numClusters}");
        _userFriendlyLogger.LogInformation($"Алгоритм назначения: {(_useHardAssignment ? "Hard (жёсткое)" : "Soft (мягкое)")}");
            
        _userFriendlyLogger.LogInformation("\nПараметры кластеров:");
        for (int k = 0; k < _numClusters; k++)
        {
            _userFriendlyLogger.LogInformation($"\nКластер {k}:");
            _userFriendlyLogger.LogInformation($"  Коэффициент смешивания α_{k}: {MixingCoefficients[k].item<float>():F4}");
            _userFriendlyLogger.LogInformation($"  Концентрация κ_{k}: {Concentrations[k].item<float>():F4}");
            _userFriendlyLogger.LogInformation($"  Направление μ_{k}: [{string.Join(", ", MeanDirections[k].data<float>().Take(5).Select(x => x.ToString("F3")))}...]");
        }
            
        if (LogLikelihoodHistory.Any())
        {
            _userFriendlyLogger.LogInformation($"\nФинальная log-likelihood: {LogLikelihoodHistory.Last():F6}");
            _userFriendlyLogger.LogInformation($"Количество итераций: {LogLikelihoodHistory.Count}");
        }
    }
}