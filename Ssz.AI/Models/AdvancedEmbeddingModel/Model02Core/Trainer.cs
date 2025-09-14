using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core;
using CrossLingualEmbeddings.Models;
using CrossLingualEmbeddings.Utils;
using Microsoft.Extensions.Logging;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using TorchSharp.Modules;

namespace CrossLingualEmbeddings.Training
{
    /// <summary>
    /// Параметры для тренера
    /// </summary>
    public sealed record TrainerParameters
    {
        /// <summary>
        /// Размер батча для обучения
        /// </summary>
        public int BatchSize { get; init; } = 32;
        
        /// <summary>
        /// Количество шагов дискриминатора за одну итерацию
        /// </summary>
        public int DiscriminatorSteps { get; init; } = 5;
        
        /// <summary>
        /// Коэффициент потерь дискриминатора
        /// </summary>
        public double DiscriminatorLambda { get; init; } = 1.0;
        
        /// <summary>
        /// Сглаживание для дискриминатора
        /// </summary>
        public double DiscriminatorSmoothing { get; init; } = 0.1;
        
        /// <summary>
        /// Количество наиболее частых слов для дискриминации
        /// </summary>
        public int MostFrequentWords { get; init; } = 75000;
        
        /// <summary>
        /// Обрезание градиентов дискриминатора
        /// </summary>
        public double DiscriminatorClipWeights { get; init; } = 0.0;
        
        /// <summary>
        /// Устройство для обучения
        /// </summary>
        public Device Device { get; init; } = CPU;
        
        /// <summary>
        /// Путь для сохранения экспериментов
        /// </summary>
        public string ExperimentPath { get; init; } = "./experiments";
    }

    /// <summary>
    /// Статистики обучения
    /// </summary>
    public sealed record TrainingStats
    {
        /// <summary>
        /// Потери дискриминатора
        /// </summary>
        public List<double> DiscriminatorLosses { get; init; } = new();
        
        /// <summary>
        /// Потери маппинга
        /// </summary>
        public List<double> MappingLosses { get; init; } = new();
        
        /// <summary>
        /// Точность дискриминатора
        /// </summary>
        public double DiscriminatorAccuracy { get; set; }
        
        /// <summary>
        /// Количество обработанных слов
        /// </summary>
        public long ProcessedWords { get; set; }
        
        /// <summary>
        /// Время обучения
        /// </summary>
        public TimeSpan ElapsedTime { get; set; }
    }

    /// <summary>
    /// Основной класс для обучения кросс-лингвальных эмбеддингов
    /// Аналог Trainer класса из Python проекта с оптимизациями для .NET 9
    /// </summary>
    public sealed class CrossLingualTrainer : IDisposable
    {
        #region Private Fields

        /// <summary>
        /// Эмбеддинги исходного языка
        /// </summary>
        private readonly Embedding _sourceEmbeddings;
        
        /// <summary>
        /// Эмбеддинги целевого языка
        /// </summary>
        private readonly Embedding _targetEmbeddings;
        
        /// <summary>
        /// Модель выравнивания
        /// </summary>
        private readonly EmbeddingMapping _mapping;
        
        /// <summary>
        /// Дискриминатор (может быть null для supervised обучения)
        /// </summary>
        private readonly Discriminator? _discriminator;
        
        /// <summary>
        /// Словарь исходного языка
        /// </summary>
        private readonly Dictionary _sourceDictionary;
        
        /// <summary>
        /// Словарь целевого языка
        /// </summary>
        private readonly Dictionary _targetDictionary;
        
        /// <summary>
        /// Параметры тренера
        /// </summary>
        private readonly TrainerParameters _parameters;
        
        /// <summary>
        /// Логгер
        /// </summary>
        private readonly ILogger? _logger;
        
        /// <summary>
        /// Оптимизатор для модели выравнивания
        /// </summary>
        private Optimizer? _mappingOptimizer;
        
        /// <summary>
        /// Оптимизатор для дискриминатора
        /// </summary>
        private Optimizer? _discriminatorOptimizer;
        
        /// <summary>
        /// Обучающий словарь (пары индексов)
        /// </summary>
        private Tensor? _trainingDictionary;
        
        /// <summary>
        /// Лучшая метрика валидации
        /// </summary>
        private double _bestValidationMetric = double.NegativeInfinity;
        
        /// <summary>
        /// Флаг уменьшения learning rate
        /// </summary>
        private bool _decreaseLearningRate = false;
        
        /// <summary>
        /// Генератор случайных чисел
        /// </summary>
        private readonly Random _random = new();
        
        /// <summary>
        /// Флаг освобождения ресурсов
        /// </summary>
        private bool _disposed = false;

        #endregion

        #region Constructor

        /// <summary>
        /// Инициализирует новый экземпляр тренера
        /// </summary>
        /// <param name="sourceEmbeddings">Эмбеддинги исходного языка</param>
        /// <param name="targetEmbeddings">Эмбеддинги целевого языка</param>
        /// <param name="mapping">Модель выравнивания</param>
        /// <param name="discriminator">Дискриминатор (может быть null)</param>
        /// <param name="sourceDictionary">Словарь исходного языка</param>
        /// <param name="targetDictionary">Словарь целевого языка</param>
        /// <param name="parameters">Параметры тренера</param>
        /// <param name="logger">Логгер</param>
        public CrossLingualTrainer(
            Embedding sourceEmbeddings,
            Embedding targetEmbeddings,
            EmbeddingMapping mapping,
            Discriminator? discriminator,
            Dictionary sourceDictionary,
            Dictionary targetDictionary,
            TrainerParameters? parameters = null,
            ILogger? logger = null)
        {
            _sourceEmbeddings = sourceEmbeddings ?? throw new ArgumentNullException(nameof(sourceEmbeddings));
            _targetEmbeddings = targetEmbeddings ?? throw new ArgumentNullException(nameof(targetEmbeddings));
            _mapping = mapping ?? throw new ArgumentNullException(nameof(mapping));
            _discriminator = discriminator;
            _sourceDictionary = sourceDictionary ?? throw new ArgumentNullException(nameof(sourceDictionary));
            _targetDictionary = targetDictionary ?? throw new ArgumentNullException(nameof(targetDictionary));
            _parameters = parameters ?? new TrainerParameters();
            _logger = logger;
            
            // Перемещаем модели на указанное устройство
            _sourceEmbeddings.to(_parameters.Device);
            _targetEmbeddings.to(_parameters.Device);
            _mapping.to(_parameters.Device);
            _discriminator?.to(_parameters.Device);
        }

        #endregion

        #region Public Properties

        /// <summary>
        /// Параметры тренера
        /// </summary>
        public TrainerParameters Parameters => _parameters;
        
        /// <summary>
        /// Лучшая метрика валидации
        /// </summary>
        public double BestValidationMetric => _bestValidationMetric;

        #endregion

        #region Public Methods - Optimizer Setup

        /// <summary>
        /// Настраивает оптимизаторы для обучения
        /// </summary>
        /// <param name="mappingOptimizerConfig">Конфигурация оптимизатора для маппинга (например, "sgd,lr=0.1")</param>
        /// <param name="discriminatorOptimizerConfig">Конфигурация оптимизатора для дискриминатора</param>
        public void SetupOptimizers(string mappingOptimizerConfig, string? discriminatorOptimizerConfig = null)
        {
            _logger?.LogInformation("Настройка оптимизаторов...");
            
            // Настраиваем оптимизатор для маппинга
            _mappingOptimizer?.Dispose();
            _mappingOptimizer = CreateOptimizerFromConfig(mappingOptimizerConfig, _mapping.parameters());
            
            // Настраиваем оптимизатор для дискриминатора (если есть)
            if (_discriminator != null && !string.IsNullOrEmpty(discriminatorOptimizerConfig))
            {
                _discriminatorOptimizer?.Dispose();
                _discriminatorOptimizer = CreateOptimizerFromConfig(discriminatorOptimizerConfig, _discriminator.parameters());
            }
            
            _logger?.LogInformation($"Оптимизатор маппинга: {mappingOptimizerConfig}");
            if (discriminatorOptimizerConfig != null)
                _logger?.LogInformation($"Оптимизатор дискриминатора: {discriminatorOptimizerConfig}");
        }

        /// <summary>
        /// Создает оптимизатор из строковой конфигурации
        /// </summary>
        /// <param name="config">Конфигурация в формате "sgd,lr=0.1,momentum=0.9"</param>
        /// <param name="parameters">Параметры для оптимизации</param>
        /// <returns>Настроенный оптимизатор</returns>
        private Optimizer CreateOptimizerFromConfig(string config, IEnumerable<Parameter> parameters)
        {
            var parts = config.Split(',');
            var optimizerType = parts[0].ToLowerInvariant();
            
            // Парсим дополнительные параметры
            var options = new Dictionary<string, double>();
            for (int i = 1; i < parts.Length; i++)
            {
                var keyValue = parts[i].Split('=');
                if (keyValue.Length == 2 && double.TryParse(keyValue[1], out var value))
                {
                    options[keyValue[0]] = value;
                }
            }
            
            // Создаем оптимизатор в зависимости от типа
            return optimizerType switch
            {
                "sgd" => SGD(parameters, 
                    learningRate: options.GetValueOrDefault("lr", 0.01),
                    momentum: options.GetValueOrDefault("momentum", 0.0),
                    weight_decay: options.GetValueOrDefault("weight_decay", 0.0)),
                    
                "adam" => Adam(parameters,
                    lr: options.GetValueOrDefault("lr", 0.001),
                    weight_decay: options.GetValueOrDefault("weight_decay", 0.0)),
                    
                "adagrad" => Adagrad(parameters,
                    lr: options.GetValueOrDefault("lr", 0.01),
                    weight_decay: options.GetValueOrDefault("weight_decay", 0.0)),
                    
                _ => throw new ArgumentException($"Неизвестный тип оптимизатора: {optimizerType}")
            };
        }

        #endregion

        #region Public Methods - Dictionary Management

        /// <summary>
        /// Загружает обучающий словарь
        /// </summary>
        /// <param name="dictionaryPath">Путь к словарю или специальные значения ("identical_char", "default")</param>
        public async Task LoadTrainingDictionaryAsync(string dictionaryPath)
        {
            _logger?.LogInformation($"Загрузка обучающего словаря: {dictionaryPath}");
            
            if (dictionaryPath == "identical_char")
            {
                _trainingDictionary = await LoadIdenticalCharacterDictionaryAsync();
            }
            else if (dictionaryPath == "default")
            {
                _trainingDictionary = await LoadDefaultDictionaryAsync();
            }
            else
            {
                _trainingDictionary = await LoadDictionaryFromFileAsync(dictionaryPath);
            }
            
            _logger?.LogInformation($"Загружен словарь с {_trainingDictionary!.size(0)} парами");
        }

        /// <summary>
        /// Загружает словарь идентичных символьных строк
        /// </summary>
        private async Task<Tensor> LoadIdenticalCharacterDictionaryAsync()
        {
            await Task.Run(() => { }); // Placeholder для async
            
            var pairs = new List<(int, int)>();
            
            foreach (var sourceWord in _sourceDictionary.GetWords())
            {
                if (_targetDictionary.TryGetIndex(sourceWord, out int targetIndex))
                {
                    var sourceIndex = _sourceDictionary.GetIndex(sourceWord);
                    pairs.Add((sourceIndex, targetIndex));
                }
            }
            
            if (pairs.Count == 0)
                throw new InvalidOperationException("Не найдено ни одной идентичной символьной строки");
            
            _logger?.LogInformation($"Найдено {pairs.Count} пар идентичных символьных строк");
            
            // Сортируем по частоте исходного слова (по индексу)
            pairs.Sort((a, b) => a.Item1.CompareTo(b.Item1));
            
            // Создаем тензор словаря
            var dictionaryData = new long[pairs.Count * 2];
            for (int i = 0; i < pairs.Count; i++)
            {
                dictionaryData[i * 2] = pairs[i].Item1;
                dictionaryData[i * 2 + 1] = pairs[i].Item2;
            }
            
            return tensor(dictionaryData, dtype: ScalarType.Int64, device: _parameters.Device)
                .reshape(pairs.Count, 2);
        }

        /// <summary>
        /// Загружает словарь по умолчанию
        /// </summary>
        private async Task<Tensor> LoadDefaultDictionaryAsync()
        {
            var defaultPath = Path.Combine("data", "crosslingual", "dictionaries",
                $"{_sourceDictionary.Language}-{_targetDictionary.Language}.0-5000.txt");
                
            if (File.Exists(defaultPath))
            {
                return await LoadDictionaryFromFileAsync(defaultPath);
            }
            
            // Если файл не найден, используем identical_char как fallback
            _logger?.LogWarning($"Файл словаря по умолчанию не найден: {defaultPath}. Используем identical_char");
            return await LoadIdenticalCharacterDictionaryAsync();
        }

        /// <summary>
        /// Загружает словарь из файла
        /// </summary>
        /// <param name="filePath">Путь к файлу словаря</param>
        private async Task<Tensor> LoadDictionaryFromFileAsync(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"Файл словаря не найден: {filePath}");
            
            var pairs = new List<(int, int)>();
            int notFound = 0;
            
            await foreach (var line in File.ReadLinesAsync(filePath))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                
                var parts = line.Trim().Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 2) continue;
                
                var sourceWord = parts[0].ToLowerInvariant();
                var targetWord = parts[1].ToLowerInvariant();
                
                if (_sourceDictionary.TryGetIndex(sourceWord, out int sourceIndex) &&
                    _targetDictionary.TryGetIndex(targetWord, out int targetIndex))
                {
                    pairs.Add((sourceIndex, targetIndex));
                }
                else
                {
                    notFound++;
                }
            }
            
            _logger?.LogInformation($"Загружено {pairs.Count} пар слов из словаря. " +
                                  $"{notFound} пар содержали неизвестные слова");
            
            // Сортируем по частоте исходного слова
            pairs.Sort((a, b) => a.Item1.CompareTo(b.Item1));
            
            // Создаем тензор
            var dictionaryData = new long[pairs.Count * 2];
            for (int i = 0; i < pairs.Count; i++)
            {
                dictionaryData[i * 2] = pairs[i].Item1;
                dictionaryData[i * 2 + 1] = pairs[i].Item2;
            }
            
            return tensor(dictionaryData, dtype: ScalarType.Int64, device: _parameters.Device)
                .reshape(pairs.Count, 2);
        }

        #endregion

        #region Public Methods - Training Steps

        /// <summary>
        /// Выполняет шаг обучения дискриминатора
        /// </summary>
        /// <param name="stats">Статистики для записи потерь</param>
        /// <returns>Потери дискриминатора</returns>
        public double DiscriminatorStep(TrainingStats stats)
        {
            if (_discriminator == null || _discriminatorOptimizer == null)
                throw new InvalidOperationException("Дискриминатор или его оптимизатор не настроены");
            
            _discriminator.train();
            
            // Получаем батч данных для дискриминатора
            var (x, y) = GetDiscriminatorBatch();
            
            // Прямой проход
            var predictions = _discriminator.forward(x);
            var loss = nn.functional.binary_cross_entropy(predictions, y);
            
            // Проверяем на NaN
            if (loss.isnan().item<bool>())
            {
                _logger?.LogError("Обнаружен NaN в потерях дискриминатора");
                throw new InvalidOperationException("NaN обнаружен в потерях дискриминатора");
            }
            
            // Обратное распространение
            _discriminatorOptimizer.zero_grad();
            loss.backward();
            _discriminatorOptimizer.step();
            
            // Обрезаем градиенты если необходимо
            if (_parameters.DiscriminatorClipWeights > 0)
            {
                _discriminator.ClipGradients(_parameters.DiscriminatorClipWeights);
            }
            
            var lossValue = loss.item<float>(); // VALFIX
            stats.DiscriminatorLosses.Add(lossValue);
            
            return lossValue;
        }

        /// <summary>
        /// Выполняет шаг обучения модели выравнивания (обман дискриминатора)
        /// </summary>
        /// <param name="stats">Статистики для записи</param>
        /// <returns>Количество обработанных слов</returns>
        public long MappingStep(TrainingStats stats)
        {
            if (_parameters.DiscriminatorLambda == 0.0)
                return 0;
            
            if (_discriminator == null || _mappingOptimizer == null)
                throw new InvalidOperationException("Дискриминатор или оптимизатор маппинга не настроены");
            
            _discriminator.eval();
            _mapping.train();
            
            // Получаем батч данных
            var (x, y) = GetDiscriminatorBatch(volatile_: false);
            
            // Инвертируем метки для обмана дискриминатора
            var invertedY = ones_like(y) - y;
            
            // Прямой проход
            var predictions = _discriminator.forward(x);
            var loss = nn.functional.binary_cross_entropy(predictions, invertedY);
            loss = loss * _parameters.DiscriminatorLambda;
            
            // Проверяем на NaN
            if (loss.isnan().item<bool>())
            {
                _logger?.LogError("Обнаружен NaN в потерях маппинга (обман дискриминатора)");
                throw new InvalidOperationException("NaN в потерях маппинга");
            }
            
            // Обратное распространение
            _mappingOptimizer.zero_grad();
            loss.backward();
            _mappingOptimizer.step();
            
            // Ортогонализуем матрицу маппинга
            _mapping.OrthogonalizeWeights();
            
            stats.MappingLosses.Add(loss.item<float>()); // VALFIX
            return 2 * _parameters.BatchSize;
        }

        /// <summary>
        /// Применяет решение Прокруста для выравнивания эмбеддингов
        /// </summary>
        public void ApplyProcrustesAlignment()
        {
            if (_trainingDictionary is null)
                throw new InvalidOperationException("Обучающий словарь не загружен");
            
            _logger?.LogInformation("Применение решения ортогонального Прокруста...");
            
            using var _ = no_grad();
            
            // Получаем эмбеддинги для обучающих пар
            var sourceIndices = _trainingDictionary.select(1, 0);
            var targetIndices = _trainingDictionary.select(1, 1);
            
            var sourceEmbeddings = _sourceEmbeddings.forward(sourceIndices);
            var targetEmbeddings = _targetEmbeddings.forward(targetIndices);
            
            // Применяем Прокруст
            _mapping.ApplyProcrustesAlignment(sourceEmbeddings, targetEmbeddings);
            
            var orthogonalityError = _mapping.CheckOrthogonality();
            _logger?.LogInformation($"Применено решение Прокруста. Ошибка ортогональности: {orthogonalityError:E6}");
        }

        #endregion

        #region Public Methods - Model Management

        /// <summary>
        /// Сохраняет лучшую модель если текущая метрика лучше предыдущей
        /// </summary>
        /// <param name="validationMetric">Значение метрики валидации</param>
        /// <param name="metricName">Название метрики</param>
        /// <returns>True если модель была сохранена</returns>
        public Task<bool> SaveBestModelAsync(double validationMetric, string metricName)
        {
            if (validationMetric > _bestValidationMetric)
            {
                _bestValidationMetric = validationMetric;
                _logger?.LogInformation($"* Новое лучшее значение для '{metricName}': {validationMetric:F5}");
                
                // Создаем директорию если не существует
                if (!Directory.Exists(_parameters.ExperimentPath))
                {
                    Directory.CreateDirectory(_parameters.ExperimentPath);
                }
                
                var modelPath = Path.Combine(_parameters.ExperimentPath, "best_mapping.pt");
                _logger?.LogInformation($"* Сохранение модели в {modelPath}...");
                
                // Сохраняем веса модели маппинга
                using var _ = no_grad();
                var weightsToSave = _mapping.Weight.cpu();
                weightsToSave.save(modelPath);
                
                return Task.FromResult(true);
            }
            
            return Task.FromResult(false);
        }

        /// <summary>
        /// Загружает лучшую сохраненную модель
        /// </summary>
        public Task ReloadBestModelAsync()
        {
            //var modelPath = Path.Combine(_parameters.ExperimentPath, "best_mapping.pt");

            //if (!File.Exists(modelPath))
            //    throw new FileNotFoundException($"Файл лучшей модели не найден: {modelPath}");

            //_logger?.LogInformation($"* Загрузка лучшей модели из {modelPath}...");

            //using var _ = no_grad();
            //var loadedWeights = load(modelPath, _parameters.Device);
            //_mapping.Weight.copy_(loadedWeights);

            //_logger?.LogInformation("Лучшая модель успешно загружена");

            return Task.CompletedTask;
        }

        /// <summary>
        /// Обновляет learning rate для SGD оптимизаторов
        /// </summary>
        /// <param name="validationMetric">Текущая метрика валидации</param>
        /// <param name="metricName">Название метрики</param>
        /// <param name="lrDecay">Коэффициент уменьшения learning rate</param>
        /// <param name="lrShrink">Коэффициент сжатия learning rate при ухудшении метрики</param>
        /// <param name="minLr">Минимальный learning rate</param>
        public void UpdateLearningRate(double validationMetric, string metricName, 
            double lrDecay = 0.98, double lrShrink = 0.5, double minLr = 1e-6)
        {
            if (_mappingOptimizer is not SGD sgdOptimizer)
                return;
            
            // Применяем decay
            foreach (var pg in sgdOptimizer.ParamGroups)
            {
                var currentLr = pg.LearningRate;
                var newLr = Math.Max(minLr, currentLr * lrDecay);

                if (newLr < currentLr)
                {
                    pg.LearningRate = newLr;
                    _logger?.LogInformation($"Уменьшение learning rate: {currentLr:E8} -> {newLr:E8}");
                }
            }
            
            
            // Проверяем ухудшение метрики и применяем shrink
            if (lrShrink < 1.0 && validationMetric >= -1e7)
            {
                if (validationMetric < _bestValidationMetric)
                {
                    _logger?.LogInformation($"Метрика валидации хуже лучшей: {validationMetric:F5} vs {_bestValidationMetric:F5}");
                    
                    if (_decreaseLearningRate)
                    {
                        foreach (var pg in sgdOptimizer.ParamGroups)
                        {
                            var shrunkLr = pg.LearningRate * lrShrink;
                            pg.LearningRate = shrunkLr;
                            _logger?.LogInformation($"Сжатие learning rate: {pg.LearningRate:E5} -> {shrunkLr:E5}");
                        }                        
                    }
                    _decreaseLearningRate = true;
                }
            }
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Получает батч данных для обучения дискриминатора
        /// </summary>
        /// <param name="volatile_">Использовать ли volatile_ тензоры</param>
        /// <returns>Кортеж из входных данных и меток</returns>
        private (Tensor x, Tensor y) GetDiscriminatorBatch(bool volatile_ = true)
        {
            var batchSize = _parameters.BatchSize;
            var maxFrequent = Math.Min(_parameters.MostFrequentWords, 
                Math.Min(_sourceDictionary.Count, _targetDictionary.Count));
            
            // Генерируем случайные индексы для исходного и целевого языков
            var sourceIds = randint(0, maxFrequent == 0 ? _sourceDictionary.Count : maxFrequent,
                new long[] { batchSize }, dtype: ScalarType.Int64, device: _parameters.Device);
            var targetIds = randint(0, maxFrequent == 0 ? _targetDictionary.Count : maxFrequent,
                new long[] { batchSize }, dtype: ScalarType.Int64, device: _parameters.Device);
            
            // Получаем эмбеддинги
            var sourceEmb = _sourceEmbeddings.forward(sourceIds);
            var targetEmb = _targetEmbeddings.forward(targetIds);
            
            // Применяем маппинг к исходным эмбеддингам
            var mappedSourceEmb = _mapping.forward(sourceEmb.detach());
            
            // Объединяем исходные (замаппированные) и целевые эмбеддинги
            var x = cat(new[] { mappedSourceEmb, targetEmb.detach() }, dim: 0);
            
            // Создаем метки: 1 - сглаживание для исходного языка, сглаживание для целевого
            var y = zeros(2 * batchSize, dtype: ScalarType.Float32, device: _parameters.Device);
            y[TensorIndex.Slice(null, batchSize)] = (float)(1 - _parameters.DiscriminatorSmoothing);
            y[TensorIndex.Slice(batchSize, null)] = (float)_parameters.DiscriminatorSmoothing;
            
            return (x, y);
        }

        #endregion

        #region IDisposable Implementation

        /// <summary>
        /// Освобождает ресурсы
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Освобождает ресурсы
        /// </summary>
        /// <param name="disposing">Признак освобождения управляемых ресурсов</param>
        private void Dispose(bool disposing)
        {
            if (!_disposed && disposing)
            {
                _mappingOptimizer?.Dispose();
                _discriminatorOptimizer?.Dispose();
                _trainingDictionary?.Dispose();
                _sourceEmbeddings?.Dispose();
                _targetEmbeddings?.Dispose();
                _mapping?.Dispose();
                _discriminator?.Dispose();
                
                _disposed = true;
            }
        }

        #endregion
    }
}