using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Models;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Utils;
using Microsoft.Extensions.Logging;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using TorchSharp.Modules;
using Ssz.Utils;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Evaluation;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Training
{
    /// <summary>
    /// Параметры для тренера
    /// </summary>
    public interface ITrainerParameters
    {
        /// <summary>
        /// Размер батча для обучения
        /// </summary>
        int BatchSize { get; }
        
        /// <summary>
        /// Количество шагов дискриминатора за одну итерацию
        /// </summary>
        int DisSteps { get; }
        
        /// <summary>
        /// Коэффициент потерь дискриминатора
        /// </summary>
        float DisLambda { get; }
        
        /// <summary>
        /// Сглаживание для дискриминатора
        /// </summary>
        float DisSmooth { get; }
        
        /// <summary>
        /// Количество наиболее частых слов для дискриминации
        /// </summary>
        int DisMostFrequent { get; }
        
        /// <summary>
        /// Обрезание градиентов дискриминатора
        /// </summary>
        float DisClipWeights { get; }
    }

    /// <summary>
    /// Основной класс для обучения кросс-лингвальных эмбеддингов
    /// Аналог Trainer класса из Python проекта с оптимизациями для .NET 9
    /// </summary>
    public sealed class Trainer : IDisposable
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
        private readonly Mapping _mapping;
        
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
        private readonly ITrainerParameters _trainerParameters;        

        /// <summary>
        /// Путь для сохранения экспериментов
        /// </summary>
        private readonly string _experimentPath;

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
        /// (self.dico)
        /// </summary>
        private Tensor? _trainingDictionary;
        
        /// <summary>
        /// Лучшая метрика валидации
        /// </summary>
        private float _bestValidationMetric = float.NegativeInfinity;
        
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
        /// <param name="trainerParameters">Параметры тренера</param>
        /// <param name="logger">Логгер</param>
        public Trainer(
            Embedding sourceEmbeddings,
            Embedding targetEmbeddings,
            Mapping mapping,
            Discriminator? discriminator,
            Dictionary sourceDictionary,
            Dictionary targetDictionary,
            ITrainerParameters trainerParameters,
            Device device,
            string experimentPath,
            ILogger? logger = null)
        {
            _sourceEmbeddings = sourceEmbeddings ?? throw new ArgumentNullException(nameof(sourceEmbeddings));
            _targetEmbeddings = targetEmbeddings ?? throw new ArgumentNullException(nameof(targetEmbeddings));
            _mapping = mapping ?? throw new ArgumentNullException(nameof(mapping));
            _discriminator = discriminator;
            _sourceDictionary = sourceDictionary ?? throw new ArgumentNullException(nameof(sourceDictionary));
            _targetDictionary = targetDictionary ?? throw new ArgumentNullException(nameof(targetDictionary));
            _trainerParameters = trainerParameters;
            Device = device;
            _experimentPath = experimentPath;
            _logger = logger;
        }

        #endregion

        #region Public Properties

        public const string FileName_Best_Mapping = "best_mapping.pt";

        /// <summary>
        /// Устройство для обучения
        /// </summary>
        public Device Device { get; }

        /// <summary>
        /// Эмбеддинги исходного языка
        /// </summary>
        public Embedding SourceEmbeddings => _sourceEmbeddings;

        /// <summary>
        /// Эмбеддинги целевого языка
        /// </summary>
        public Embedding TargetEmbeddings => _targetEmbeddings;

        /// <summary>
        /// Словарь исходного языка
        /// </summary>
        public Dictionary SourceDictionary => _sourceDictionary;

        /// <summary>
        /// Словарь целевого языка
        /// </summary>
        public Dictionary TargetDictionary => _targetDictionary;

        /// <summary>
        /// Лучшая метрика валидации
        /// </summary>
        public float BestValidationMetric => _bestValidationMetric;

        /// <summary>
        /// Оптимизатор для модели выравнивания
        /// </summary>
        public Optimizer? MappingOptimizer => _mappingOptimizer;

        /// <summary>
        /// Оптимизатор для дискриминатора
        /// </summary>
        public Optimizer? DiscriminatorOptimizer => _discriminatorOptimizer;

        /// <summary>
        /// Модель выравнивания
        /// </summary>
        public Mapping Mapping => _mapping;

        /// <summary>
        /// Дискриминатор (может быть null для supervised обучения)
        /// </summary>
        public Discriminator? Discriminator => _discriminator;

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
            var nameValues = NameValueCollectionHelper.Parse(config);
            var optimizerType = (nameValues.TryGetValue(@"") ?? @"").ToLowerInvariant();            
            
            // Создаем оптимизатор в зависимости от типа
            return optimizerType switch
            {
                "sgd" => SGD(parameters, 
                    learningRate: ConfigurationHelper.GetValue<float>(nameValues, "lr", 0.01f),
                    momentum: ConfigurationHelper.GetValue<float>(nameValues, "momentum", 0.0f),
                    weight_decay: ConfigurationHelper.GetValue<float>(nameValues, "weight_decay", 0.0f)),
                    
                "adam" => Adam(parameters,
                    lr: ConfigurationHelper.GetValue<float>(nameValues, "lr", 0.001f),
                    weight_decay: ConfigurationHelper.GetValue<float>(nameValues, "weight_decay", 0.0f)),
                    
                "adagrad" => Adagrad(parameters,
                    lr: ConfigurationHelper.GetValue<float>(nameValues, "lr", 0.01f),
                    weight_decay: ConfigurationHelper.GetValue<float>(nameValues, "weight_decay", 0.0f)),
                    
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
            
            return tensor(dictionaryData, dtype: ScalarType.Int64, device: Device)
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
            
            return tensor(dictionaryData, dtype: ScalarType.Int64, device: Device)
                .reshape(pairs.Count, 2);
        }

        #endregion

        #region Public Methods - Training Steps

        /// <summary>
        /// Выполняет шаг обучения дискриминатора
        /// </summary>
        /// <param name="stats">Статистики для записи потерь</param>
        /// <returns>Потери дискриминатора</returns>
        public float DiscriminatorStep(TrainingStats stats)
        {
            if (_discriminator == null || _discriminatorOptimizer == null)
                throw new InvalidOperationException("Дискриминатор или его оптимизатор не настроены");
            
            _discriminator.train();
            
            // Получаем батч данных для дискриминатора
            var (x, y) = GetDiscriminatorBatch(volatile_: true);
            
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
            if (_trainerParameters.DisClipWeights > 0)
            {
                _discriminator.ClipGradients(_trainerParameters.DisClipWeights);
            }
            
            var lossValue = loss.item<float>();
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
            if (_trainerParameters.DisLambda == 0.0)
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
            loss = loss * _trainerParameters.DisLambda;
            
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
            
            stats.MappingLosses.Add(loss.item<float>());
            return 2 * _trainerParameters.BatchSize;
        }

        public async Task BuildDictionaryAsync(IDictionaryBuilderParameters parameters)
        {
            torch.Tensor sourceEmbeddings = Mapping.forward(SourceEmbeddings.weight!);
            torch.Tensor targetEmbeddings = TargetEmbeddings.weight!;

            sourceEmbeddings = sourceEmbeddings / sourceEmbeddings.norm(p: 2, dim: 1, keepdim: true).expand_as(sourceEmbeddings);
            targetEmbeddings = targetEmbeddings / targetEmbeddings.norm(p: 2, dim: 1, keepdim: true).expand_as(targetEmbeddings);
            
            // Строим словарь
            _trainingDictionary = await DictionaryBuilder.BuildDictionaryAsync(
                sourceEmbeddings,
                targetEmbeddings,
                parameters,
                logger: _logger);
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
        /// <param name="stats">Значение метрики валидации</param>
        /// <param name="metricName">Название метрики</param>
        /// <returns>True если модель была сохранена</returns>
        public Task<bool> SaveBestMappingWeightsAsync(TrainingStats stats, string metricName)
        {
            float validationMetric = stats.ToLog.TryGetValue(metricName);
            if (validationMetric > _bestValidationMetric)
            {
                _bestValidationMetric = validationMetric;
                _logger?.LogInformation($"* Новое лучшее значение для '{metricName}': {validationMetric:F5}");
                
                // Создаем директорию если не существует
                if (!Directory.Exists(_experimentPath))
                {
                    Directory.CreateDirectory(_experimentPath);
                }
                
                var modelPath = Path.Combine(_experimentPath, FileName_Best_Mapping);
                _logger?.LogInformation($"* Сохранение модели в {modelPath}...");
                
                // Сохраняем веса модели маппинга
                using var _ = no_grad();
                var weightsToSave = _mapping.MappingLinear.weight.cpu();
                weightsToSave.save(modelPath);
                
                return Task.FromResult(true);
            }
            
            return Task.FromResult(false);
        }

        /// <summary>
        /// Загружает лучшую сохраненную модель
        /// </summary>
        public Task ReloadBestMappingWeightsAsync()
        {
            var modelPath = Path.Combine(_experimentPath, FileName_Best_Mapping);

            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"Файл лучшей модели не найден: {modelPath}");

            _logger?.LogInformation($"* Загрузка лучшей модели из {modelPath}...");

            using var _ = no_grad();
            var loadedWeights = load(modelPath);
            _mapping.MappingLinear.weight!.copy_(loadedWeights);

            _logger?.LogInformation("Лучшая модель успешно загружена");

            return Task.CompletedTask;
        }

        /// <summary>
        /// Обновляет learning rate для SGD оптимизаторов
        /// </summary>
        /// <param name="trainingStats">Текущая метрика валидации</param>
        /// <param name="metricName">Название метрики</param>
        /// <param name="lrDecay">Коэффициент уменьшения learning rate</param>
        /// <param name="lrShrink">Коэффициент сжатия learning rate при ухудшении метрики</param>
        /// <param name="minLr">Минимальный learning rate</param>
        public void UpdateLearningRate(TrainingStats trainingStats, string metricName, 
            float lrDecay = 0.98f, float lrShrink = 0.5f, float minLr = 1e-6f)
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
            float validationMetric = trainingStats.ToLog.TryGetValue(metricName);
            if (lrShrink < 1.0f && validationMetric >= -1e7f)
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
            var batchSize = _trainerParameters.BatchSize;
            var maxFrequent = Math.Min(_trainerParameters.DisMostFrequent, 
                Math.Min(_sourceDictionary.Count, _targetDictionary.Count));
            
            // Генерируем случайные индексы для исходного и целевого языков
            var sourceIds = randint(low: 0, high: maxFrequent == 0 ? _sourceDictionary.Count : maxFrequent,
                size: new long[] { batchSize }, dtype: ScalarType.Int64, device: Device);
            var targetIds = randint(low: 0, high: maxFrequent == 0 ? _targetDictionary.Count : maxFrequent,
                size: new long[] { batchSize }, dtype: ScalarType.Int64, device: Device);
            
            // Получаем эмбеддинги
            var sourceEmb = _sourceEmbeddings.forward(sourceIds);
            var targetEmb = _targetEmbeddings.forward(targetIds);
            
            // Применяем маппинг к исходным эмбеддингам
            var mappedSourceEmb = _mapping.forward(sourceEmb.detach());
            
            // Объединяем исходные (замаппированные) и целевые эмбеддинги
            var x = cat(new[] { mappedSourceEmb, targetEmb.detach() }, dim: 0);
            
            // Создаем метки: 1 - сглаживание для исходного языка, сглаживание для целевого
            var y = zeros(2 * batchSize, dtype: ScalarType.Float32, device: Device);
            y[TensorIndex.Slice(start: null, stop: batchSize)] = 1.0f - _trainerParameters.DisSmooth;
            y[TensorIndex.Slice(start: batchSize, stop: null)] = _trainerParameters.DisSmooth;
            
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