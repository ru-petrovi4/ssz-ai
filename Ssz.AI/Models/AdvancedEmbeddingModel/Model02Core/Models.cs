using System;
using System.Collections.Generic;
using Microsoft.Extensions.Logging;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Models
{
    /// <summary>
    /// Параметры для создания дискриминатора
    /// </summary>
    public sealed record DiscriminatorParameters
    {
        /// <summary>
        /// Размерность эмбеддингов
        /// </summary>
        public int EmbeddingDimension { get; init; } = 300;
        
        /// <summary>
        /// Количество скрытых слоев дискриминатора
        /// </summary>
        public int HiddenLayers { get; init; } = 2;
        
        /// <summary>
        /// Размерность скрытых слоев
        /// </summary>
        public int HiddenDimension { get; init; } = 2048;
        
        /// <summary>
        /// Dropout для скрытых слоев
        /// </summary>
        public double Dropout { get; init; } = 0.0;
        
        /// <summary>
        /// Input dropout
        /// </summary>
        public double InputDropout { get; init; } = 0.1;
    }

    /// <summary>
    /// Дискриминатор для состязательного обучения эмбеддингов
    /// Аналог Discriminator класса из models.py с оптимизациями для .NET 9
    /// </summary>
    public sealed class Discriminator : Module<Tensor, Tensor>
    {
        #region Private Fields

        /// <summary>
        /// Последовательность слоев дискриминатора
        /// </summary>
        private readonly Sequential _layers;

        /// <summary>
        /// Параметры дискриминатора
        /// </summary>
        private readonly new DiscriminatorParameters _parameters;

        #endregion

        #region Constructor

        /// <summary>
        /// Инициализирует новый экземпляр дискриминатора
        /// </summary>
        /// <param name="parameters">Параметры дискриминатора</param>
        /// <param name="name">Имя модуля</param>
        /// <exception cref="ArgumentNullException">Если параметры равны null</exception>
        public Discriminator(DiscriminatorParameters parameters, string name = "discriminator") : base(name)
        {
            _parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            
            // Строим архитектуру дискриминатора
            var layers = new List<Module<Tensor, Tensor>>();
            
            // Input dropout слой
            if (_parameters.InputDropout > 0)
            {
                layers.Add(Dropout(_parameters.InputDropout));
            }
            
            // Строим скрытые слои
            for (int i = 0; i <= _parameters.HiddenLayers; i++)
            {
                int inputDim = i == 0 ? _parameters.EmbeddingDimension : _parameters.HiddenDimension;
                int outputDim = i == _parameters.HiddenLayers ? 1 : _parameters.HiddenDimension;
                
                // Линейный слой
                layers.Add(Linear(inputDim, outputDim));
                
                // Активация и dropout для всех слоев кроме последнего
                if (i < _parameters.HiddenLayers)
                {
                    layers.Add(LeakyReLU(0.2)); // Используем LeakyReLU как в оригинале
                    
                    if (_parameters.Dropout > 0)
                    {
                        layers.Add(Dropout(_parameters.Dropout));
                    }
                }
            }
            
            // Финальная сигмоидная активация
            layers.Add(Sigmoid());
            
            // Создаем последовательную модель
            _layers = Sequential(layers.ToArray());
            
            // Регистрируем подмодуль
            RegisterComponents();
        }

        #endregion

        #region Public Properties

        /// <summary>
        /// Параметры дискриминатора
        /// </summary>
        public DiscriminatorParameters Parameters => _parameters;

        #endregion

        #region Public Methods

        /// <summary>
        /// Прямой проход через дискриминатор
        /// </summary>
        /// <param name="input">Входной тензор эмбеддингов [batch_size, embedding_dim]</param>
        /// <returns>Вероятности принадлежности к целевому языку [batch_size]</returns>
        /// <exception cref="ArgumentException">Если размерности входа некорректны</exception>
        public override Tensor forward(Tensor input)
        {
            // Проверяем размерности входного тензора
            if (input.dim() != 2)
                throw new ArgumentException($"Ожидается 2D тензор, получен {input.dim()}D", nameof(input));
                
            if (input.size(1) != _parameters.EmbeddingDimension)
                throw new ArgumentException(
                    $"Ожидается размерность эмбеддингов {_parameters.EmbeddingDimension}, получена {input.size(1)}", 
                    nameof(input));

            // Пропускаем через последовательность слоев
            var output = _layers.forward(input);
            
            // Возвращаем одномерный тензор вероятностей
            return output.view(-1);
        }

        /// <summary>
        /// Инициализирует веса дискриминатора
        /// </summary>
        /// <param name="seed">Seed для генератора случайных чисел</param>
        public void InitializeWeights(int? seed = null)
        {
            if (seed.HasValue)
            {
                manual_seed(seed.Value);
            }

            // Применяем Xavier/Glorot инициализацию к линейным слоям
            foreach (var module in _layers.modules())
            {
                if (module is Linear linear)
                {
                    init.xavier_uniform_(linear.weight!);
                    if (linear.bias is not null)
                    {
                        init.zeros_(linear.bias);
                    }
                }
            }
        }

        /// <summary>
        /// Обрезает градиенты для стабильности обучения
        /// </summary>
        /// <param name="maxNorm">Максимальная норма градиентов</param>
        public void ClipGradients(double maxNorm)
        {
            if (maxNorm > 0)
            {
                nn.utils.clip_grad_norm_(parameters(), maxNorm);
            }
        }

        /// <summary>
        /// Получает информацию об архитектуре дискриминатора
        /// </summary>
        /// <returns>Строковое описание архитектуры</returns>
        public string GetArchitectureInfo()
        {
            var totalParams = 0L;
            var trainableParams = 0L;
            
            foreach (var param in parameters())
            {
                var paramCount = param.numel();
                totalParams += paramCount;
                if (param.requires_grad)
                {
                    trainableParams += paramCount;
                }
            }
            
            return $"Discriminator Architecture:\n" +
                   $"  Embedding Dimension: {_parameters.EmbeddingDimension}\n" +
                   $"  Hidden Layers: {_parameters.HiddenLayers}\n" +
                   $"  Hidden Dimension: {_parameters.HiddenDimension}\n" +
                   $"  Dropout: {_parameters.Dropout}\n" +
                   $"  Input Dropout: {_parameters.InputDropout}\n" +
                   $"  Total Parameters: {totalParams:N0}\n" +
                   $"  Trainable Parameters: {trainableParams:N0}";
        }

        #endregion

        #region Protected Methods

        /// <summary>
        /// Освобождает ресурсы
        /// </summary>
        /// <param name="disposing">Признак освобождения управляемых ресурсов</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _layers?.Dispose();
            }
            base.Dispose(disposing);
        }

        #endregion
    }

    /// <summary>
    /// Параметры для создания модели выравнивания
    /// </summary>
    public sealed record MappingParameters
    {
        /// <summary>
        /// Размерность эмбеддингов
        /// </summary>
        public int EmbeddingDimension { get; init; } = 300;
        
        /// <summary>
        /// Инициализировать как единичную матрицу
        /// </summary>
        public bool InitializeAsIdentity { get; init; } = true;
        
        /// <summary>
        /// Параметр для ортогонализации (beta)
        /// </summary>
        public double OrthogonalizationBeta { get; init; } = 0.001;
        
        /// <summary>
        /// Использовать ли bias в линейном преобразовании
        /// </summary>
        public bool UseBias { get; init; } = false;
    }

    /// <summary>
    /// Модель линейного преобразования для выравнивания эмбеддингов
    /// Аналог mapping из models.py с поддержкой ортогонализации
    /// </summary>
    public sealed class EmbeddingMapping : Module<Tensor, Tensor>
    {
        #region Private Fields

        /// <summary>
        /// Линейное преобразование
        /// </summary>
        private readonly Linear _mapping;

        /// <summary>
        /// Параметры модели
        /// </summary>
        private readonly new MappingParameters _parameters;

        #endregion

        #region Constructor

        /// <summary>
        /// Инициализирует новую модель выравнивания эмбеддингов
        /// </summary>
        /// <param name="parameters">Параметры модели</param>
        /// <param name="name">Имя модуля</param>
        /// <exception cref="ArgumentNullException">Если параметры равны null</exception>
        public EmbeddingMapping(MappingParameters parameters, string name = "mapping") : base(name)
        {
            _parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            
            // Создаем линейное преобразование
            _mapping = Linear(_parameters.EmbeddingDimension, _parameters.EmbeddingDimension, _parameters.UseBias);
            
            // Инициализируем веса
            InitializeWeights();
            
            // Регистрируем компоненты
            RegisterComponents();
        }

        #endregion

        #region Public Properties

        /// <summary>
        /// Параметры модели выравнивания
        /// </summary>
        public MappingParameters Parameters => _parameters;
        
        /// <summary>
        /// Веса преобразования
        /// </summary>
        public Tensor Weight => _mapping.weight!;

        #endregion

        #region Public Methods

        /// <summary>
        /// Прямой проход через модель выравнивания
        /// </summary>
        /// <param name="input">Входные эмбеддинги [batch_size, embedding_dim] или [embedding_dim]</param>
        /// <returns>Преобразованные эмбеддинги</returns>
        /// <exception cref="ArgumentException">Если размерности некорректны</exception>
        public override Tensor forward(Tensor input)
        {
            // Проверяем размерности
            var inputDim = input.dim();
            var lastDimSize = input.size(-1);
            
            if (lastDimSize != _parameters.EmbeddingDimension)
                throw new ArgumentException(
                    $"Ожидается размерность эмбеддингов {_parameters.EmbeddingDimension}, получена {lastDimSize}", 
                    nameof(input));

            return _mapping.forward(input);
        }

        /// <summary>
        /// Применяет решение задачи ортогонального Прокруста
        /// Находит оптимальную ортогональную матрицу для выравнивания эмбеддингов
        /// </summary>
        /// <param name="sourceEmbeddings">Исходные эмбеддинги [n_pairs, embedding_dim]</param>
        /// <param name="targetEmbeddings">Целевые эмбеддинги [n_pairs, embedding_dim]</param>
        /// <exception cref="ArgumentException">Если размерности не совпадают</exception>
        public void ApplyProcrustesAlignment(Tensor sourceEmbeddings, Tensor targetEmbeddings)
        {
            if (sourceEmbeddings.shape[0] != targetEmbeddings.shape[0])
                throw new ArgumentException("Количество исходных и целевых эмбеддингов должно совпадать");
                
            if (sourceEmbeddings.shape[1] != _parameters.EmbeddingDimension ||
                targetEmbeddings.shape[1] != _parameters.EmbeddingDimension)
                throw new ArgumentException("Размерности эмбеддингов должны соответствовать параметрам модели");

            using var _ = no_grad();
            
            // Вычисляем матрицу M = B^T * A (где A - source, B - target)
            var M = targetEmbeddings.transpose(0, 1).mm(sourceEmbeddings);
            
            // Применяем SVD разложение: M = U * S * V^T
            var (U, S, Vt) = linalg.svd(M, fullMatrices: true);
            
            // Оптимальная ортогональная матрица: W = U * V^T
            var optimalW = U.mm(Vt);
            
            // Обновляем веса модели
            _mapping.weight!.copy_(optimalW);
        }

        /// <summary>
        /// Применяет ортогонализацию к весам модели
        /// Использует итеративную процедуру для поддержания ортогональности
        /// </summary>
        public void OrthogonalizeWeights()
        {
            if (_parameters.OrthogonalizationBeta <= 0)
                return;

            using var _ = no_grad();
            
            var W = _mapping.weight!;
            var beta = _parameters.OrthogonalizationBeta;
            
            // Применяем формулу: W = (1 + β) * W - β * W * W^T * W
            var WtW = W.transpose(0, 1).mm(W);
            var WWtW = W.mm(WtW);
            
            var newW = W.mul(1 + beta).sub(WWtW.mul(beta));
            W.copy_(newW);
        }

        /// <summary>
        /// Инициализирует веса модели
        /// </summary>
        /// <param name="seed">Seed для генератора случайных чисел</param>
        public void InitializeWeights(int? seed = null)
        {
            if (seed.HasValue)
            {
                manual_seed(seed.Value);
            }

            using var _ = no_grad();
            
            if (_parameters.InitializeAsIdentity)
            {
                // Инициализируем как единичную матрицу
                init.eye_(_mapping.weight!);
            }
            else
            {
                // Используем ортогональную инициализацию
                init.orthogonal_(_mapping.weight!);
            }
            
            // Инициализируем bias нулями (если используется)
            if (_mapping.bias is not null)
            {
                init.zeros_(_mapping.bias);
            }
        }

        /// <summary>
        /// Проверяет ортогональность матрицы преобразования
        /// </summary>
        /// <param name="tolerance">Допустимая погрешность</param>
        /// <returns>Степень ортогональности (0 = идеально ортогональная)</returns>
        public double CheckOrthogonality(double tolerance = 1e-6)
        {
            using var _ = no_grad();
            
            var W = _mapping.weight!;
            var WtW = W.transpose(0, 1).mm(W);
            var identity = eye(W.size(0), device: W.device, dtype: W.dtype);
            
            var orthogonalityError = (WtW - identity).norm().item<float>(); // VALFIX
            return orthogonalityError;
        }

        /// <summary>
        /// Получает информацию о модели
        /// </summary>
        /// <returns>Строковое описание модели</returns>
        public string GetModelInfo()
        {
            var paramCount = _mapping.weight!.numel();
            var orthogonalityError = CheckOrthogonality();
            
            return $"Embedding Mapping Model:\n" +
                   $"  Dimension: {_parameters.EmbeddingDimension}x{_parameters.EmbeddingDimension}\n" +
                   $"  Parameters: {paramCount:N0}\n" +
                   $"  Use Bias: {_parameters.UseBias}\n" +
                   $"  Orthogonalization Beta: {_parameters.OrthogonalizationBeta}\n" +
                   $"  Current Orthogonality Error: {orthogonalityError:E6}";
        }

        #endregion

        #region Protected Methods

        /// <summary>
        /// Освобождает ресурсы
        /// </summary>
        /// <param name="disposing">Признак освобождения управляемых ресурсов</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _mapping?.Dispose();
            }
            base.Dispose(disposing);
        }

        #endregion
    }

    /// <summary>
    /// Фабрика для создания моделей
    /// </summary>
    public static class ModelFactory
    {
        /// <summary>
        /// Создает полный набор моделей для кросс-лингвального выравнивания
        /// </summary>
        /// <param name="embeddingDim">Размерность эмбеддингов</param>
        /// <param name="discriminatorParams">Параметры дискриминатора</param>
        /// <param name="mappingParams">Параметры модели выравнивания</param>
        /// <param name="device">Устройство для размещения моделей</param>
        /// <param name="logger">Логгер</param>
        /// <returns>Кортеж из моделей</returns>
        public static (EmbeddingMapping mapping, Discriminator discriminator) CreateModels(
            int embeddingDim,
            DiscriminatorParameters? discriminatorParams = null,
            MappingParameters? mappingParams = null,
            Device? device = null,
            ILogger? logger = null)
        {
            // Используем параметры по умолчанию если не переданы
            discriminatorParams ??= new DiscriminatorParameters { EmbeddingDimension = embeddingDim };
            mappingParams ??= new MappingParameters { EmbeddingDimension = embeddingDim };
            
            // Создаем модели
            var mapping = new EmbeddingMapping(mappingParams);
            var discriminator = new Discriminator(discriminatorParams);
            
            // Перемещаем на устройство если указано
            if (device is not null)
            {
                mapping = mapping.to(device);
                discriminator = discriminator.to(device);
            }
            
            // Инициализируем веса
            mapping.InitializeWeights();
            discriminator.InitializeWeights();
            
            logger?.LogInformation($"Созданы модели на устройстве {device?.type ?? DeviceType.CPU}:");
            logger?.LogInformation(mapping.GetModelInfo());
            logger?.LogInformation(discriminator.GetArchitectureInfo());
            
            return (mapping, discriminator);
        }
    }
}