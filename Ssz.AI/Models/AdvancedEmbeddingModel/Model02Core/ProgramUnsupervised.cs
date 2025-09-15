//using System;
//using System.CommandLine;
//using System.CommandLine.Parsing;
//using System.IO;
//using System.Threading.Tasks;
//using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core;
//using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Models;
//using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Training;
//using Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Utils;
//using Microsoft.Extensions.Logging;
//using TorchSharp;
//using static TorchSharp.torch;

//namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core.Programs
//{
//    /// <summary>
//    /// Основная программа для обучения без учителя (unsupervised training)
//    /// Аналог unsupervised.py из Python проекта с использованием System.CommandLine
//    /// </summary>
//    public static class UnsupervisedProgram
//    {
//        #region Constants
        
//        /// <summary>
//        /// Метрика валидации для unsupervised обучения
//        /// </summary>
//        private const string ValidationMetric = "mean_cosine-csls_knn_10-S2T-10000";
        
//        #endregion

//        #region Main Method

//        /// <summary>
//        /// Точка входа для unsupervised обучения
//        /// </summary>
//        /// <param name="args">Аргументы командной строки</param>
//        public static async Task<int> Main(string[] args)
//        {
//            var rootCommand = BuildCommandLineInterface();
//            return await rootCommand.InvokeAsync(args);
//        }

//        #endregion

//        #region Command Line Interface

//        /// <summary>
//        /// Строит интерфейс командной строки
//        /// </summary>
//        /// <returns>Корневая команда</returns>
//        private static RootCommand BuildCommandLineInterface()
//        {
//            var rootCommand = new RootCommand("Обучение кросс-лингвальных эмбеддингов без учителя")
//            {
//                // Основные параметры эксперимента
//                new Option<int>("--seed", () => -1, "Seed для инициализации (-1 для случайного)"),
//                new Option<int>("--verbose", () => 2, "Уровень подробности (2:debug, 1:info, 0:warning)"),
//                new Option<string>("--exp-path", () => "", "Путь для сохранения экспериментов"),
//                new Option<string>("--exp-name", () => "debug", "Название эксперимента"),
//                new Option<string>("--exp-id", () => "", "ID эксперимента"),
//                new Option<bool>("--cuda", () => true, "Использовать GPU"),
//                new Option<string>("--export", () => "txt", "Формат экспорта эмбеддингов (txt / pth)"),

//                // Параметры данных
//                new Option<string>("--src-lang", () => "en", "Исходный язык"),
//                new Option<string>("--tgt-lang", () => "es", "Целевой язык"),
//                new Option<int>("--emb-dim", () => 300, "Размерность эмбеддингов"),
//                new Option<int>("--max-vocab", () => 200000, "Максимальный размер словаря (-1 для неограниченного)"),

//                // Параметры маппинга
//                new Option<bool>("--map-id-init", () => true, "Инициализировать маппинг как единичную матрицу"),
//                new Option<double>("--map-beta", () => 0.001, "Параметр бета для ортогонализации"),

//                // Параметры дискриминатора
//                new Option<int>("--dis-layers", () => 2, "Количество слоев дискриминатора"),
//                new Option<int>("--dis-hid-dim", () => 2048, "Размерность скрытых слоев дискриминатора"),
//                new Option<double>("--dis-dropout", () => 0.0, "Dropout дискриминатора"),
//                new Option<double>("--dis-input-dropout", () => 0.1, "Input dropout дискриминатора"),
//                new Option<int>("--dis-steps", () => 5, "Количество шагов дискриминатора"),
//                new Option<double>("--dis-lambda", () => 1.0, "Коэффициент потерь дискриминатора"),
//                new Option<int>("--dis-most-frequent", () => 75000, "Количество наиболее частых слов для дискриминации"),
//                new Option<double>("--dis-smooth", () => 0.1, "Сглаживание предсказаний дискриминатора"),
//                new Option<double>("--dis-clip-weights", () => 0.0, "Обрезание весов дискриминатора"),

//                // Параметры состязательного обучения
//                new Option<bool>("--adversarial", () => true, "Использовать состязательное обучение"),
//                new Option<int>("--n-epochs", () => 5, "Количество эпох"),
//                new Option<int>("--epoch-size", () => 1000000, "Итераций на эпоху"),
//                new Option<int>("--batch-size", () => 32, "Размер батча"),
//                new Option<string>("--map-optimizer", () => "sgd,lr=0.1", "Оптимизатор маппинга"),
//                new Option<string>("--dis-optimizer", () => "sgd,lr=0.1", "Оптимизатор дискриминатора"),
//                new Option<double>("--lr-decay", () => 0.98, "Уменьшение learning rate (только SGD)"),
//                new Option<double>("--min-lr", () => 1e-6, "Минимальный learning rate (только SGD)"),
//                new Option<double>("--lr-shrink", () => 0.5, "Сжатие learning rate при ухудшении метрики"),

//                // Параметры refinement
//                new Option<int>("--n-refinement", () => 5, "Количество итераций refinement (0 для отключения)"),

//                // Параметры построения словаря
//                new Option<string>("--dico-eval", () => "default", "Путь к словарю для оценки"),
//                new Option<string>("--dico-method", () => "csls_knn_10", "Метод построения словаря"),
//                new Option<string>("--dico-build", () => "S2T", "Режим построения словаря"),
//                new Option<double>("--dico-threshold", () => 0.0, "Порог уверенности для построения словаря"),
//                new Option<int>("--dico-max-rank", () => 15000, "Максимальный ранг слов в словаре"),
//                new Option<int>("--dico-min-size", () => 0, "Минимальный размер создаваемого словаря"),
//                new Option<int>("--dico-max-size", () => 0, "Максимальный размер создаваемого словаря"),

//                // Параметры эмбеддингов
//                new Option<string>("--src-emb", () => "", "Путь к исходным эмбеддингам"),
//                new Option<string>("--tgt-emb", () => "", "Путь к целевым эмбеддингам"),
//                new Option<string>("--normalize-embeddings", () => "", "Нормализация эмбеддингов перед обучением")
//            };

//            rootCommand.SetHandler(ExecuteUnsupervisedTrainingAsync);
//            return rootCommand;
//        }

//        #endregion

//        #region Main Training Logic

//        /// <summary>
//        /// Выполняет unsupervised обучение
//        /// </summary>
//        /// <param name="context">Контекст вызова команды</param>
//        private static async Task ExecuteUnsupervisedTrainingAsync(InvocationContext context)
//        {
//            try
//            {
//                // Парсим параметры
//                var parameters = ParseParameters(context);
                
//                // Валидация параметров
//                ValidateParameters(parameters);
                
//                // Инициализация эксперимента
//                var logger = InitializeExperiment(parameters);
                
//                // Загрузка данных и создание моделей
//                var (sourceEmbeddings, targetEmbeddings, mapping, discriminator, sourceDictionary, targetDictionary) = 
//                    await LoadDataAndCreateModelsAsync(parameters, logger);
                
//                // Создание тренера
//                using var trainer = CreateTrainer(sourceEmbeddings, targetEmbeddings, mapping, discriminator, 
//                    sourceDictionary, targetDictionary, parameters, logger);
                
//                // Состязательное обучение
//                if (parameters.Adversarial)
//                {
//                    await RunAdversarialTrainingAsync(trainer, parameters, logger);
//                }
                
//                // Procrustes refinement
//                if (parameters.NRefinement > 0)
//                {
//                    await RunProcrustesRefinementAsync(trainer, parameters, logger);
//                }
                
//                // Экспорт финальных эмбеддингов
//                if (!string.IsNullOrEmpty(parameters.Export))
//                {
//                    await ExportFinalEmbeddingsAsync(trainer, parameters, logger);
//                }
                
//                logger.LogInformation("Обучение успешно завершено!");
//            }
//            catch (Exception ex)
//            {
//                Console.Error.WriteLine($"Ошибка: {ex.Message}");
//                if (ex.InnerException != null)
//                {
//                    Console.Error.WriteLine($"Внутренняя ошибка: {ex.InnerException.Message}");
//                }
//                Environment.Exit(1);
//            }
//        }

//        #endregion

//        #region Parameter Parsing and Validation

//        /// <summary>
//        /// Параметры unsupervised обучения
//        /// </summary>
//        private sealed record UnsupervisedParameters
//        {
//            public int Seed { get; init; }
//            public int Verbose { get; init; }
//            public string ExpPath { get; init; } = "";
//            public string ExpName { get; init; } = "debug";
//            public string ExpId { get; init; } = "";
//            public bool Cuda { get; init; }
//            public string Export { get; init; } = "txt";
            
//            // Data parameters
//            public string SrcLang { get; init; } = "en";
//            public string TgtLang { get; init; } = "es";
//            public int EmbDim { get; init; } = 300;
//            public int MaxVocab { get; init; } = 200000;
            
//            // Mapping parameters
//            public bool MapIdInit { get; init; }
//            public double MapBeta { get; init; }
            
//            // Discriminator parameters
//            public int DisLayers { get; init; }
//            public int DisHidDim { get; init; }
//            public double DisDropout { get; init; }
//            public double DisInputDropout { get; init; }
//            public int DisSteps { get; init; }
//            public double DisLambda { get; init; }
//            public int DisMostFrequent { get; init; }
//            public double DisSmooth { get; init; }
//            public double DisClipWeights { get; init; }
            
//            // Training parameters
//            public bool Adversarial { get; init; }
//            public int NEpochs { get; init; }
//            public int EpochSize { get; init; }
//            public int BatchSize { get; init; }
//            public string MapOptimizer { get; init; } = "sgd,lr=0.1";
//            public string DisOptimizer { get; init; } = "sgd,lr=0.1";
//            public double LrDecay { get; init; }
//            public double MinLr { get; init; }
//            public double LrShrink { get; init; }
            
//            // Refinement parameters
//            public int NRefinement { get; init; }
            
//            // Dictionary parameters
//            public string DicoEval { get; init; } = "default";
//            public string DicoMethod { get; init; } = "csls_knn_10";
//            public string DicoBuild { get; init; } = "S2T";
//            public double DicoThreshold { get; init; }
//            public int DicoMaxRank { get; init; }
//            public int DicoMinSize { get; init; }
//            public int DicoMaxSize { get; init; }
            
//            // Embeddings parameters
//            public string SrcEmb { get; init; } = "";
//            public string TgtEmb { get; init; } = "";
//            public string NormalizeEmbeddings { get; init; } = "";
//        }

//        /// <summary>
//        /// Парсит параметры из контекста команды
//        /// </summary>
//        /// <param name="context">Контекст команды</param>
//        /// <returns>Параметры обучения</returns>
//        private static UnsupervisedParameters ParseParameters(InvocationContext context)
//        {
//            var result = context.ParseResult;
            
//            return new UnsupervisedParameters
//            {
//                Seed = result.GetValueForOption<int>("--seed"),
//                Verbose = result.GetValueForOption<int>("--verbose"),
//                ExpPath = result.GetValueForOption<string>("--exp-path") ?? "",
//                ExpName = result.GetValueForOption<string>("--exp-name") ?? "debug",
//                ExpId = result.GetValueForOption<string>("--exp-id") ?? "",
//                Cuda = result.GetValueForOption<bool>("--cuda"),
//                Export = result.GetValueForOption<string>("--export") ?? "txt",
                
//                SrcLang = result.GetValueForOption<string>("--src-lang") ?? "en",
//                TgtLang = result.GetValueForOption<string>("--tgt-lang") ?? "es",
//                EmbDim = result.GetValueForOption<int>("--emb-dim"),
//                MaxVocab = result.GetValueForOption<int>("--max-vocab"),
                
//                MapIdInit = result.GetValueForOption<bool>("--map-id-init"),
//                MapBeta = result.GetValueForOption<double>("--map-beta"),
                
//                DisLayers = result.GetValueForOption<int>("--dis-layers"),
//                DisHidDim = result.GetValueForOption<int>("--dis-hid-dim"),
//                DisDropout = result.GetValueForOption<double>("--dis-dropout"),
//                DisInputDropout = result.GetValueForOption<double>("--dis-input-dropout"),
//                DisSteps = result.GetValueForOption<int>("--dis-steps"),
//                DisLambda = result.GetValueForOption<double>("--dis-lambda"),
//                DisMostFrequent = result.GetValueForOption<int>("--dis-most-frequent"),
//                DisSmooth = result.GetValueForOption<double>("--dis-smooth"),
//                DisClipWeights = result.GetValueForOption<double>("--dis-clip-weights"),
                
//                Adversarial = result.GetValueForOption<bool>("--adversarial"),
//                NEpochs = result.GetValueForOption<int>("--n-epochs"),
//                EpochSize = result.GetValueForOption<int>("--epoch-size"),
//                BatchSize = result.GetValueForOption<int>("--batch-size"),
//                MapOptimizer = result.GetValueForOption<string>("--map-optimizer") ?? "sgd,lr=0.1",
//                DisOptimizer = result.GetValueForOption<string>("--dis-optimizer") ?? "sgd,lr=0.1",
//                LrDecay = result.GetValueForOption<double>("--lr-decay"),
//                MinLr = result.GetValueForOption<double>("--min-lr"),
//                LrShrink = result.GetValueForOption<double>("--lr-shrink"),
                
//                NRefinement = result.GetValueForOption<int>("--n-refinement"),
                
//                DicoEval = result.GetValueForOption<string>("--dico-eval") ?? "default",
//                DicoMethod = result.GetValueForOption<string>("--dico-method") ?? "csls_knn_10",
//                DicoBuild = result.GetValueForOption<string>("--dico-build") ?? "S2T",
//                DicoThreshold = result.GetValueForOption<double>("--dico-threshold"),
//                DicoMaxRank = result.GetValueForOption<int>("--dico-max-rank"),
//                DicoMinSize = result.GetValueForOption<int>("--dico-min-size"),
//                DicoMaxSize = result.GetValueForOption<int>("--dico-max-size"),
                
//                SrcEmb = result.GetValueForOption<string>("--src-emb") ?? "",
//                TgtEmb = result.GetValueForOption<string>("--tgt-emb") ?? "",
//                NormalizeEmbeddings = result.GetValueForOption<string>("--normalize-embeddings") ?? ""
//            };
//        }

//        /// <summary>
//        /// Валидирует параметры обучения
//        /// </summary>
//        /// <param name="parameters">Параметры для валидации</param>
//        /// <exception cref="ArgumentException">При некорректных параметрах</exception>
//        private static void ValidateParameters(UnsupervisedParameters parameters)
//        {
//            // Проверяем CUDA доступность
//            if (parameters.Cuda && !cuda.is_available())
//                throw new ArgumentException("CUDA не доступна, но была запрошена");
            
//            // Проверяем параметры dropout
//            if (parameters.DisDropout < 0 || parameters.DisDropout >= 1)
//                throw new ArgumentException("dis-dropout должен быть в диапазоне [0, 1)");
                
//            if (parameters.DisInputDropout < 0 || parameters.DisInputDropout >= 1)
//                throw new ArgumentException("dis-input-dropout должен быть в диапазоне [0, 1)");
                
//            if (parameters.DisSmooth < 0 || parameters.DisSmooth >= 0.5)
//                throw new ArgumentException("dis-smooth должен быть в диапазоне [0, 0.5)");
            
//            // Проверяем параметры дискриминатора
//            if (parameters.DisLambda <= 0 || parameters.DisSteps <= 0)
//                throw new ArgumentException("dis-lambda и dis-steps должны быть положительными");
            
//            // Проверяем learning rate параметры
//            if (parameters.LrShrink <= 0 || parameters.LrShrink > 1)
//                throw new ArgumentException("lr-shrink должен быть в диапазоне (0, 1]");
            
//            // Проверяем файлы эмбеддингов
//            if (string.IsNullOrEmpty(parameters.SrcEmb) || !File.Exists(parameters.SrcEmb))
//                throw new ArgumentException($"Файл исходных эмбеддингов не найден: {parameters.SrcEmb}");
                
//            if (string.IsNullOrEmpty(parameters.TgtEmb) || !File.Exists(parameters.TgtEmb))
//                throw new ArgumentException($"Файл целевых эмбеддингов не найден: {parameters.TgtEmb}");
            
//            // Проверяем словарь для оценки
//            if (parameters.DicoEval != "default" && !File.Exists(parameters.DicoEval))
//                throw new ArgumentException($"Файл словаря для оценки не найден: {parameters.DicoEval}");
            
//            // Проверяем формат экспорта
//            if (!string.IsNullOrEmpty(parameters.Export) && 
//                parameters.Export != "txt" && parameters.Export != "pth")
//                throw new ArgumentException("Формат экспорта должен быть 'txt' или 'pth'");
//        }

//        #endregion

//        #region Helper Methods

//        /// <summary>
//        /// Инициализирует эксперимент и создает логгер
//        /// </summary>
//        /// <param name="parameters">Параметры эксперимента</param>
//        /// <returns>Настроенный логгер</returns>
//        private static ILogger InitializeExperiment(UnsupervisedParameters parameters)
//        {
//            // Устанавливаем seed если указан
//            if (parameters.Seed >= 0)
//            {
//                manual_seed(parameters.Seed);
//                if (parameters.Cuda)
//                {
//                    cuda.manual_seed(parameters.Seed);
//                }
//            }
            
//            // Создаем путь для эксперимента
//            var expPath = GetExperimentPath(parameters);
            
//            // Создаем логгер
//            var logFile = Path.Combine(expPath, "train.log");
//            var logger = LoggerFactory.CreateLogger(logFile, parameters.Verbose);
            
//            logger.LogSeparator("Инициализация эксперимента");
//            logger.LogInformation($"Путь эксперимента: {expPath}");
//            logger.LogInformation($"Устройство: {(parameters.Cuda ? "CUDA" : "CPU")}");
//            logger.LogInformation($"Seed: {(parameters.Seed >= 0 ? parameters.Seed.ToString() : "случайный")}");
            
//            return logger;
//        }

//        /// <summary>
//        /// Получает путь для сохранения эксперимента
//        /// </summary>
//        /// <param name="parameters">Параметры эксперимента</param>
//        /// <returns>Путь к директории эксперимента</returns>
//        private static string GetExperimentPath(UnsupervisedParameters parameters)
//        {
//            var basePath = string.IsNullOrEmpty(parameters.ExpPath) ? "./dumped" : parameters.ExpPath;
            
//            if (!Directory.Exists(basePath))
//                Directory.CreateDirectory(basePath);
            
//            var expFolder = Path.Combine(basePath, parameters.ExpName);
//            if (!Directory.Exists(expFolder))
//                Directory.CreateDirectory(expFolder);
            
//            string expPath;
//            if (string.IsNullOrEmpty(parameters.ExpId))
//            {
//                // Генерируем случайный ID
//                var chars = "abcdefghijklmnopqrstuvwxyz0123456789";
//                var random = new Random();
//                string expId;
//                do
//                {
//                    expId = new string(Enumerable.Repeat(chars, 10)
//                        .Select(s => s[random.Next(s.Length)]).ToArray());
//                    expPath = Path.Combine(expFolder, expId);
//                } while (Directory.Exists(expPath));
//            }
//            else
//            {
//                expPath = Path.Combine(expFolder, parameters.ExpId);
//                if (Directory.Exists(expPath))
//                    throw new ArgumentException($"Директория эксперимента уже существует: {expPath}");
//            }
            
//            Directory.CreateDirectory(expPath);
//            return expPath;
//        }

//        /// <summary>
//        /// Загружает данные и создает модели
//        /// </summary>
//        /// <param name="parameters">Параметры</param>
//        /// <param name="logger">Логгер</param>
//        /// <returns>Загруженные данные и модели</returns>
//        private static async Task<(nn.Embedding sourceEmbeddings, nn.Embedding targetEmbeddings, 
//            EmbeddingMapping mapping, Discriminator discriminator, 
//            Dictionary sourceDictionary, Dictionary targetDictionary)> 
//            LoadDataAndCreateModelsAsync(UnsupervisedParameters parameters, ILogger logger)
//        {
//            logger.LogInformation("Загрузка эмбеддингов и создание моделей...");
            
//            // Устройство для размещения тензоров
//            var device = parameters.Cuda ? CUDA : CPU;
            
//            // Загружаем исходные эмбеддинги
//            var (sourceDictionary, sourceEmbeddingMatrix) = await EmbeddingUtils.LoadTextEmbeddingsAsync(
//                parameters.SrcEmb, parameters.MaxVocab, parameters.EmbDim, true, logger);
            
//            // Загружаем целевые эмбеддинги
//            var (targetDictionary, targetEmbeddingMatrix) = await EmbeddingUtils.LoadTextEmbeddingsAsync(
//                parameters.TgtEmb, parameters.MaxVocab, parameters.EmbDim, true, logger);
            
//            // Создаем тензоры эмбеддингов
//            var sourceTensor = tensor(sourceEmbeddingMatrix.Data)
//                .reshape(sourceEmbeddingMatrix.RowsCount, sourceEmbeddingMatrix.ColumnsCount)
//                .to(device);
//            var targetTensor = tensor(targetEmbeddingMatrix.Data)
//                .reshape(targetEmbeddingMatrix.RowsCount, targetEmbeddingMatrix.ColumnsCount)
//                .to(device);
            
//            // Создаем embedding слои
//            var sourceEmbeddings = nn.Embedding(sourceDictionary.Count, parameters.EmbDim, sparse: true);
//            var targetEmbeddings = nn.Embedding(targetDictionary.Count, parameters.EmbDim, sparse: true);
            
//            sourceEmbeddings.weight.copy_(sourceTensor);
//            targetEmbeddings.weight.copy_(targetTensor);
            
//            sourceEmbeddings.to(device);
//            targetEmbeddings.to(device);
            
//            // Применяем нормализацию если указана
//            if (!string.IsNullOrEmpty(parameters.NormalizeEmbeddings))
//            {
//                EmbeddingUtils.NormalizeEmbeddings(sourceEmbeddingMatrix, parameters.NormalizeEmbeddings);
//                EmbeddingUtils.NormalizeEmbeddings(targetEmbeddingMatrix, parameters.NormalizeEmbeddings);
                
//                // Обновляем веса после нормализации
//                sourceEmbeddings.weight.copy_(tensor(sourceEmbeddingMatrix.Data)
//                    .reshape(sourceEmbeddingMatrix.RowsCount, sourceEmbeddingMatrix.ColumnsCount));
//                targetEmbeddings.weight.copy_(tensor(targetEmbeddingMatrix.Data)
//                    .reshape(targetEmbeddingMatrix.RowsCount, targetEmbeddingMatrix.ColumnsCount));
//            }
            
//            // Создаем модели
//            var mappingParams = new MappingParameters
//            {
//                EmbeddingDimension = parameters.EmbDim,
//                InitializeAsIdentity = parameters.MapIdInit,
//                OrthogonalizationBeta = parameters.MapBeta
//            };
            
//            var discriminatorParams = new DiscriminatorParameters
//            {
//                EmbeddingDimension = parameters.EmbDim,
//                HiddenLayers = parameters.DisLayers,
//                HiddenDimension = parameters.DisHidDim,
//                Dropout = parameters.DisDropout,
//                InputDropout = parameters.DisInputDropout
//            };
            
//            var (mapping, discriminator) = ModelFactory.CreateModels(
//                parameters.EmbDim, discriminatorParams, mappingParams, device, logger);
            
//            logger.LogInformation("Модели успешно созданы и инициализированы");
            
//            return (sourceEmbeddings, targetEmbeddings, mapping, discriminator, sourceDictionary, targetDictionary);
//        }

//        /// <summary>
//        /// Создает тренер
//        /// </summary>
//        private static CrossLingualTrainer CreateTrainer(
//            nn.Embedding sourceEmbeddings, nn.Embedding targetEmbeddings,
//            EmbeddingMapping mapping, Discriminator discriminator,
//            Dictionary sourceDictionary, Dictionary targetDictionary,
//            UnsupervisedParameters parameters, ILogger logger)
//        {
//            var trainerParams = new TrainerParameters
//            {
//                BatchSize = parameters.BatchSize,
//                DiscriminatorSteps = parameters.DisSteps,
//                DiscriminatorLambda = parameters.DisLambda,
//                DiscriminatorSmoothing = parameters.DisSmooth,
//                MostFrequentWords = parameters.DisMostFrequent,
//                DiscriminatorClipWeights = parameters.DisClipWeights,
//                Device = parameters.Cuda ? CUDA : CPU,
//                ExperimentPath = GetExperimentPath(parameters)
//            };
            
//            var trainer = new CrossLingualTrainer(
//                sourceEmbeddings, targetEmbeddings, mapping, discriminator,
//                sourceDictionary, targetDictionary, trainerParams, logger);
            
//            // Настраиваем оптимизаторы
//            trainer.SetupOptimizers(parameters.MapOptimizer, parameters.DisOptimizer);
            
//            return trainer;
//        }

//        /// <summary>
//        /// Запускает состязательное обучение
//        /// </summary>
//        private static async Task RunAdversarialTrainingAsync(CrossLingualTrainer trainer, 
//            UnsupervisedParameters parameters, ILogger logger)
//        {
//            logger.LogSeparator("СОСТЯЗАТЕЛЬНОЕ ОБУЧЕНИЕ");
            
//            var stats = new TrainingStats();
//            var startTime = DateTime.UtcNow;
            
//            for (int epoch = 0; epoch < parameters.NEpochs; epoch++)
//            {
//                logger.LogInformation($"Начало эпохи состязательного обучения {epoch}...");
                
//                var epochStartTime = DateTime.UtcNow;
//                long processedWords = 0;
//                stats.DiscriminatorLosses.Clear();
                
//                for (int iteration = 0; iteration < parameters.EpochSize; iteration += parameters.BatchSize)
//                {
//                    // Обучение дискриминатора
//                    for (int disStep = 0; disStep < parameters.DisSteps; disStep++)
//                    {
//                        trainer.DiscriminatorStep(stats);
//                    }
                    
//                    // Обучение маппинга (обман дискриминатора)
//                    processedWords += trainer.MappingStep(stats);
                    
//                    // Логирование прогресса каждые 500 итераций
//                    if (iteration % 500 == 0)
//                    {
//                        var elapsedTime = DateTime.UtcNow - epochStartTime;
//                        var avgDiscLoss = stats.DiscriminatorLosses.Count > 0 ? 
//                            stats.DiscriminatorLosses.Average() : 0.0;
//                        var rate = processedWords / elapsedTime.TotalSeconds;
                        
//                        logger.LogInformation($"{iteration:000000} - Потери дискриминатора: {avgDiscLoss:F4} - " +
//                                            $"{rate:F0} образцов/с");
                        
//                        // Сброс статистик
//                        epochStartTime = DateTime.UtcNow;
//                        processedWords = 0;
//                        stats.DiscriminatorLosses.Clear();
//                    }
//                }
                
//                // TODO: Добавить оценку и сохранение лучшей модели
//                // evaluator.all_eval(to_log);
//                // trainer.save_best(to_log, VALIDATION_METRIC);
                
//                logger.LogInformation($"Конец эпохи {epoch}");
                
//                // Обновление learning rate
//                // trainer.UpdateLearningRate(validationMetric, ValidationMetric, 
//                //     parameters.LrDecay, parameters.LrShrink, parameters.MinLr);
                
//                // Проверка минимального learning rate
//                // if (currentLr < parameters.MinLr)
//                // {
//                //     logger.LogInformation("Learning rate < 1e-6. Прерывание обучения.");
//                //     break;
//                // }
//            }
            
//            logger.LogInformation("Состязательное обучение завершено");
//        }

//        /// <summary>
//        /// Запускает Procrustes refinement
//        /// </summary>
//        private static async Task RunProcrustesRefinementAsync(CrossLingualTrainer trainer,
//            UnsupervisedParameters parameters, ILogger logger)
//        {
//            logger.LogSeparator("ИТЕРАТИВНОЕ PROCRUSTES REFINEMENT");
            
//            // Загружаем лучшую модель
//            await trainer.ReloadBestModelAsync();
            
//            for (int iteration = 0; iteration < parameters.NRefinement; iteration++)
//            {
//                logger.LogInformation($"Начало итерации refinement {iteration}...");
                
//                // TODO: Построение словаря из выровненных эмбеддингов
//                // trainer.BuildDictionary();
                
//                // Применение решения Прокруста
//                trainer.ApplyProcrustesAlignment();
                
//                // TODO: Оценка эмбеддингов
//                // evaluator.all_eval(to_log);
//                // trainer.save_best(to_log, VALIDATION_METRIC);
                
//                logger.LogInformation($"Конец итерации refinement {iteration}");
//            }
//        }

//        /// <summary>
//        /// Экспортирует финальные эмбеддинги
//        /// </summary>
//        private static async Task ExportFinalEmbeddingsAsync(CrossLingualTrainer trainer,
//            UnsupervisedParameters parameters, ILogger logger)
//        {
//            logger.LogInformation("Экспорт финальных эмбеддингов...");
            
//            // Загружаем лучшую модель
//            await trainer.ReloadBestModelAsync();
            
//            // TODO: Реализовать экспорт эмбеддингов
//            // trainer.Export();
            
//            logger.LogInformation("Экспорт завершен");
//        }

//        #endregion
//    }
//}