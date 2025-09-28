

//namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core
//{
//    /// <summary>
//    /// Главная программа для кросс-лингвального выравнивания эмбеддингов
//    /// Объединяет все режимы: evaluate, supervised, unsupervised
//    /// Полностью переписанная версия Python проекта на C# с использованием .NET 9
//    /// </summary>
//    public static class Program
//    {
//        #region Main Entry Point

//        /// <summary>
//        /// Главная точка входа в программу
//        /// </summary>
//        /// <param name="args">Аргументы командной строки</param>
//        /// <returns>Код возврата (0 - успех, 1 - ошибка)</returns>
//        public static async Task<int> Main(string[] args)
//        {
//            try
//            {
//                // Отображаем информацию о программе
//                DisplayProgramInfo();

//                // Создаем корневую команду с подкомандами
//                var rootCommand = CreateRootCommand();

//                // Выполняем команду
//                return await rootCommand.InvokeAsync(args);
//            }
//            catch (Exception ex)
//            {
//                // Обрабатываем необработанные исключения
//                Console.Error.WriteLine($"Критическая ошибка: {ex.Message}");
                
//                if (ex.InnerException != null)
//                {
//                    Console.Error.WriteLine($"Внутренняя ошибка: {ex.InnerException.Message}");
//                }

//#if DEBUG
//                // В режиме отладки показываем полный stack trace
//                Console.Error.WriteLine($"Stack Trace:\n{ex.StackTrace}");
//#endif

//                return 1;
//            }
//        }

//        #endregion

//        #region Command Setup

//        /// <summary>
//        /// Создает корневую команду с подкомандами
//        /// </summary>
//        /// <returns>Настроенная корневая команда</returns>
//        private static RootCommand CreateRootCommand()
//        {
//            var rootCommand = new RootCommand("Кросс-лингвальное выравнивание эмбеддингов слов")
//            {
//                Description = "Высокопроизводительная C#/.NET 9 реализация системы для выравнивания " +
//                             "эмбеддингов слов между различными языками с использованием состязательного " +
//                             "обучения и метода ортогонального Прокруста."
//            };

//            // Добавляем подкоманды для различных режимов работы
//            rootCommand.AddCommand(CreateUnsupervisedCommand());
//            rootCommand.AddCommand(CreateSupervisedCommand());
//            rootCommand.AddCommand(CreateEvaluateCommand());

//            return rootCommand;
//        }

//        /// <summary>
//        /// Создает команду для unsupervised обучения
//        /// </summary>
//        /// <returns>Команда для unsupervised обучения</returns>
//        private static Command CreateUnsupervisedCommand()
//        {
//            var command = new Command("unsupervised", "Обучение без учителя с использованием состязательных сетей")
//            {
//                // Основные параметры
//                new Option<int>("--seed", () => -1, "Seed для воспроизводимости результатов (-1 для случайного)"),
//                new Option<int>("--verbose", () => 2, "Уровень детализации логов (0: warning, 1: info, 2: debug)"),
//                new Option<string>("--exp-path", () => "./experiments", "Путь для сохранения результатов эксперимента"),
//                new Option<string>("--exp-name", () => "unsupervised", "Название эксперимента"),
//                new Option<string>("--exp-id", () => "", "Уникальный ID эксперимента (автоматический если пустой)"),
//                new Option<bool>("--cuda", () => true, "Использовать GPU для ускорения вычислений"),
//                new Option<string>("--export", () => "txt", "Формат экспорта эмбеддингов (txt/pth)"),

//                // Параметры данных
//                new Option<string>("--src-lang", "Код исходного языка (например, 'en')") { IsRequired = true },
//                new Option<string>("--tgt-lang", "Код целевого языка (например, 'es')") { IsRequired = true },
//                new Option<string>("--src-emb", "Путь к файлу исходных эмбеддингов") { IsRequired = true },
//                new Option<string>("--tgt-emb", "Путь к файлу целевых эмбеддингов") { IsRequired = true },
//                new Option<int>("--emb-dim", () => 300, "Размерность эмбеддингов"),
//                new Option<int>("--max-vocab", () => 200000, "Максимальный размер словаря (-1 для неограниченного)"),
//                new Option<string>("--normalize-embeddings", () => "", "Типы нормализации (center,renorm)"),

//                // Параметры состязательного обучения
//                new Option<int>("--n-epochs", () => 5, "Количество эпох состязательного обучения"),
//                new Option<int>("--epoch-size", () => 1000000, "Количество итераций в эпохе"),
//                new Option<int>("--batch-size", () => 32, "Размер батча для обучения"),
                
//                // Параметры дискриминатора
//                new Option<int>("--dis-layers", () => 2, "Количество скрытых слоев дискриминатора"),
//                new Option<int>("--dis-hid-dim", () => 2048, "Размерность скрытых слоев дискриминатора"),
//                new Option<double>("--dis-dropout", () => 0.0, "Dropout для скрытых слоев дискриминатора"),
//                new Option<double>("--dis-input-dropout", () => 0.1, "Input dropout дискриминатора"),
//                new Option<int>("--dis-steps", () => 5, "Количество шагов обучения дискриминатора за итерацию"),
//                new Option<double>("--dis-lambda", () => 1.0, "Весовой коэффициент потерь дискриминатора"),
//                new Option<double>("--dis-smooth", () => 0.1, "Параметр сглаживания меток дискриминатора"),
//                new Option<double>("--dis-clip-weights", () => 0.0, "Обрезание градиентов дискриминатора (0 для отключения)"),
//                new Option<int>("--dis-most-frequent", () => 75000, "Количество наиболее частых слов для дискриминации"),

//                // Параметры оптимизации
//                new Option<string>("--map-optimizer", () => "sgd,lr=0.1", "Оптимизатор для матрицы преобразования"),
//                new Option<string>("--dis-optimizer", () => "sgd,lr=0.1", "Оптимизатор для дискриминатора"),
//                new Option<double>("--lr-decay", () => 0.98, "Коэффициент уменьшения learning rate"),
//                new Option<double>("--min-lr", () => 1e-6, "Минимальный learning rate"),
//                new Option<double>("--lr-shrink", () => 0.5, "Коэффициент сжатия LR при ухудшении метрики"),

//                // Параметры Procrustes refinement
//                new Option<int>("--n-refinement", () => 5, "Количество итераций Procrustes refinement"),
//                new Option<string>("--dico-method", () => "csls_knn_10", "Метод построения словаря (nn/csls_knn_10)"),
//                new Option<string>("--dico-build", () => "SourceToTarget", "Стратегия построения словаря (SourceToTarget/TargetToSource/SourceToTarget|TargetToSource/SourceToTarget&TargetToSource)"),
//                new Option<int>("--dico-max-rank", () => 15000, "Максимальный ранг слов в словаре"),
//                new Option<int>("--dico-max-size", () => 0, "Максимальный размер словаря (0 для неограниченного)"),
//                new Option<double>("--dico-threshold", () => 0.0, "Порог уверенности для словаря"),

//                // Параметры маппинга
//                new Option<bool>("--map-id-init", () => true, "Инициализировать преобразование как единичную матрицу"),
//                new Option<double>("--map-beta", () => 0.001, "Параметр beta для ортогонализации")
//            };

//            command.SetHandler(async (context) =>
//            {
//                try
//                {
//                    // Вызываем unsupervised программу с аргументами
//                    var args = Environment.GetCommandLineArgs().Skip(2).ToArray(); // Пропускаем имя программы и "unsupervised"
//                    return await UnsupervisedProgram.Main(args);
//                }
//                catch (Exception ex)
//                {
//                    Console.Error.WriteLine($"Ошибка в unsupervised обучении: {ex.Message}");
//                    return 1;
//                }
//            });

//            return command;
//        }

//        /// <summary>
//        /// Создает команду для supervised обучения
//        /// </summary>
//        /// <returns>Команда для supervised обучения</returns>
//        private static Command CreateSupervisedCommand()
//        {
//            var command = new Command("supervised", "Обучение с учителем с использованием готового словаря")
//            {
//                // Основные параметры
//                new Option<int>("--seed", () => -1, "Seed для воспроизводимости результатов"),
//                new Option<int>("--verbose", () => 2, "Уровень детализации логов"),
//                new Option<string>("--exp-path", () => "./experiments", "Путь для сохранения результатов"),
//                new Option<string>("--exp-name", () => "supervised", "Название эксперимента"),
//                new Option<bool>("--cuda", () => true, "Использовать GPU"),

//                // Параметры данных
//                new Option<string>("--src-lang", "Код исходного языка") { IsRequired = true },
//                new Option<string>("--tgt-lang", "Код целевого языка") { IsRequired = true },
//                new Option<string>("--src-emb", "Путь к исходным эмбеддингам") { IsRequired = true },
//                new Option<string>("--tgt-emb", "Путь к целевым эмбеддингам") { IsRequired = true },
//                new Option<int>("--emb-dim", () => 300, "Размерность эмбеддингов"),
//                new Option<int>("--max-vocab", () => 200000, "Максимальный размер словаря"),

//                // Параметры обучения
//                new Option<string>("--dico-train", () => "default", "Путь к обучающему словарю или 'default'/'identical_char'"),
//                new Option<int>("--n-refinement", () => 5, "Количество итераций refinement"),
//                new Option<string>("--dico-method", () => "csls_knn_10", "Метод построения словаря"),
//                new Option<string>("--dico-build", () => "SourceToTarget&TargetToSource", "Стратегия построения словаря"),
//                new Option<int>("--dico-max-rank", () => 10000, "Максимальный ранг слов в словаре")
//            };

//            command.SetHandler(async () =>
//            {
//                Console.WriteLine("Supervised обучение будет реализовано в следующих версиях.");
//                Console.WriteLine("Используйте пока unsupervised режим.");
//                return 0;
//            });

//            return command;
//        }

//        /// <summary>
//        /// Создает команду для оценки
//        /// </summary>
//        /// <returns>Команда для оценки</returns>
//        private static Command CreateEvaluateCommand()
//        {
//            var command = new Command("evaluate", "Оценка качества выравненных эмбеддингов")
//            {
//                // Основные параметры
//                new Option<int>("--verbose", () => 2, "Уровень детализации логов"),
//                new Option<bool>("--cuda", () => true, "Использовать GPU"),

//                // Параметры данных
//                new Option<string>("--src-lang", "Код исходного языка") { IsRequired = true },
//                new Option<string>("--tgt-lang", "Код целевого языка") { IsRequired = true },
//                new Option<string>("--src-emb", "Путь к исходным эмбеддингам") { IsRequired = true },
//                new Option<string>("--tgt-emb", "Путь к целевым эмбеддингам") { IsRequired = true },
//                new Option<int>("--emb-dim", () => 300, "Размерность эмбеддингов"),
//                new Option<int>("--max-vocab", () => 200000, "Максимальный размер словаря"),

//                // Параметры оценки
//                new Option<string>("--dico-eval", () => "default", "Путь к словарю для оценки"),
//                new Option<bool>("--crosslingual", () => false, "Выполнять кросс-лингвальную оценку"),
//                new Option<string>("--normalize-embeddings", () => "", "Нормализация эмбеддингов")
//            };

//            command.SetHandler(async () =>
//            {
//                Console.WriteLine("Модуль оценки будет реализован в следующих версиях.");
//                Console.WriteLine("Включит в себя:");
//                Console.WriteLine("- Оценку семантического сходства слов");
//                Console.WriteLine("- Точность перевода слов");
//                Console.WriteLine("- Точность перевода предложений");
//                Console.WriteLine("- Кросс-лингвальные бенчмарки");
//                return 0;
//            });

//            return command;
//        }

//        #endregion

//        #region Helper Methods

//        /// <summary>
//        /// Отображает информацию о программе
//        /// </summary>
//        private static void DisplayProgramInfo()
//        {
//            var version = System.Reflection.Assembly.GetExecutingAssembly().GetName().Version;
            
//            Console.WriteLine("╔════════════════════════════════════════════════════════════════════════════════╗");
//            Console.WriteLine("║                   Кросс-лингвальное выравнивание эмбеддингов                  ║");
//            Console.WriteLine("║                              C# / .NET 9 версия                               ║");
//            Console.WriteLine($"║                                 v{version}                                ║");
//            Console.WriteLine("╠════════════════════════════════════════════════════════════════════════════════╣");
//            Console.WriteLine("║ Высокопроизводительная реализация алгоритмов:                                 ║");
//            Console.WriteLine("║ • Состязательное обучение для выравнивания эмбеддингов                        ║");
//            Console.WriteLine("║ • Ортогональное решение Прокруста                                              ║");
//            Console.WriteLine("║ • CSLS (Cross-domain Similarity Local Scaling)                                ║");
//            Console.WriteLine("║ • Множественные метрики оценки качества                                        ║");
//            Console.WriteLine("║                                                                                ║");
//            Console.WriteLine("║ Особенности реализации:                                                       ║");
//            Console.WriteLine("║ • System.Numerics.Tensors для высокой производительности                      ║");
//            Console.WriteLine("║ • TorchSharp для нейронных сетей                                               ║");
//            Console.WriteLine("║ • Параллельные вычисления                                                      ║");
//            Console.WriteLine("║ • Оптимизированные матричные операции                                          ║");
//            Console.WriteLine("║ • Подробное логирование и мониторинг                                           ║");
//            Console.WriteLine("╚════════════════════════════════════════════════════════════════════════════════╝");
//            Console.WriteLine();
//        }

//        #endregion
//    }
//}

//// Расширения для System.Linq
//namespace System.Linq
//{
//    /// <summary>
//    /// Дополнительные расширения LINQ
//    /// </summary>
//    public static class LinqExtensions
//    {
//        /// <summary>
//        /// Пропускает указанное количество элементов
//        /// </summary>
//        /// <typeparam name="T">Тип элементов</typeparam>
//        /// <param name="source">Источник данных</param>
//        /// <param name="count">Количество элементов для пропуска</param>
//        /// <returns>Отфильтрованная последовательность</returns>
//        public static IEnumerable<T> Skip<T>(this T[] source, int count)
//        {
//            return ((IEnumerable<T>)source).Skip(count);
//        }
//    }
//}