using System;
using System.IO;
using System.Linq;
using Microsoft.Extensions.Logging;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core
{
    /// <summary>
    /// Кастомный форматтер для логов с отображением времени выполнения
    /// Аналог LogFormatter из Python проекта
    /// </summary>
    public sealed class LogFormatter
    {
        #region Private Fields
        
        /// <summary>
        /// Время начала работы программы
        /// </summary>
        private readonly DateTime _startTime;
        
        #endregion

        #region Constructor

        /// <summary>
        /// Инициализирует новый экземпляр форматтера логов
        /// </summary>
        public LogFormatter()
        {
            _startTime = DateTime.UtcNow;
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Форматирует сообщение лога с добавлением времени выполнения
        /// </summary>
        /// <param name="level">Уровень лога</param>
        /// <param name="message">Сообщение</param>
        /// <returns>Отформатированное сообщение</returns>
        public string Format(LogLevel level, string message)
        {
            var elapsedTime = DateTime.UtcNow - _startTime;
            var prefix = $"{level.ToString().ToUpper()} - {DateTime.Now:yyyy-MM-dd HH:mm:ss} - {elapsedTime:hh\\:mm\\:ss}";
            
            // Форматируем многострочные сообщения с отступами
            var formattedMessage = message.Replace("\n", $"\n{new string(' ', prefix.Length + 3)}");
            
            return $"{prefix} - {formattedMessage}";
        }

        /// <summary>
        /// Сбрасывает время начала работы
        /// </summary>
        public void ResetTime()
        {
            // Для Serilog мы не можем изменить время после создания,
            // но можем пересоздать логгер при необходимости
        }

        #endregion
    }

    /// <summary>
    /// Утилитарный класс для создания и настройки логгеров
    /// Аналог функции create_logger из Python проекта
    /// </summary>
    public static class LoggerExtensions
    {
        #region Public Methods        

        /// <summary>
        /// Логирует разделительную линию для улучшения читаемости
        /// </summary>
        /// <param name="logger">Логгер</param>
        /// <param name="message">Сообщение для разделителя</param>
        /// <param name="level">Уровень лога</param>
        public static void LogSeparator(this Microsoft.Extensions.Logging.ILogger logger, string message, LogLevel level = LogLevel.Information)
        {
            var separator = new string('=', Math.Max(50, message.Length + 4));
            var formattedMessage = $"{separator}\n{message.PadLeft((separator.Length + message.Length) / 2)}\n{separator}";
            
            logger.Log(level, formattedMessage);
        }

        /// <summary>
        /// Логирует параметры в удобочитаемом формате
        /// </summary>
        /// <param name="logger">Логгер</param>
        /// <param name="parameters">Словарь параметров</param>
        /// <param name="title">Заголовок для параметров</param>
        public static void LogParameters(this Microsoft.Extensions.Logging.ILogger logger, 
            System.Collections.Generic.Dictionary<string, object> parameters, 
            string title = "Параметры")
        {
            logger.LogSeparator(title);
            
            foreach (var param in parameters.OrderBy(p => p.Key))
            {
                logger.LogInformation($"{param.Key}: {param.Value}");
            }
            
            logger.LogSeparator($"Конец: {title}");
        }

        /// <summary>
        /// Логирует прогресс выполнения операции
        /// </summary>
        /// <param name="logger">Логгер</param>
        /// <param name="current">Текущее значение</param>
        /// <param name="total">Общее количество</param>
        /// <param name="operation">Название операции</param>
        public static void LogProgress(this Microsoft.Extensions.Logging.ILogger logger, 
            int current, int total, string operation)
        {
            var percentage = (double)current / total * 100;
            logger.LogInformation($"{operation}: {current}/{total} ({percentage:F1}%)");
        }

        /// <summary>
        /// Логирует статистики производительности
        /// </summary>
        /// <param name="logger">Логгер</param>
        /// <param name="elapsedTime">Затраченное время</param>
        /// <param name="itemsProcessed">Количество обработанных элементов</param>
        /// <param name="operation">Название операции</param>
        public static void LogPerformance(this Microsoft.Extensions.Logging.ILogger logger, 
            TimeSpan elapsedTime, int itemsProcessed, string operation)
        {
            var rate = itemsProcessed / elapsedTime.TotalSeconds;
            logger.LogInformation($"{operation} завершена: {itemsProcessed} элементов за {elapsedTime:hh\\:mm\\:ss\\.fff} ({rate:F1} элементов/сек)");
        }

        #endregion
    }
}