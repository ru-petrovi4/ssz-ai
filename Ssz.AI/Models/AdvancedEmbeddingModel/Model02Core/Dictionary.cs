using System;
using System.Collections.Generic;
using System.Linq;

namespace Ssz.AI.Models.AdvancedEmbeddingModel.Model02Core
{
    /// <summary>
    /// Класс для работы со словарем слов и их индексов
    /// Аналог Dictionary класса из Python проекта
    /// </summary>
    public sealed class Dictionary
    {
        #region Private Fields
        
        /// <summary>
        /// Отображение индекса на слово
        /// </summary>
        private readonly SortedDictionary<int, string> _idToWord;

        /// <summary>
        /// Отображение слова на индекс
        /// Все ключи Lower-Case
        /// </summary>
        private readonly Dictionary<string, int> _wordToId;
        
        /// <summary>
        /// Язык данного словаря
        /// </summary>
        private readonly string _language;
        
        #endregion

        #region Constructors

        /// <summary>
        /// Инициализирует новый экземпляр словаря
        /// </summary>
        /// <param name="idToWord">Отображение индекса на слово</param>
        /// <param name="wordToId">Отображение слова на индекс</param>
        /// <param name="language">Код языка (например, 'en', 'es')</param>
        /// <exception cref="ArgumentNullException">Если любой из параметров равен null</exception>
        /// <exception cref="ArgumentException">Если размеры словарей не совпадают</exception>
        public Dictionary(SortedDictionary<int, string> idToWord, Dictionary<string, int> wordToId, string language)
        {
            _idToWord = idToWord ?? throw new ArgumentNullException(nameof(idToWord));
            _wordToId = wordToId ?? throw new ArgumentNullException(nameof(wordToId));
            _language = language ?? throw new ArgumentNullException(nameof(language));
            
            // Проверяем валидность словарей
            CheckValid();
        }

        #endregion

        #region Public Properties

        /// <summary>
        /// Количество слов в словаре
        /// </summary>
        public int Count => _idToWord.Count;
        
        /// <summary>
        /// Код языка словаря
        /// </summary>
        public string Language => _language;
        
        /// <summary>
        /// Отображение слова на индекс (только для чтения)
        /// Все ключи Lower-Case
        /// </summary>
        public IReadOnlyDictionary<string, int> WordToId => _wordToId;

        /// <summary>
        /// Отображение индекса на слово (только для чтения)        
        /// </summary>
        public IReadOnlyDictionary<int, string> IdToWord => _idToWord;

        #endregion

        #region Indexers

        /// <summary>
        /// Получает слово по индексу
        /// </summary>
        /// <param name="index">Индекс слова</param>
        /// <returns>Слово по указанному индексу</returns>
        /// <exception cref="KeyNotFoundException">Если индекс не найден</exception>
        public string this[int index]
        {
            get
            {
                if (!_idToWord.TryGetValue(index, out string? word))
                {
                    throw new KeyNotFoundException($"Индекс {index} не найден в словаре");
                }
                return word;
            }
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Проверяет, содержится ли слово в словаре
        /// </summary>
        /// <param name="word">Слово для проверки</param>
        /// <returns>true, если слово содержится в словаре</returns>
        public bool Contains(string word)
        {
            return _wordToId.ContainsKey(word.ToLowerInvariant());
        }

        /// <summary>
        /// Получает индекс слова
        /// </summary>
        /// <param name="word">Слово</param>
        /// <returns>Индекс слова</returns>
        /// <exception cref="KeyNotFoundException">Если слово не найдено</exception>
        public int GetIndex(string word)
        {
            if (!_wordToId.TryGetValue(word.ToLowerInvariant(), out int index))
            {
                throw new KeyNotFoundException($"Слово '{word}' не найдено в словаре");
            }
            return index;
        }

        /// <summary>
        /// Пытается получить индекс слова
        /// </summary>
        /// <param name="word">Слово</param>
        /// <param name="index">Индекс слова (если найден)</param>
        /// <returns>true, если слово найдено</returns>
        public bool TryGetIndex(string word, out int index)
        {
            return _wordToId.TryGetValue(word.ToLowerInvariant(), out index);
        }

        /// <summary>
        /// Ограничивает размер словаря максимальным количеством слов
        /// Сохраняет только самые частые слова (с меньшими индексами)
        /// </summary>
        /// <param name="maxVocab">Максимальный размер словаря</param>
        /// <exception cref="ArgumentException">Если maxVocab меньше 1</exception>
        public void Prune(int maxVocab)
        {
            if (maxVocab < 1)
                throw new ArgumentException("Максимальный размер словаря должен быть больше 0", nameof(maxVocab));

            // Если словарь уже меньше или равен максимальному размеру, ничего не делаем
            if (_idToWord.Count <= maxVocab)
                return;

            // Создаем новые словари с ограниченным размером
            var keysToRemove = _idToWord.Keys.Where(k => k >= maxVocab).ToList();
            
            // Удаляем лишние записи из id2word
            foreach (var key in keysToRemove)
            {
                if (_idToWord.TryGetValue(key, out string? word))
                {
                    _idToWord.Remove(key);
                    _wordToId.Remove(word.ToLowerInvariant());
                }
            }

            // Проверяем валидность после обрезания
            CheckValid();
        }

        /// <summary>
        /// Проверяет равенство с другим словарем
        /// </summary>
        /// <param name="other">Другой словарь для сравнения</param>
        /// <returns>true, если словари равны</returns>
        public bool Equals(Dictionary? other)
        {
            if (other == null) return false;
            if (ReferenceEquals(this, other)) return true;
            
            CheckValid();
            other.CheckValid();

            if (_idToWord.Count != other._idToWord.Count)
                return false;

            if (_language != other._language)
                return false;

            // Проверяем все пары ключ-значение
            return _idToWord.All(kvp => 
                other._idToWord.TryGetValue(kvp.Key, out string? otherWord) && 
                kvp.Value == otherWord);
        }

        /// <summary>
        /// Создает глубокую копию словаря
        /// </summary>
        /// <returns>Новый экземпляр словаря с теми же данными</returns>
        public Dictionary Clone()
        {
            var newId2Word = new SortedDictionary<int, string>(_idToWord);
            var newWord2Id = new Dictionary<string, int>(_wordToId);
            return new Dictionary(newId2Word, newWord2Id, _language);
        }

        /// <summary>
        /// Возвращает строковое представление словаря
        /// </summary>
        /// <returns>Строковое представление</returns>
        public override string ToString()
        {
            return $"Dictionary(language={_language}, count={_idToWord.Count})";
        }

        /// <summary>
        /// Получает все слова в словаре в порядке их индексов
        /// </summary>
        /// <returns>Перечисление слов</returns>
        public IEnumerable<string> GetWords()
        {
            return _idToWord.Values;
        }

        /// <summary>
        /// Получает все индексы в словаре в порядке возрастания
        /// </summary>
        /// <returns>Перечисление индексов</returns>
        public IEnumerable<int> GetIndices()
        {
            return _idToWord.Keys;
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Проверяет валидность внутреннего состояния словаря
        /// </summary>
        /// <exception cref="InvalidOperationException">Если состояние невалидно</exception>
        private void CheckValid()
        {
            // Проверяем, что размеры совпадают
            if (_idToWord.Count != _wordToId.Count)
            {
                throw new InvalidOperationException(
                    $"Размеры словарей не совпадают: id2word={_idToWord.Count}, word2id={_wordToId.Count}");
            }

            // Проверяем соответствие между словарями (только для небольших словарей для производительности)
            if (_idToWord.Count < 10000)
            {
                foreach (var kvp in _idToWord)
                {
                    if (!_wordToId.TryGetValue(kvp.Value.ToLowerInvariant(), out int mappedIndex) || mappedIndex != kvp.Key)
                    {
                        throw new InvalidOperationException(
                            $"Несоответствие в словарях для индекса {kvp.Key} и слова '{kvp.Value}'");
                    }
                }
            }
        }

        #endregion
    }
}