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
        private readonly SortedDictionary<int, string> _id2Word;
        
        /// <summary>
        /// Отображение слова на индекс
        /// </summary>
        private readonly Dictionary<string, int> _word2Id;
        
        /// <summary>
        /// Язык данного словаря
        /// </summary>
        private readonly string _language;
        
        #endregion

        #region Constructors

        /// <summary>
        /// Инициализирует новый экземпляр словаря
        /// </summary>
        /// <param name="id2Word">Отображение индекса на слово</param>
        /// <param name="word2Id">Отображение слова на индекс</param>
        /// <param name="language">Код языка (например, 'en', 'es')</param>
        /// <exception cref="ArgumentNullException">Если любой из параметров равен null</exception>
        /// <exception cref="ArgumentException">Если размеры словарей не совпадают</exception>
        public Dictionary(SortedDictionary<int, string> id2Word, Dictionary<string, int> word2Id, string language)
        {
            _id2Word = id2Word ?? throw new ArgumentNullException(nameof(id2Word));
            _word2Id = word2Id ?? throw new ArgumentNullException(nameof(word2Id));
            _language = language ?? throw new ArgumentNullException(nameof(language));
            
            // Проверяем валидность словарей
            CheckValid();
        }

        #endregion

        #region Public Properties

        /// <summary>
        /// Количество слов в словаре
        /// </summary>
        public int Count => _id2Word.Count;
        
        /// <summary>
        /// Код языка словаря
        /// </summary>
        public string Language => _language;
        
        /// <summary>
        /// Отображение слова на индекс (только для чтения)
        /// </summary>
        public IReadOnlyDictionary<string, int> Word2Id => _word2Id;
        
        /// <summary>
        /// Отображение индекса на слово (только для чтения)
        /// </summary>
        public IReadOnlyDictionary<int, string> Id2Word => _id2Word;

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
                if (!_id2Word.TryGetValue(index, out string? word))
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
            return _word2Id.ContainsKey(word);
        }

        /// <summary>
        /// Получает индекс слова
        /// </summary>
        /// <param name="word">Слово</param>
        /// <returns>Индекс слова</returns>
        /// <exception cref="KeyNotFoundException">Если слово не найдено</exception>
        public int GetIndex(string word)
        {
            if (!_word2Id.TryGetValue(word, out int index))
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
            return _word2Id.TryGetValue(word, out index);
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
            if (_id2Word.Count <= maxVocab)
                return;

            // Создаем новые словари с ограниченным размером
            var keysToRemove = _id2Word.Keys.Where(k => k >= maxVocab).ToList();
            
            // Удаляем лишние записи из id2word
            foreach (var key in keysToRemove)
            {
                if (_id2Word.TryGetValue(key, out string? word))
                {
                    _id2Word.Remove(key);
                    _word2Id.Remove(word);
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

            if (_id2Word.Count != other._id2Word.Count)
                return false;

            if (_language != other._language)
                return false;

            // Проверяем все пары ключ-значение
            return _id2Word.All(kvp => 
                other._id2Word.TryGetValue(kvp.Key, out string? otherWord) && 
                kvp.Value == otherWord);
        }

        /// <summary>
        /// Создает глубокую копию словаря
        /// </summary>
        /// <returns>Новый экземпляр словаря с теми же данными</returns>
        public Dictionary Clone()
        {
            var newId2Word = new SortedDictionary<int, string>(_id2Word);
            var newWord2Id = new Dictionary<string, int>(_word2Id);
            return new Dictionary(newId2Word, newWord2Id, _language);
        }

        /// <summary>
        /// Возвращает строковое представление словаря
        /// </summary>
        /// <returns>Строковое представление</returns>
        public override string ToString()
        {
            return $"Dictionary(language={_language}, count={_id2Word.Count})";
        }

        /// <summary>
        /// Получает все слова в словаре в порядке их индексов
        /// </summary>
        /// <returns>Перечисление слов</returns>
        public IEnumerable<string> GetWords()
        {
            return _id2Word.Values;
        }

        /// <summary>
        /// Получает все индексы в словаре в порядке возрастания
        /// </summary>
        /// <returns>Перечисление индексов</returns>
        public IEnumerable<int> GetIndices()
        {
            return _id2Word.Keys;
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
            if (_id2Word.Count != _word2Id.Count)
            {
                throw new InvalidOperationException(
                    $"Размеры словарей не совпадают: id2word={_id2Word.Count}, word2id={_word2Id.Count}");
            }

            // Проверяем соответствие между словарями (только для небольших словарей для производительности)
            if (_id2Word.Count < 10000)
            {
                foreach (var kvp in _id2Word)
                {
                    if (!_word2Id.TryGetValue(kvp.Value, out int mappedIndex) || mappedIndex != kvp.Key)
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