using Avalonia.Media;
using MathNet.Numerics;
using Microsoft.Extensions.Logging;
using Ssz.AI.Helpers;
using Ssz.Utils;
using Ssz.Utils.Logging;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;

namespace Ssz.AI.Models.AdvancedEmbeddingModel;

public class MappingData_V2 : ISerializableModelObject
{
    public const string FileName_PrimaryWords_RU_EN = "PrimaryWords_RU_EN.csv";

    private ILoggersSet _loggersSet;
    public LanguageInfo LanguageInfo_RU;
    public LanguageInfo LanguageInfo_EN;    

    public MatrixFloat Temp_ProxBits_RU = new();
    /// <summary>
    ///     Word for each bit.
    /// </summary>
    public Word[] Temp_BitWords_RU = null!;    

    public MatrixFloat Temp_ProxBits_EN = new();
    /// <summary>
    ///     Word for each bit.
    /// </summary>
    public Word[] Temp_BitWords_EN = null!;

    /// <summary>
    ///     Index RU -> Index EN
    /// </summary>
    public int[] Mapping_RU_EN = null!;

    public MappingData_V2(ILoggersSet loggersSet, LanguageInfo languageInfo_RU, LanguageInfo languageInfo_EN)
    {
        _loggersSet = loggersSet;
        LanguageInfo_RU = languageInfo_RU;
        LanguageInfo_EN = languageInfo_EN;        
    }

    public void GenerateOwnedData(int vectorLength)
    {
        Mapping_RU_EN = new int[Model01.Constants.DiscreteVectorLength];
    }

    public void Prepare()
    {
        Temp_ProxBits_RU = new MatrixFloat(Model01.Constants.DiscreteVectorLength, Model01.Constants.DiscreteVectorLength);
        Temp_BitWords_RU = new Word[Model01.Constants.DiscreteVectorLength];        
        foreach (var primaryWord in LanguageInfo_RU.Clusterization_AlgorithmData.PrimaryWords)
        {
            int prjectionIndex = LanguageInfo_RU.ProjectionOptimization_AlgorithmData.WordsProjectionIndices[primaryWord.Index];
            Temp_BitWords_RU[prjectionIndex] = primaryWord;
        }
        for (int i = 0; i < Model01.Constants.DiscreteVectorLength; i += 1)
        {
            for (int j = 0; j < Model01.Constants.DiscreteVectorLength; j += 1)
            {
                Temp_ProxBits_RU[i, j] = TensorPrimitives.CosineSimilarity(
                    LanguageInfo_RU.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_RU[i].Index],
                    LanguageInfo_RU.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_RU[j].Index]);
            }
        }

        Temp_ProxBits_EN = new MatrixFloat(Model01.Constants.DiscreteVectorLength, Model01.Constants.DiscreteVectorLength);
        Temp_BitWords_EN = new Word[Model01.Constants.DiscreteVectorLength];        
        foreach (var primaryWord in LanguageInfo_EN.Clusterization_AlgorithmData.PrimaryWords)
        {
            int prjectionIndex = LanguageInfo_EN.ProjectionOptimization_AlgorithmData.WordsProjectionIndices[primaryWord.Index];
            Temp_BitWords_EN[prjectionIndex] = primaryWord;
        }
        for (int i = 0; i < Model01.Constants.DiscreteVectorLength; i += 1)
        {
            for (int j = 0; j < Model01.Constants.DiscreteVectorLength; j += 1)
            {
                Temp_ProxBits_EN[i, j] = TensorPrimitives.CosineSimilarity(
                    LanguageInfo_EN.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_EN[i].Index],
                    LanguageInfo_EN.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_EN[j].Index]);
            }
        }
    }

    public void Prepare2()
    {
        var fileData = CsvHelper.LoadCsvFile(Path.Combine(@"Data", FileName_PrimaryWords_RU_EN), includeFiles: false);

        List<(Word, Word)> primaryWords_RU_EN = new(300);
        foreach (var line in fileData.Values)
        {
            string word_RU_String = line[0] ?? @"";
            string word_EN_String = (line[1] ?? @"").ToLowerInvariant().Trim();
            var word_RU = LanguageInfo_RU.Words.FirstOrDefault(w => w.Name == word_RU_String);
            var word_EN = LanguageInfo_EN.Words.FirstOrDefault(w => w.Name == word_EN_String);
            if (word_RU is not null && word_EN is not null)
            {
                primaryWords_RU_EN.Add((word_RU, word_EN));
            }
        }

        var d = primaryWords_RU_EN.Count;
        Temp_ProxBits_RU = new MatrixFloat(d, d);
        Temp_BitWords_RU = new Word[d];
        foreach (var i in Enumerable.Range(0, d))
        {      
            var it = primaryWords_RU_EN[i];
            Temp_BitWords_RU[i] = it.Item1;
        }
        for (int i = 0; i < d; i += 1)
        {
            for (int j = 0; j < d; j += 1)
            {
                Temp_ProxBits_RU[i, j] = TensorPrimitives.CosineSimilarity(
                    LanguageInfo_RU.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_RU[i].Index],
                    LanguageInfo_RU.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_RU[j].Index]);
            }
        }

        Temp_ProxBits_EN = new MatrixFloat(d, d);
        Temp_BitWords_EN = new Word[d];
        foreach (var i in Enumerable.Range(0, d))
        {
            var it = primaryWords_RU_EN[i];
            Temp_BitWords_EN[i] = it.Item2;
        }
        for (int i = 0; i < d; i += 1)
        {
            for (int j = 0; j < d; j += 1)
            {
                Temp_ProxBits_EN[i, j] = TensorPrimitives.CosineSimilarity(
                    LanguageInfo_EN.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_EN[i].Index],
                    LanguageInfo_EN.DiscreteVectorsAndMatrices.DiscreteVectors[Temp_BitWords_EN[j].Index]);
            }
        }
    }

    public void Prepare3()
    {
        var fileData = CsvHelper.LoadCsvFile(Path.Combine(@"Data", FileName_PrimaryWords_RU_EN), includeFiles: false);

        List<(Word, Word)> primaryWords_RU_EN = new(300);
        HashSet<Word> primaryWords_EN = new(300);
        foreach (var line in fileData.Values)
        {
            string word_RU_String = line[0] ?? @"";
            string word_EN_String = (line[1] ?? @"").ToLowerInvariant().Trim();
            var word_RU = LanguageInfo_RU.Words.FirstOrDefault(w => w.Name == word_RU_String);
            var word_EN = LanguageInfo_EN.Words.FirstOrDefault(w => w.Name == word_EN_String);
            if (word_RU is not null && word_EN is not null)
            {
                primaryWords_RU_EN.Add((word_RU, word_EN));
                if (primaryWords_EN.TryGetValue(word_EN, out var word_EN2))
                {
                    _loggersSet.UserFriendlyLogger.LogInformation($"Duplicate EN. word_EN = {word_EN.Name}");
                    throw new Exception();
                }
                primaryWords_EN.Add(word_EN);
            }
        }

        var r = new Random();

        var d = primaryWords_RU_EN.Count;
        
        Temp_BitWords_RU = WordsHelper.GetRandomOrderWords(primaryWords_RU_EN.Select(it => it.Item1).ToList(), r).ToArray();
        Temp_ProxBits_RU = new MatrixFloat(d, d);
        for (int i = 0; i < d; i += 1)
        {
            for (int j = 0; j < d; j += 1)
            {
                Temp_ProxBits_RU[i, j] = TensorPrimitives.Dot(
                    Temp_BitWords_RU[i].OldVectorNormalized,
                    Temp_BitWords_RU[j].OldVectorNormalized);
            }
        }
        
        Temp_BitWords_EN = WordsHelper.GetRandomOrderWords(primaryWords_RU_EN.Select(it => it.Item2).ToList(), r).ToArray();
        Temp_ProxBits_EN = new MatrixFloat(d, d);
        for (int i = 0; i < d; i += 1)
        {
            for (int j = 0; j < d; j += 1)
            {
                Temp_ProxBits_EN[i, j] = TensorPrimitives.Dot(
                    Temp_BitWords_EN[i].OldVectorNormalized,
                    Temp_BitWords_EN[j].OldVectorNormalized);
            }
        }

        _loggersSet.UserFriendlyLogger.LogInformation($"Prepare3() Done. D = {d}");
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {
            writer.WriteArray(Mapping_RU_EN);
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    Mapping_RU_EN = reader.ReadArray<int>()!;
                    break;
            }
        }
    }

    public void CalculatePrimaryWordsMapping()
    {
        var stopwatch = Stopwatch.StartNew();

        // Строим матрицу стоимости сопоставления
        var costMatrix = BuildStructureCostMatrix(Temp_ProxBits_RU, Temp_ProxBits_EN);

        // Ищем взаимно однозначное соответствие с минимальной структурной ошибкой
        Mapping_RU_EN = HungarianAlgorithm(costMatrix);

        stopwatch.Stop();
        _loggersSet.UserFriendlyLogger.LogInformation("CalculateMapping totally done. Elapsed Milliseconds = " + stopwatch.ElapsedMilliseconds);
    }

    /// <summary>
    /// Получить матрицу стоимости сопоставления по разнице попарных косинусных расстояний.
    /// </summary>
    /// <param name="distA">Матрица расстояний для первого множества.</param>
    /// <param name="distB">Матрица расстояний для второго множества.</param>
    /// <returns>Матрица стоимости размера N x N, где элемент (i,j) — "разница структур" при сопоставлении i и j.</returns>
    private static MatrixFloat BuildStructureCostMatrix(MatrixFloat distA, MatrixFloat distB)
    {
        int N = distA.Dimensions[0];
        var costMatrix = new MatrixFloat(new int[] { N, N });

        // Для каждого возможного соответствия рассчитываем сумму разниц по всем остальным парам
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float sum = 0f;
                for (int k = 0; k < N; k++)
                {
                    for (int l = 0; l < N; l++)
                    {
                        // Сравниваем структуру: расстояние между i, k в A и между j, l в B
                        sum += MathF.Abs(distA[i, k] - distB[j, l]);
                    }
                }
                costMatrix[i, j] = sum;
            }
        }
        return costMatrix;
    }

    /// <summary>
    /// Реализация Венгерского алгоритма (Munkres) для оптимального сопоставления.
    /// Работает с квадратной матрицей стоимости cost размером N x N (MatrixFloat).
    /// Возвращает массив correspondences: для i из первого множества индекс во втором.
    /// </summary>
    private int[] HungarianAlgorithm(MatrixFloat cost)
    {
        int N = cost.Dimensions[0]; // предполагается квадратная матрица
        float[,] matrix = new float[N, N];

        // 1. Копирование из MatrixFloat в float[,] для удобства манипулирования
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                matrix[i, j] = cost[i, j];

        // 2. Поэтапное улучшение исходной матрицы через вычитание минимумов
        // 2.1. В каждой строке вычитаем минимальный элемент
        for (int i = 0; i < N; ++i)
        {
            float minVal = matrix[i, 0];
            for (int j = 1; j < N; ++j)
                if (matrix[i, j] < minVal) minVal = matrix[i, j];
            for (int j = 0; j < N; ++j)
                matrix[i, j] -= minVal;
        }

        // 2.2. В каждом столбце вычитаем минимальный элемент
        for (int j = 0; j < N; ++j)
        {
            float minVal = matrix[0, j];
            for (int i = 1; i < N; ++i)
                if (matrix[i, j] < minVal) minVal = matrix[i, j];
            for (int i = 0; i < N; ++i)
                matrix[i, j] -= minVal;
        }

        // 3. Маркируем строки/столбцы, где можно сделать назначение по нулевому элементу (используем вспомогательные массивы)
        bool[] starredRows = new bool[N];        // строки, для которых звёздочка поставлена
        bool[] starredCols = new bool[N];        // столбцы, занятые звёздочкой
        int[] starsByRow = new int[N];           // в каждой строке: номер столбца с назначением, -1 если нет
        for (int i = 0; i < N; ++i) starsByRow[i] = -1;

        // 3.1. Назначаем задачи по первым найденным нулям так, чтобы каждая строка и столбец имела не более одной звёздочки
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                if (matrix[i, j] == 0f && !starredRows[i] && !starredCols[j])
                {
                    starsByRow[i] = j;
                    starredRows[i] = true;
                    starredCols[j] = true;
                    break;
                }
            }
        }

        // 3.2. Очищаем отметки для повторного использования
        Array.Fill(starredRows, false);
        Array.Fill(starredCols, false);

        // 4. Основной цикл — пока количество назначений < N (т.е. нет уникального сопоставления)
        int iterationN = 0;
        while (true)
        {   
            if (iterationN % 100 == 0)
                _loggersSet.UserFriendlyLogger.LogInformation($"HungarianAlgorithm() Iteration started. N = {iterationN}");
            iterationN += 1;

            // Для каждой строки, помечаем столбцы, занятые назначением
            Array.Fill(starredCols, false);
            int starsCount = 0;
            for (int i = 0; i < N; ++i)
            {
                if (starsByRow[i] != -1)
                {
                    starredCols[starsByRow[i]] = true;
                    starsCount++;
                }
            }

            // Если назначили N задач — готово!
            if (starsCount == N)
                break;

            // 5. Находим строки без назначений
            bool[] coveredRows = new bool[N];
            bool[] coveredCols = new bool[N];
            for (int i = 0; i < N; i++)
                coveredRows[i] = (starsByRow[i] == -1);

            // 6. Пока есть открытые строки, ищем не покрытый ноль и строим путь чередования
            int[] primeByRow = new int[N];
            Array.Fill(primeByRow, -1);

            bool foundPath = false;
            while (true)
            {
                foundPath = false;
                for (int i = 0; i < N; ++i)
                {
                    if (!coveredRows[i])
                        continue;
                    for (int j = 0; j < N; ++j)
                    {
                        if (!coveredCols[j] && matrix[i, j] == 0f)
                        {
                            primeByRow[i] = j;

                            // Если в столбце нет звёздочки — строим путь чередования
                            int starredRow = -1;
                            for (int k = 0; k < N; ++k)
                            {
                                if (starsByRow[k] == j)
                                {
                                    starredRow = k;
                                    break;
                                }
                            }
                            if (starredRow == -1)
                            {
                                // Формируем путь чередования и меняем назначения
                                int row = i;
                                while (true)
                                {
                                    int col = primeByRow[row];
                                    int prevRow = -1;
                                    for (int k = 0; k < N; ++k)
                                    {
                                        if (starsByRow[k] == col)
                                        {
                                            prevRow = k;
                                            break;
                                        }
                                    }
                                    starsByRow[row] = col;
                                    if (prevRow == -1)
                                        break;
                                    row = prevRow;
                                }
                                // После построения пути, обновляем покрытие и прерываем поиск
                                Array.Fill(coveredRows, false);
                                Array.Fill(coveredCols, false);
                                Array.Fill(primeByRow, -1);
                                foundPath = true;
                                break;
                            }
                            else
                            {
                                // Покрываем найденный столбец и открываем строку с назначением
                                coveredCols[j] = true;
                                coveredRows[starredRow] = true;
                            }
                        }
                    }
                    if (foundPath)
                        break;
                }
                if (!foundPath)
                {
                    // Если не найден ноль, уменьшаем покрытие по минимальному не покрытому элементу
                    float minUncovered = float.MaxValue;
                    for (int i = 0; i < N; ++i)
                    {
                        if (!coveredRows[i])
                            continue;
                        for (int j = 0; j < N; ++j)
                        {
                            if (!coveredCols[j] && matrix[i, j] < minUncovered)
                                minUncovered = matrix[i, j];
                        }
                    }
                    for (int i = 0; i < N; ++i)
                        for (int j = 0; j < N; ++j)
                        {
                            if (coveredRows[i] && !coveredCols[j]) matrix[i, j] -= minUncovered;
                            else if (!coveredRows[i] && coveredCols[j]) matrix[i, j] += minUncovered;
                        }
                }
                else
                    break;
            }
        }        

        // 7. Формируем ответ: для каждой строки номер назначенного столбца
        return starsByRow;
    }

    public void DisplayWords()
    {
        List<string?[]> fileData = new List<string?[]>();

        for (int primaryWordIndex_RU = 0; primaryWordIndex_RU < Temp_BitWords_RU.Length; primaryWordIndex_RU += 1)
        {
            var word_RU = Temp_BitWords_RU[primaryWordIndex_RU].Name;            
            int primaryWordIndex_EN = Mapping_RU_EN[primaryWordIndex_RU];
            string word_EN = Temp_BitWords_EN[primaryWordIndex_EN].Name;
            _loggersSet.UserFriendlyLogger.LogInformation($"{word_RU} -> {word_EN}");
        }

        _loggersSet.UserFriendlyLogger.LogInformation($"DisplayWords() Done. D = {Temp_BitWords_RU.Length}");
    }

    public async void TranslatePrimaryWords()
    {
        List<string?[]> fileData = new List<string?[]>();

        for (int primaryWordIndex_RU = 0; primaryWordIndex_RU < Model01.Constants.DiscreteVectorLength; primaryWordIndex_RU += 1)
        {
            var word_RU = Temp_BitWords_RU[primaryWordIndex_RU].Name;

            var parts = word_RU.Split('_');
            if (parts.Length != 2)
                throw new InvalidOperationException();

            string word = parts[0];
            string pos = parts[1];

            // Получение перевода
            string word_EN = await TranslationHelper.TranslateWordAsync(word, _loggersSet);

            fileData.Add([word_RU, word_EN]);
            //fileData.Add([ word_RU.Substring(0, word_RU.IndexOf("_")) ]);
            //int primaryWordIndex_EN = Mapping_RU_EN[primaryWordIndex_RU];
            _loggersSet.UserFriendlyLogger.LogInformation($"{word_RU} -> {word_EN}");
        }

        CsvHelper.SaveCsvFile(Path.Combine(@"Data", FileName_PrimaryWords_RU_EN), fileData);
    }
}
