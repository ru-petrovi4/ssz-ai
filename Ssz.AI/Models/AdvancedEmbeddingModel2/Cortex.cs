using Microsoft.Extensions.Logging;
using Ssz.AI.Helpers;
using Ssz.AI.Models.AdvancedEmbeddingModel2.EmbeddingsEvaluation;
using Ssz.Utils;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;

namespace Ssz.AI.Models.AdvancedEmbeddingModel2;

public partial class Cortex : ISerializableModelObject
{
    /// <summary>
    ///     Если задано SubAreaMiniColumnsCount, то генерируется только подмножество миниколонок с центром SubAreaCenter_Cx, SubAreaCenter_Cy и количеством SubAreaMiniColumnsCount
    /// </summary>
    /// <param name="constants"></param>        
    public Cortex(
        Model01.ModelConstants constants)
    {
        Constants = constants;        
    }

    #region public functions

    public readonly Model01.ModelConstants Constants;

    public List<Word> Words { get; private set; } = null!;

    public DenseMatrix<MiniColumn> MiniColumns { get; private set; } = null!;

    public readonly ActivitiyMaxInfo Temp_ActivitiyMaxInfo = new();

    public string Temp_InputCurrentDesc = null!;

    public void GenerateOwnedData(List<Word> words)
    {
        Words = words;
        MiniColumns = new DenseMatrix<MiniColumn>(Constants.CortexWidth_MiniColumns, Constants.CortexHeight_MiniColumns);

        foreach (int mcy in Enumerable.Range(0, MiniColumns.Dimensions[1]))
            foreach (int mcx in Enumerable.Range(0, MiniColumns.Dimensions[0]))
            {
                MiniColumn miniColumn = new MiniColumn(
                    Constants,
                    mcx,
                    mcy);

                miniColumn.GenerateOwnedData();

                MiniColumns[mcx, mcy] = miniColumn;
            }
    }

    public void Prepare()
    {
        // Находим ближайшие миниколонки для каждой миниколонки
        Parallel.For(
            fromInclusive: 0,
            toExclusive: MiniColumns.Data.Length,
            mci =>
            {
                MiniColumn mc = MiniColumns.Data[mci];
                mc.Prepare();
                mc.Temp_K_ForNearestMiniColumns.Add((Constants.PositiveK[0], Constants.NegativeK[0], mc));

                for (int mcy = mc.MCY - (int)Constants.SuperActivityRadius_MiniColumns - 1; mcy <= mc.MCY + (int)Constants.SuperActivityRadius_MiniColumns + 1; mcy += 1)
                    for (int mcx = mc.MCX - (int)Constants.SuperActivityRadius_MiniColumns - 1; mcx <= mc.MCX + (int)Constants.SuperActivityRadius_MiniColumns + 1; mcx += 1)
                    {
                        if (mcx < 0 ||
                                mcx >= MiniColumns.Dimensions[0] ||
                                mcy < 0 ||
                                mcy >= MiniColumns.Dimensions[1] ||
                                (mcx == mc.MCX && mcy == mc.MCY))
                            continue;

                        MiniColumn nearestMc = MiniColumns[mcx, mcy];
                        if (nearestMc is null)
                            continue;
                        float r = MathF.Sqrt((mcx - mc.MCX) * (mcx - mc.MCX) + (mcy - mc.MCY) * (mcy - mc.MCY));
                        if (r < Constants.SuperActivityRadius_MiniColumns + 0.00001f)
                        {
                            float k0 = MathHelper.GetInterpolatedValue(Constants.PositiveK, r);
                            float k1 = MathHelper.GetInterpolatedValue(Constants.NegativeK, r);
                            mc.Temp_K_ForNearestMiniColumns.Add((k0, k1, nearestMc));
                        }
                    }

                //for (int mcy = mc.MCY - (int)constants.HyperColumnSupposedRadius_ForMemorySaving_MiniColumns - 1; mcy <= mc.MCY + (int)constants.HyperColumnSupposedRadius_ForMemorySaving_MiniColumns + 1; mcy += 1)
                //    for (int mcx = mc.MCX - (int)constants.HyperColumnSupposedRadius_ForMemorySaving_MiniColumns - 1; mcx <= mc.MCX + (int)constants.HyperColumnSupposedRadius_ForMemorySaving_MiniColumns + 1; mcx += 1)
                //    {
                //        if (mcx < 0 ||
                //                mcx >= constants.CortexWidth_MiniColumns ||
                //                mcy < 0 ||
                //                mcy >= constants.CortexHeight_MiniColumns)
                //            continue;

                //        MiniColumn nearestMc = MiniColumns[mcx, mcy];
                //        if (nearestMc is null)
                //            continue;
                //        float r = MathF.Sqrt((mcx - mc.MCX) * (mcx - mc.MCX) + (mcy - mc.MCY) * (mcy - mc.MCY));
                //        if (r < constants.HyperColumnSupposedRadius_ForMemorySaving_MiniColumns + 0.00001f)
                //            mc.NearestMiniColumnsAndSelf_ForMemorySaving.Add(nearestMc);
                //    }
            });
    }

    /// <summary>
    ///     Returns true when finished.
    /// </summary>
    /// <param name="inputCorpusData"></param>
    /// <param name="cortexMemoriesCount"></param>
    /// <param name="random"></param>
    /// <returns></returns>
    public bool CalculateCortexMemories(InputCorpusData inputCorpusData, int cortexMemoriesCount, Random random)
    {
        for (int i = 0; i < cortexMemoriesCount; i += 1)
        {
            if (inputCorpusData.CurrentCortexMemoryIndex >= inputCorpusData.CortexMemories.Count - 1)
                return true;

            inputCorpusData.CurrentCortexMemoryIndex += 1;

            var cortexMemory = inputCorpusData.CortexMemories[inputCorpusData.CurrentCortexMemoryIndex];

            Temp_InputCurrentDesc = GetDesc(cortexMemory);
            CalculateActivityAndSuperActivity(cortexMemory.DiscreteRandomVector, Temp_ActivitiyMaxInfo);

            MiniColumn? winnerMiniColumn;
            // Сохраняем воспоминание в миниколонке-победителе.
            //if (randomInitialization)
            //{
            //    var winnerIndex = random.Next(MiniColumns.Length);
            //    winnerMiniColumn = MiniColumns[winnerIndex];
            //}
            //else
            {
                winnerMiniColumn = MiniColumnsActivityHelper.GetSuperActivityMax_MiniColumn(Temp_ActivitiyMaxInfo, random);
            }            
            if (winnerMiniColumn is not null)
            {
                winnerMiniColumn.AddCortexMemory(cortexMemory);
            }
        }
        return false;
    }

    public async Task ReorderMemoriesAsync(int epochCount, Random random, ILogger logger, Func<Task>? epochRefreshAction = null)
    {
        ActivitiyMaxInfo activitiyMaxInfo = new();        
        int min_EpochChangesCount = Int32.MaxValue;

        Stopwatch sw = new();
        for (int epochIndex = 0; epochIndex < epochCount; epochIndex += 1)
        {
            sw.Restart();

            int epochChangesCount = 0;
            foreach (var mci in Enumerable.Range(0, MiniColumns.Data.Length))
            {
                MiniColumn mc = MiniColumns.Data[mci];

                foreach (var mi in Enumerable.Range(0, mc.CortexMemories.Count))
                {
                    Memory? cortexMemory = mc.CortexMemories[mi];
                    if (cortexMemory is null)
                        continue;

                    mc.CortexMemories[mi] = null;

                    Temp_InputCurrentDesc = GetDesc(cortexMemory);
                    CalculateActivityAndSuperActivity(cortexMemory.DiscreteRandomVector, activitiyMaxInfo);

                    // Сохраняем воспоминание в миниколонке-победителе.
                    MiniColumn? winnerMiniColumn = MiniColumnsActivityHelper.GetSuperActivityMax_MiniColumn(activitiyMaxInfo, random);
                    if (winnerMiniColumn is not null)
                    {
                        if (!ReferenceEquals(winnerMiniColumn, mc))
                        {                            
                            winnerMiniColumn.AddCortexMemory(cortexMemory);
                            epochChangesCount += 1;
                        }
                        else
                        {
                            mc.CortexMemories[mi] = cortexMemory;
                        }
                    }                    
                }
            }

            foreach (var mci in Enumerable.Range(0, MiniColumns.Data.Length))
            {
                MiniColumn mc = MiniColumns.Data[mci];
                mc.Temp_CortexMemories.Clear();

                foreach (var mi in Enumerable.Range(0, mc.CortexMemories.Count))
                {
                    Memory? memory = mc.CortexMemories[mi];
                    if (memory is null)
                        continue;

                    mc.Temp_CortexMemories.Add(memory);
                }

                mc.CortexMemories.Swap(mc.Temp_CortexMemories);
                mc.Temp_CortexMemories.Clear();
            }

            sw.Stop();

            logger.LogInformation($"ReorderMemories() epoch finished. ChangedCount: {epochChangesCount}; ElapsedMilliseconds: {sw.ElapsedMilliseconds}");

            if (epochChangesCount < min_EpochChangesCount)
            {                
                min_EpochChangesCount = epochChangesCount;
            }            

            if (epochChangesCount < 10)
            {
                break;
            }
            else
            {
                if (epochRefreshAction is not null)
                    await epochRefreshAction();
            }
        }
    }

    public void CalculateWords(InputCorpusData inputCorpusData, int wordsCount, Random random)
    {
        for (int i = 0; i < wordsCount; i += 1)
        {
            if (inputCorpusData.CurrentWordIndex >= inputCorpusData.Words.Count - 1)
                break;

            inputCorpusData.CurrentWordIndex += 1;

            var word = inputCorpusData.Words[inputCorpusData.CurrentWordIndex];

            Temp_InputCurrentDesc = word.Name;
            CalculateActivityAndSuperActivity(word.DiscreteRandomVector, Temp_ActivitiyMaxInfo);
        }
    }

    public void CalculateCurrentWord(InputCorpusData inputCorpusData, Random random)
    {
        if (inputCorpusData.CurrentWordIndex > 0 && inputCorpusData.CurrentWordIndex < inputCorpusData.Words.Count)
        {
            var word = inputCorpusData.Words[inputCorpusData.CurrentWordIndex];

            Temp_InputCurrentDesc = word.Name;
            CalculateActivityAndSuperActivity(word.DiscreteRandomVector, Temp_ActivitiyMaxInfo);
        }
        else
        {
            Temp_InputCurrentDesc = @"";
            ClearActivityAndSuperActivity(Temp_ActivitiyMaxInfo);
        }
    }

    public void CalculateWords_DiscreteOptimizedVectors(Random random)
    {
        if (MiniColumns.Data.Length <= Constants.DiscreteVectorLength)
        {
            int[] discreteOptimizedVectorIndices = new int[Constants.DiscreteVectorLength];
            for (int i = 0; i < discreteOptimizedVectorIndices.Length; i += 1)
            {
                discreteOptimizedVectorIndices[i] = i;
            }
            random.Shuffle(discreteOptimizedVectorIndices);
            foreach (var mci in Enumerable.Range(0, MiniColumns.Data.Length))
            {
                MiniColumn mc = MiniColumns.Data[mci];
                mc.DiscreteVectorProjectionIndex = discreteOptimizedVectorIndices[mci];
            }
        }
        else
        {
            foreach (var mci in Enumerable.Range(0, MiniColumns.Data.Length))
            {
                MiniColumn mc = MiniColumns.Data[mci];
                mc.DiscreteVectorProjectionIndex = random.Next(Constants.DiscreteVectorLength);
            }
        }

        for (int wordIndex = 0; wordIndex < Words.Count; wordIndex += 1)
        {
            var word = Words[wordIndex];

            Temp_InputCurrentDesc = word.Name;
            CalculateActivityAndSuperActivity(word.DiscreteRandomVector, null);

            Array.Clear(word.DiscreteOptimizedVector);
            foreach (var mc in MiniColumns.Data.OrderByDescending(mc => mc.Temp_Activity).Take(7))
            {
                word.DiscreteOptimizedVector[mc.DiscreteVectorProjectionIndex] = 1.0f;
            }
        }
    }

    public void CalculateWords_DiscreteOptimizedVectors_Metrics(Random random, ILogger logger)
    {
        var wordPairs = DatasetLoader.LoadDataset(Path.Combine(@"Data", @"Ssz.AI.AdvancedEmbedding2", @"EmbeddingsEvaluation", @"ru_simlex965_tagged.tsv"),
            tagsMappingFilePath: Path.Combine(@"Data", @"Ssz.AI.AdvancedEmbedding2", @"EmbeddingsEvaluation", @"TagsMapping.csv"));

        List<WordPair> filteredWordPairs = new List<WordPair>(wordPairs.Count);
        var wordsDicitionary = Words.ToFrozenDictionary(w => w.Name);
        
        foreach (var wordPair in wordPairs)
        {
            var word1 = wordsDicitionary.GetValueOrDefault(wordPair.Word1);
            if (word1 is null)
                continue;

            var word2 = wordsDicitionary.GetValueOrDefault(wordPair.Word2);
            if (word2 is null)
                continue;

            wordPair.CosineSimilarity = TensorPrimitives.CosineSimilarity(word1.DiscreteOptimizedVector, word2.DiscreteOptimizedVector);
            filteredWordPairs.Add(wordPair);
        }

        EmbeddingUtilities.EvaluateCorrelation(filteredWordPairs, logger);
        EmbeddingUtilities.SaveResults(
            filteredWordPairs,
            Path.Combine(@"Data", @"Ssz.AI.AdvancedEmbedding2", @"EmbeddingsEvaluation", @"EvaluationResults.csv"),
            logger);
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {
            writer.WriteListOfOwnedDataSerializable(Words, context);
            Ssz.AI.Helpers.SerializationHelper.SerializeOwnedData_DenseMatrix(MiniColumns, writer, context);            
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    Words = reader.ReadListOfOwnedDataSerializable(() => new Word(), context);
                    MiniColumns = Ssz.AI.Helpers.SerializationHelper.DeserializeOwnedData_DenseMatrix(reader, context, (mcx, mcy) => new MiniColumn(Constants, mcx, mcy));
                    break;
            }
        }
    }

    #endregion    

    private string GetDesc(Memory cortexMemory)
    {
        return String.Join(@" ", cortexMemory.WordIndices.Select(i => Words[i].Name));
    }

    public class MiniColumn : ISerializableModelObject
    {
        public MiniColumn(Model01.ModelConstants constants, int mcx, int mcy)
        {
            Constants = constants;            
            MCX = mcx;
            MCY = mcy;
        }

        public readonly Model01.ModelConstants Constants;

        /// <summary>
        ///     Индекс миниколонки в матрице по оси X (горизонтально вправо)
        /// </summary>
        public readonly int MCX;

        /// <summary>
        ///     Индекс миниколонки в матрице по оси Y (вертикально вниз)
        /// </summary>
        public readonly int MCY;

        /// <summary>
        ///     K для расчета суперактивности.
        ///     <para>(K для позитива, K для негатива, MiniColumn)</para>
        ///     <para>Нулевой элемент, это коэффициент для самой колонки.</para>
        /// </summary>
        public List<(float, float, MiniColumn)> Temp_K_ForNearestMiniColumns = null!;                

        /// <summary>
        ///     Сохраненные хэш-коды
        /// </summary>
        public FastList<Memory?> CortexMemories = null!;

        public int DiscreteVectorProjectionIndex;

        /// <summary>
        ///     Временный список для сохраненных хэш-кодов
        /// </summary>
        public FastList<Memory?> Temp_CortexMemories = null!;

        /// <summary>
        ///     Последнее добавленное воспомининие            
        /// </summary>
        public Memory? Temp_CortexMemory;

        /// <summary>
        ///     Текущая активность миниколонки при подаче примера.
        ///     (Позитивная активность, Негативная активность, Количество воспоминаний)
        /// </summary>
        public (float PositiveActivity, float NegativeActivity, int MemoriesCount) Temp_Activity;

        /// <summary>
        ///     Текущая суммарная активность миниколонки с учетом активностей соседей при подаче примера
        /// </summary>
        public float Temp_SuperActivity;        

        public void AddCortexMemory(Cortex.Memory cortexMemory)
        {
            Temp_CortexMemory = cortexMemory;
            CortexMemories.Add(cortexMemory);
        }

        public void GenerateOwnedData()
        {
            CortexMemories = new(1000);
        }

        public void Prepare()
        {
            Temp_CortexMemories = new(1000);

            Temp_K_ForNearestMiniColumns = new List<(float, float, MiniColumn)>((int)(Math.PI * Constants.SuperActivityRadius_MiniColumns * Constants.SuperActivityRadius_MiniColumns) + 10);
        }

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(2))
            {
                Ssz.AI.Helpers.SerializationHelper.SerializeOwnedData_FastList(CortexMemories, writer, context);
                writer.WriteOptimized(DiscreteVectorProjectionIndex);
            }
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            using (Block block = reader.EnterBlock())
            {
                switch (block.Version)
                {
                    case 1:
                        CortexMemories = Ssz.AI.Helpers.SerializationHelper.DeserializeOwnedData_FastList(reader, context, idx => (Memory?)new Memory());
                        break;
                    case 2:
                        CortexMemories = Ssz.AI.Helpers.SerializationHelper.DeserializeOwnedData_FastList(reader, context, idx => (Memory?)new Memory());
                        DiscreteVectorProjectionIndex = reader.ReadOptimizedInt32();
                        break;
                }
            }
        }
    }

    public class Memory : IOwnedDataSerializable
    {        
        public float[] DiscreteRandomVector = null!;

        public Color DiscreteRandomVector_Color;

        public int[] WordIndices = null!;

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(1))
            {
                writer.WriteArray(DiscreteRandomVector);
                writer.Write(DiscreteRandomVector_Color);
                writer.WriteArray(WordIndices);
            }
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            using (Block block = reader.EnterBlock())
            {
                switch (block.Version)
                {
                    case 1:
                        DiscreteRandomVector = reader.ReadArray<float>()!;
                        DiscreteRandomVector_Color = reader.ReadColor();
                        WordIndices = reader.ReadArray<int>()!;                        
                        break;
                }
            }
        }        
    }

    public class ActivitiyMaxInfo
    {
        public MiniColumn? SelectedSuperActivityMax_MiniColumn;

        public float MaxActivity = float.MinValue;
        public readonly List<MiniColumn> ActivityMax_MiniColumns = new();

        public float MaxSuperActivity = float.MinValue;
        public readonly List<MiniColumn> SuperActivityMax_MiniColumns = new();        
    }
}
