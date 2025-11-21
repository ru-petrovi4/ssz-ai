using Microsoft.Extensions.Logging;
using Ssz.AI.Helpers;
using Ssz.AI.Models.AdvancedEmbeddingModel2.EmbeddingsEvaluation;
using Ssz.Utils;
using Ssz.Utils.Logging;
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
        IMiniColumnsActivityConstants constants, 
        ILogger logger)
    {
        Constants = constants;
        Logger = logger;
    }

    #region public functions

    public readonly IMiniColumnsActivityConstants Constants;

    public readonly ILogger Logger;

    public List<Word> Words { get; private set; } = null!;

    public List<Memory> CortexMemories { get; private set; } = null!;

    public DenseMatrix<MiniColumn> MiniColumns { get; private set; } = null!;

    public readonly ActivitiyMaxInfo Temp_ActivitiyMaxInfo = new();

    public string Temp_InputCurrentDesc = null!;

    public void GenerateOwnedData(List<Word> words, List<Memory> cortexMemories)
    {
        Words = words;
        CortexMemories = cortexMemories;

        MiniColumns = new DenseMatrix<MiniColumn>(Constants.CortexWidth_MiniColumns, Constants.CortexHeight_MiniColumns);

        for (int mcy = 0; mcy < MiniColumns.Dimensions[1]; mcy += 1)
            for (int mcx = 0; mcx < MiniColumns.Dimensions[0]; mcx += 1)
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
                MiniColumn miniColumn = MiniColumns.Data[mci];
                miniColumn.Prepare();
                miniColumn.Temp_K_ForNearestMiniColumns.Add((Constants.PositiveK[0], Constants.NegativeK[0], miniColumn));

                for (int mcy = miniColumn.MCY - (int)Constants.PositiveK.Length - 1; mcy <= miniColumn.MCY + (int)Constants.PositiveK.Length + 1; mcy += 1)
                    for (int mcx = miniColumn.MCX - (int)Constants.PositiveK.Length - 1; mcx <= miniColumn.MCX + (int)Constants.PositiveK.Length + 1; mcx += 1)
                    {
                        if (mcx < 0 ||
                                mcx >= MiniColumns.Dimensions[0] ||
                                mcy < 0 ||
                                mcy >= MiniColumns.Dimensions[1] ||
                                (mcx == miniColumn.MCX && mcy == miniColumn.MCY))
                            continue;

                        MiniColumn nearestMc = MiniColumns[mcx, mcy];
                        if (nearestMc is null)
                            continue;
                        float r = MathF.Sqrt((mcx - miniColumn.MCX) * (mcx - miniColumn.MCX) + (mcy - miniColumn.MCY) * (mcy - miniColumn.MCY));
                        if (r < Constants.PositiveK.Length + 0.00001f)
                        {
                            float k0 = MathHelper.GetInterpolatedValue(Constants.PositiveK, r);
                            float k1 = MathHelper.GetInterpolatedValue(Constants.NegativeK, r);
                            miniColumn.Temp_K_ForNearestMiniColumns.Add((k0, k1, nearestMc));
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
    public bool Calculate_PutPhrases_Randomly(InputCorpusData inputCorpusData, int cortexMemoriesCount, Random random)
    {
        ClearActivityAndSuperActivity(Temp_ActivitiyMaxInfo);

        try
        {
            for (int i = 0; i < cortexMemoriesCount; i += 1)
            {
                if (inputCorpusData.CurrentCortexMemoryIndex >= inputCorpusData.CortexMemories.Count - 1)
                    return true;

                inputCorpusData.CurrentCortexMemoryIndex += 1;

                var cortexMemory = inputCorpusData.CortexMemories[inputCorpusData.CurrentCortexMemoryIndex];

                Temp_InputCurrentDesc = GetDesc(cortexMemory);                

                MiniColumn? winnerMiniColumn;
                // Сохраняем воспоминание в миниколонке-победителе.
                //if (randomInitialization)
                //{
                //    var winnerIndex = random.Next(MiniColumns.Length);
                //    winnerMiniColumn = MiniColumns[winnerIndex];
                //}
                //else
                {
                    winnerMiniColumn = Temp_ActivitiyMaxInfo.GetSuperActivityMax_MiniColumn(random);
                }
                if (winnerMiniColumn is not null)
                {
                    winnerMiniColumn.AddCortexMemory(cortexMemory);
                }
            }
            return false;
        }
        finally
        {
            Logger.LogInformation($"CalculateCortexMemories() {inputCorpusData.CurrentCortexMemoryIndex}/{inputCorpusData.CortexMemories.Count} finished.");
        }
    }       

    public void Calculate_CurrentWord(InputCorpusData inputCorpusData, Random random)
    {
        if (inputCorpusData.Current_OrderedWords_Index > 0 && inputCorpusData.Current_OrderedWords_Index < inputCorpusData.Words.Count)
        {
            var word = inputCorpusData.OrderedWords[inputCorpusData.Current_OrderedWords_Index];

            Temp_InputCurrentDesc = word.Name;
            CalculateActivityAndSuperActivity(word.DiscreteRandomVector, Temp_ActivitiyMaxInfo);
        }
        else
        {
            Temp_InputCurrentDesc = @"";
            ClearActivityAndSuperActivity(Temp_ActivitiyMaxInfo);
        }
    }

    public void Calculate_CurrentCortexMemory(InputCorpusData inputCorpusData, Random random)
    {
        if (inputCorpusData.CurrentCortexMemoryIndex > 0 && inputCorpusData.CurrentCortexMemoryIndex < inputCorpusData.CortexMemories.Count)
        {
            var cortexMemory = inputCorpusData.CortexMemories[inputCorpusData.CurrentCortexMemoryIndex];

            Temp_InputCurrentDesc = GetDesc(cortexMemory);
            CalculateActivityAndSuperActivity(cortexMemory.DiscreteRandomVector, Temp_ActivitiyMaxInfo);
        }
        else
        {
            Temp_InputCurrentDesc = @"";
            ClearActivityAndSuperActivity(Temp_ActivitiyMaxInfo);
        }
    }

    public void Calculate_Words_DiscreteOptimizedVectors(Random random)
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

    public void Calculate_Words_DiscreteOptimizedVectors_Metrics(Random random, ILogger logger)
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

    public class MiniColumn : IMiniColumn, IMiniColumnActivity, ISerializableModelObject
    {
        public MiniColumn(IMiniColumnsActivityConstants constants, int mcx, int mcy)
        {
            Constants = constants;            
            MCX = mcx;
            MCY = mcy;
        }

        public readonly IMiniColumnsActivityConstants Constants;

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
        public FastList<(float, float, IMiniColumnActivity)> Temp_K_ForNearestMiniColumns = null!;

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
        public (float PositiveActivity, float NegativeActivity, int CortexMemoriesCount) Temp_Activity;

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

            Temp_K_ForNearestMiniColumns = new FastList<(float, float, IMiniColumnActivity)>((int)(Math.PI * Constants.PositiveK.Length * Constants.PositiveK.Length) + 10);
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

        IFastList<ICortexMemory?> IMiniColumn.CortexMemories => CortexMemories;

        IMiniColumn IMiniColumnActivity.MiniColumn => this;

        (float PositiveActivity, float NegativeActivity, int CortexMemoriesCount) IMiniColumnActivity.Activity => Temp_Activity;

        float IMiniColumnActivity.SuperActivity => Temp_SuperActivity;

        IFastList<(float, float, IMiniColumnActivity)> IMiniColumnActivity.K_ForNearestMiniColumns => Temp_K_ForNearestMiniColumns;        
    }

    public class MiniColumnActivity : IMiniColumnActivity
    {
        public MiniColumnActivity(IMiniColumn miniColumn) 
        {
            MiniColumn = miniColumn;
        }

        public IMiniColumn MiniColumn { get; }

        /// <summary>
        ///     K для расчета суперактивности.
        ///     <para>(K для позитива, K для негатива, MiniColumn)</para>
        ///     <para>Нулевой элемент, это коэффициент для самой колонки.</para>
        /// </summary>
        public IFastList<(float, float, IMiniColumnActivity)> K_ForNearestMiniColumns { get; set; } = null!;        

        /// <summary>
        ///     Текущая активность миниколонки при подаче примера.
        ///     (Позитивная активность, Негативная активность, Количество воспоминаний)
        /// </summary>
        public (float PositiveActivity, float NegativeActivity, int CortexMemoriesCount) Activity { get; set; }

        /// <summary>
        ///     Текущая суммарная активность миниколонки с учетом активностей соседей при подаче примера
        /// </summary>
        public float SuperActivity { get; set; }
    }

    public class Memory : ICortexMemory, IOwnedDataSerializable
    {        
        public float[] DiscreteRandomVector = null!;

        public Color DiscreteRandomVector_Color;

        public int[] WordIndices = null!;

        public DenseMatrix<MiniColumnActivity> Temp_MiniColumnActivities { get; set; } = null!;

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

        float[] ICortexMemory.DiscreteVector => DiscreteRandomVector;
    }

    public class ActivitiyMaxInfo
    {
        public MiniColumn? SelectedSuperActivityMax_MiniColumn;

        public float MaxActivity = float.MinValue;
        public readonly List<MiniColumn> ActivityMax_MiniColumns = new();

        public float MaxSuperActivity = float.MinValue;
        public readonly List<MiniColumn> SuperActivityMax_MiniColumns = new();

        public MiniColumn? GetSuperActivityMax_MiniColumn(Random random)
        {
            if (SuperActivityMax_MiniColumns.Count == 0)
                SelectedSuperActivityMax_MiniColumn = null;
            else if (SuperActivityMax_MiniColumns.Count == 1)
                SelectedSuperActivityMax_MiniColumn = SuperActivityMax_MiniColumns[0];
            else
                SelectedSuperActivityMax_MiniColumn = SuperActivityMax_MiniColumns[random.Next(SuperActivityMax_MiniColumns.Count)];
            return SelectedSuperActivityMax_MiniColumn;
        }
    }
}
