using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Linq;

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

        MiniColumns = new DenseMatrix<MiniColumn>(constants.CortexWidth_MiniColumns, constants.CortexHeight_MiniColumns);

        // Создаем только миниколонки для подобласти            
        foreach (int mcy in Enumerable.Range(0, constants.CortexHeight_MiniColumns))
            foreach (int mcx in Enumerable.Range(0, constants.CortexWidth_MiniColumns))
            {
                MiniColumn miniColumn = new MiniColumn(
                    constants,
                    mcx,
                    mcy);

                MiniColumns[mcx, mcy] = miniColumn;
            }
    }

    #region public functions

    public Model01.ModelConstants Constants { get; }

    public DenseMatrix<MiniColumn> MiniColumns { get; }

    public ActivitiyMaxInfo Temp_ActivitiyMaxInfo { get; } = new();

    public void GenerateOwnedData()
    {
        
    }

    public void Prepare()
    {        
    }

    public void CalculateCortexMemories(InputCorpusData inputCorpusData, Random r)
    {
        for (int cortexMemoryIndex = 0; cortexMemoryIndex < inputCorpusData.CortexMemories.Count; cortexMemoryIndex += 1)
        {
            var cortexMemory = inputCorpusData.CortexMemories[cortexMemoryIndex];

            CalculateActivityAndSuperActivity(cortexMemory, Temp_ActivitiyMaxInfo);

            MiniColumn? winnerMiniColumn;
            // Сохраняем воспоминание в миниколонке-победителе.
            //if (randomInitialization)
            //{
            //    var winnerIndex = random.Next(Cortex.SubArea_MiniColumns.Length);
            //    winnerMiniColumn = Cortex.SubArea_MiniColumns[winnerIndex];
            //}
            //else
            {
                winnerMiniColumn = Temp_ActivitiyMaxInfo.GetSuperActivityMax_MiniColumn(r);
            }            
            if (winnerMiniColumn is not null)
            {
                winnerMiniColumn.AddCortexMemory(cortexMemory);
            }
        }
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    break;
            }
        }
    }

    #endregion    

    public class MiniColumn : ISerializableModelObject
    {
        public MiniColumn(Model01.ModelConstants constants, int mcx, int mcy)
        {
            Constants = constants;            
            MCX = mcx;
            MCY = mcy;
            CortexMemories = new(1000);
            Temp_CortexMemories = new(1000);            

            K_ForNearestMiniColumns = new List<(float, float, MiniColumn)>((int)(Math.PI * constants.SuperActivityRadius_MiniColumns * constants.SuperActivityRadius_MiniColumns) + 10);
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
        ///     (K для позитива, K для негатива, MiniColumn)
        /// </summary>
        public readonly List<(float, float, MiniColumn)> K_ForNearestMiniColumns;

        /// <summary>
        ///     K0 для расчета суперактивности.
        /// </summary>
        public (float, float) K0;        

        /// <summary>
        ///     Сохраненные хэш-коды
        /// </summary>
        public List<Memory?> CortexMemories;

        /// <summary>
        ///     Временный список для сохраненных хэш-кодов
        /// </summary>
        public List<Memory> Temp_CortexMemories;

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

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(1))
            {
            }
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            using (Block block = reader.EnterBlock())
            {
                switch (block.Version)
                {
                    case 1:
                        break;
                }
            }
        }
    }

    public class Memory
    {
        public Memory(Model01.ModelConstants constants)
        {
            DiscreteRandomVector = new float[constants.DiscreteVectorLength];
        }

        public float[] DiscreteRandomVector = null!;

        public Word[] Words = null!;
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
            {
                SelectedSuperActivityMax_MiniColumn = null;
            }
            else if (SuperActivityMax_MiniColumns.Count == 1)
            {
                SelectedSuperActivityMax_MiniColumn = SuperActivityMax_MiniColumns[0];
            }
            else
            {
                SelectedSuperActivityMax_MiniColumn = SuperActivityMax_MiniColumns[random.Next(SuperActivityMax_MiniColumns.Count)];
            }
            return SelectedSuperActivityMax_MiniColumn;
        }
    }
}
