using Microsoft.Extensions.Logging;
using Ssz.AI.Helpers;
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
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace Ssz.AI.Models.CortexVisualisationModel;

public interface ICortexConstants
{
    int CotrexWidth_MiniColumns { get; }

    int CotrexHeight_MiniColumns { get; }

    /// <summary>
    ///     Олпределенный зараниие радиус гиперколонки в миниколонках.
    /// </summary>
    int HypercolumnDefinedRadius_MiniColumns { get; }

    /// <summary>
    ///     Уровень подобия для нулевой активности
    /// </summary>
    float K0 { get; set; }

    /// <summary>
    ///     Уровень подобия с пустой миниколонкой.
    ///     Штраф за воспоминания (для равномерности заполнения).
    /// </summary>
    float K2 { get; set; }

    /// <summary>
    ///     Порог энергии
    /// </summary>
    float K4 { get; set; }

    float[] PositiveK { get; set; }

    float[] NegativeK { get; set; }

    /// <summary>
    ///     Включен ли порог энергии при накоплении воспоминаний
    /// </summary>
    public bool TotalEnergyThreshold { get; set; }
}

public partial class Cortex : ISerializableModelObject
{
    /// <summary>
    ///     Если задано SubAreaMiniColumnsCount, то генерируется только подмножество миниколонок с центром SubAreaCenter_Cx, SubAreaCenter_Cy и количеством SubAreaMiniColumnsCount
    /// </summary>
    /// <param name="constants"></param>        
    public Cortex(
        ICortexConstants constants, 
        ILogger logger)
    {
        Constants = constants;
        Logger = logger;
        InputItems = new(10000);
    }

    #region public functions

    public readonly ICortexConstants Constants;

    public readonly ILogger Logger;

    /// <summary>
    ///     Первое воспоминеие нулевое в идеальной вертушке. Следующие 6 воспоминаний вокруг нулевого в идеальной вертушке.
    /// </summary>
    public List<InputItem> InputItems { get; private set; } = null!;

    public FastList<MiniColumn> MiniColumns { get; private set; } = null!;    

    public string Temp_InputCurrentDesc = null!;

    public void GenerateOwnedData(Random random, bool onlyCeneterHypercolumn)
    {
        MiniColumns = new FastList<MiniColumn>(Constants.CotrexWidth_MiniColumns * Constants.CotrexHeight_MiniColumns);
        
        float delta_MCY = MathF.Sqrt(1.0f - 0.5f * 0.5f);        

        MiniColumn? centerMiniColumn = null;
        FastList<MiniColumn> centerMiniColumn_AdjacentMiniColumns = new FastList<MiniColumn>(6);

        if (onlyCeneterHypercolumn)
        {
            float maxRadius = Constants.HypercolumnDefinedRadius_MiniColumns + 0.00001f;

            for (int mcj = -(int)(Constants.HypercolumnDefinedRadius_MiniColumns / delta_MCY); mcj <= (int)(Constants.HypercolumnDefinedRadius_MiniColumns / delta_MCY); mcj += 1)
                for (int mci = -Constants.HypercolumnDefinedRadius_MiniColumns; mci <= Constants.HypercolumnDefinedRadius_MiniColumns; mci += 1)
                {
                    float mcx = mci + ((mcj % 2 == 0) ? 0.0f : 0.5f);
                    float mcy = mcj * delta_MCY;

                    float radius = MathF.Sqrt(mcx * mcx + mcy * mcy);
                    if (radius < maxRadius)
                    {
                        MiniColumn miniColumn = new MiniColumn(
                            Constants)
                        {
                            MCX = mcx,
                            MCY = mcy
                        };

                        miniColumn.GenerateOwnedData();

                        MiniColumns.Add(miniColumn);

                        if (radius < 0.00001f)
                            centerMiniColumn = miniColumn;
                        else if (radius < 1.00001f)
                            centerMiniColumn_AdjacentMiniColumns.Add(miniColumn);
                    }
                }
        }
        else
        {
            for (int mcj = -(int)(Constants.CotrexHeight_MiniColumns / (2.0f * delta_MCY)); mcj <= (int)(Constants.CotrexHeight_MiniColumns / (2.0f * delta_MCY)); mcj += 1)
                for (int mci = -(int)(Constants.CotrexWidth_MiniColumns / 2.0f); mci <= (int)(Constants.CotrexWidth_MiniColumns / 2.0f); mci += 1)
                {
                    float mcx = mci + ((mcj % 2 == 0) ? 0.0f : 0.5f);
                    float mcy = mcj * delta_MCY;

                    float radius = MathF.Sqrt(mcx * mcx + mcy * mcy);

                    MiniColumn miniColumn = new MiniColumn(
                            Constants)
                    {
                        MCX = mcx,
                        MCY = mcy
                    };

                    miniColumn.GenerateOwnedData();

                    MiniColumns.Add(miniColumn);

                    if (radius < 0.00001f)
                        centerMiniColumn = miniColumn;
                    else if (radius < 1.00001f)
                        centerMiniColumn_AdjacentMiniColumns.Add(miniColumn);
                }
        }

        // Воспоминания для оценки качества вертушки
        AddInputItem(random, centerMiniColumn!);
        foreach (var miniColumn in centerMiniColumn_AdjacentMiniColumns
            .OrderBy(mc => MathF.Atan2(mc.MCY, mc.MCX)))
        {
            AddInputItem(random, miniColumn);
        }
    }

    public void Prepare()
    {
        // Находим ближайшие миниколонки для каждой миниколонки
        Parallel.For(
            fromInclusive: 0,
            toExclusive: MiniColumns.Count,
            mci =>
            {
                MiniColumn miniColumn = MiniColumns[mci];                
                miniColumn.Prepare();                                           

                for (int mci2 = 0; mci2 < MiniColumns.Count; mci2 += 1)                    
                {
                    if (mci2 == mci)
                    {                        
                        miniColumn.Temp_NearestForEnergyMiniColumns.Add((0, miniColumn));
                        continue;
                    }

                    MiniColumn nearestMc = MiniColumns[mci2];

                    //double k = MathF.Pow((mcx - miniColumn.MCX) * (mcx - miniColumn.MCX) + (mcy - miniColumn.MCY) * (mcy - miniColumn.MCY), 1.0f);
                    //if (Math.Abs(mcx - miniColumn.MCX) <= 15 && Math.Abs(mcy - miniColumn.MCY) <= 15)
                    //    miniColumn.Temp_K_ForNearestMiniColumns.Add((k, nearestMc));

                    //miniColumn.Temp_CandidateForSwapMiniColumns.Add(nearestMc);

                    float k = (nearestMc.MCX - miniColumn.MCX) * (nearestMc.MCX - miniColumn.MCX) + (nearestMc.MCY - miniColumn.MCY) * (nearestMc.MCY - miniColumn.MCY);
                    float r = MathF.Sqrt(k);

                    if (r < 2.00001f)
                        miniColumn.Temp_K_ForNearestMiniColumns.Add(
                            (MathHelper.GetInterpolatedValue(Constants.PositiveK, r),
                            MathHelper.GetInterpolatedValue(Constants.NegativeK, r),
                            nearestMc));

                    //if (r < 3.00001f)
                    miniColumn.Temp_NearestForEnergyMiniColumns.Add((k, nearestMc));
                    
                    if (r < 1.00001)
                        miniColumn.Temp_CandidateForSwapMiniColumns.Add((r, nearestMc));

                    if (r < 1.00001f)
                        miniColumn.Temp_AdjacentMiniColumns.Add((r, nearestMc));
                }

                miniColumn.Temp_AdjacentMiniColumns = miniColumn.Temp_AdjacentMiniColumns
                    .OrderBy(it => MathF.Atan2(miniColumn.MCY - it.Item2.MCY, miniColumn.MCX - it.Item2.MCX))
                    .ToFastList();
            });
    }

    public InputItem AddInputItem(Random random, MiniColumn miniColumn)
    {
        InputItem inputItem = new();
        inputItem.Index = InputItems.Count;
        inputItem.Angle = MathHelper.NormalizeAngle(MathF.Atan2(miniColumn.MCY, miniColumn.MCX));
        inputItem.Magnitude = MathF.Sqrt(miniColumn.MCY * miniColumn.MCY + miniColumn.MCX * miniColumn.MCX);

        float s = MathF.Sqrt(inputItem.Magnitude / (Constants.HypercolumnDefinedRadius_MiniColumns + 1));
        inputItem.Color = Visualisation.ColorFromHSV((double)(inputItem.Angle + MathF.PI) / (2 * MathF.PI), s, 1.0);
        inputItem.SimilarityThreshold = 0.00f * (1.0f - inputItem.Magnitude / 3.0f);

        InputItems.Add(inputItem);
        return inputItem;
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {
            writer.WriteListOfOwnedDataSerializable(InputItems, context);
            Ssz.AI.Helpers.SerializationHelper.SerializeOwnedData_FastList(MiniColumns, writer, context);            
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    InputItems = reader.ReadListOfOwnedDataSerializable(() => new InputItem(), context);
                    MiniColumns = Ssz.AI.Helpers.SerializationHelper.DeserializeOwnedData_FastList(reader, context, idx => new MiniColumn(Constants));
                    break;
            }
        }
    }

    #endregion         

    public class MiniColumn : ISerializableModelObject
    {
        public MiniColumn(ICortexConstants constants)
        {
            Constants = constants;
        }

        public readonly ICortexConstants Constants;

        /// <summary>
        ///     Координата миниколонки по оси X (горизонтально вправо)
        /// </summary>
        public float MCX;

        /// <summary>
        ///     Координата миниколонки по оси Y (вертикально вниз)
        /// </summary>
        public float MCY;

        /// <summary>
        ///     Окружающие миниколонки, для которых считается суперактивность.
        ///     <para>(k, MiniColumn)</para>        
        /// </summary>
        public FastList<(float PositiveK, float NegativeK, MiniColumn MiniColumn)> Temp_K_ForNearestMiniColumns = null!;

        /// <summary>
        ///     Сама миниколонка и окружающие миниколонки, для которых считается энергия.
        ///     <para>(r^2, MiniColumn)</para>        
        /// </summary>
        public FastList<(double, MiniColumn)> Temp_NearestForEnergyMiniColumns = null!;

        /// <summary>
        ///     Миниколонки - кандидаты для перестановки воспоминаний.
        ///     <para>(r, MiniColumn)</para>        
        /// </summary>
        public FastList<(double, MiniColumn)> Temp_CandidateForSwapMiniColumns = null!;

        /// <summary>
        ///     Миниколонки - ближайшие соседи.
        ///     <para>(r, MiniColumn)</para>        
        /// </summary>
        public FastList<(double, MiniColumn)> Temp_AdjacentMiniColumns = null!;

        public double Temp_MiniColumnEnergy;

        public (float PositiveActivity, float NegativeActivity, int CortexMemoriesCount) Temp_Activity;

        /// <summary>
        ///     Total system energy if memory in this minicolumn.
        /// </summary>
        public float Temp_TotalEnergy;

        /// <summary>
        ///     Сохраненные хэш-коды
        /// </summary>
        public FastList<Memory?> CortexMemories = null!;

        /// <summary>
        ///     Сохраненные хэш-коды
        /// </summary>
        public FastList<Memory?> Temp_CortexMemories = null!;

        public void GenerateOwnedData()
        {
            CortexMemories = new(10);            
        }

        public void Prepare()
        {
            Temp_K_ForNearestMiniColumns = new FastList<(float, float, MiniColumn)>(18);
            Temp_NearestForEnergyMiniColumns = new FastList<(double, MiniColumn)>((int)(Math.PI * Constants.HypercolumnDefinedRadius_MiniColumns * Constants.HypercolumnDefinedRadius_MiniColumns));
            Temp_CandidateForSwapMiniColumns = new FastList<(double, MiniColumn)>((int)(Math.PI * Constants.HypercolumnDefinedRadius_MiniColumns * Constants.HypercolumnDefinedRadius_MiniColumns));
            Temp_AdjacentMiniColumns = new FastList<(double, MiniColumn)>(6);
            Temp_CortexMemories = new(10);
        }

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(1))
            {
                writer.Write(MCX);
                writer.Write(MCY);
                Ssz.AI.Helpers.SerializationHelper.SerializeOwnedData_FastList(CortexMemories, writer, context);                
            }
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            using (Block block = reader.EnterBlock())
            {
                switch (block.Version)
                {
                    case 1:
                        MCX = reader.ReadSingle();
                        MCY = reader.ReadSingle();
                        CortexMemories = Ssz.AI.Helpers.SerializationHelper.DeserializeOwnedData_FastList(reader, context, idx => (Memory?)new Memory());
                        break;                    
                }
            }
        } 
    }

    public class Memory : IOwnedDataSerializable
    {   
        public static readonly Memory IdealPinwheelCenterMemory = new Memory() { InputItemIndex = 0 };

        public static readonly Memory[] IdealPinwheelMemories = [
            new Memory() { InputItemIndex = 1 },
            new Memory() { InputItemIndex = 2 },
            new Memory() { InputItemIndex = 3 },
            new Memory() { InputItemIndex = 4 },
            new Memory() { InputItemIndex = 5 },
            new Memory() { InputItemIndex = 6 },
            ];

        public int InputItemIndex;      

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(1))
            {
                writer.Write(InputItemIndex);                
            }
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            using (Block block = reader.EnterBlock())
            {
                switch (block.Version)
                {
                    case 1:
                        InputItemIndex = reader.ReadInt32();                        
                        break;
                }
            }
        }
    }    
}
