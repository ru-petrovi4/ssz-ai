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
    ///     Минимальное подобие для мини-вертушки.
    /// </summary>
    float K3 { get; set; }

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

    public FastList<MiniColumn> Temp_CenterHyperColumn_MiniColumns = null!;

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
        Temp_CenterHyperColumn_MiniColumns = new FastList<MiniColumn>((int)(Math.PI * Constants.HypercolumnDefinedRadius_MiniColumns * Constants.HypercolumnDefinedRadius_MiniColumns));

        float hypercolumnDefinedRadius_MiniColumns = Constants.HypercolumnDefinedRadius_MiniColumns + 0.000001f;

        for (int mci = 0; mci < MiniColumns.Count; mci += 1)            
        {
            MiniColumn miniColumn = MiniColumns[mci];
            miniColumn.Prepare();

            float abs_r2 = miniColumn.MCX * miniColumn.MCX + miniColumn.MCY * miniColumn.MCY;
            float abs_r = MathF.Sqrt(abs_r2);

            if (abs_r < hypercolumnDefinedRadius_MiniColumns)
                Temp_CenterHyperColumn_MiniColumns.Add(miniColumn);
        };

        // Находим ближайшие миниколонки для каждой миниколонки
        Parallel.For(
            fromInclusive: 0,
            toExclusive: MiniColumns.Count,
            (Action<int>)(mci =>
            {
                MiniColumn miniColumn = MiniColumns[mci];                                

                for (int mci2 = 0; mci2 < MiniColumns.Count; mci2 += 1)                    
                {
                    if (mci2 == mci)
                        continue;

                    MiniColumn nearestMc = MiniColumns[mci2];                    

                    float r2 = (nearestMc.MCX - miniColumn.MCX) * (nearestMc.MCX - miniColumn.MCX) + (nearestMc.MCY - miniColumn.MCY) * (nearestMc.MCY - miniColumn.MCY);
                    float r = MathF.Sqrt(r2);

                    if (r < 2.00001f)
                        miniColumn.Temp_K_ForNearestMiniColumns.Add(
                            (MathHelper.GetInterpolatedValue(Constants.PositiveK, r),
                            MathHelper.GetInterpolatedValue(Constants.NegativeK, r),
                            nearestMc));                    
                    
                    if (r < hypercolumnDefinedRadius_MiniColumns)
                        miniColumn.Temp_HyperColumnMiniColumns.Add((r, nearestMc));

                    if (r < 1.00001f)
                        miniColumn.Temp_AdjacentMiniColumns.Add((r, nearestMc));
                }

                miniColumn.Temp_AdjacentMiniColumns = miniColumn.Temp_AdjacentMiniColumns
                    .OrderBy(it => MathF.Atan2(miniColumn.MCY - it.Item2.MCY, miniColumn.MCX - it.Item2.MCX))
                    .ToFastList();
            }));
    }

    public InputItem AddInputItem(Random random, MiniColumn centerHyperColumn_MiniColumn)
    {
        InputItem inputItem = new();
        inputItem.Index = InputItems.Count;
        inputItem.Angle = MathHelper.NormalizeAngle(MathF.Atan2(centerHyperColumn_MiniColumn.MCY, centerHyperColumn_MiniColumn.MCX));
        inputItem.Magnitude = MathF.Sqrt(centerHyperColumn_MiniColumn.MCY * centerHyperColumn_MiniColumn.MCY + centerHyperColumn_MiniColumn.MCX * centerHyperColumn_MiniColumn.MCX);
        inputItem.DistanceFromCenter = inputItem.Magnitude;
        var distanceFromCenterNormalized = MathF.Sqrt(inputItem.Magnitude / (Constants.HypercolumnDefinedRadius_MiniColumns + 1));
        inputItem.Color = Visualisation.ColorFromHSV((double)(inputItem.Angle + MathF.PI) / (2 * MathF.PI), distanceFromCenterNormalized, 1.0);        

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
        ///     Сама миниколонка и окружающие миниколонки размером примерно с гиперколонку.
        ///     <para>(r, MiniColumn)</para>        
        /// </summary>
        public FastList<(double, MiniColumn)> Temp_HyperColumnMiniColumns = null!;        

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
            Temp_HyperColumnMiniColumns = new FastList<(double, MiniColumn)>((int)(Math.PI * Constants.HypercolumnDefinedRadius_MiniColumns * Constants.HypercolumnDefinedRadius_MiniColumns));            
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
        public static readonly Memory IdealPinwheelCenterMemory = new Memory()
        {
            InputItemIndex = 0,
            DistanceFromCenter = 0
        };
        
        public static readonly Memory[] IdealPinwheelMemories = [
            new Memory() { InputItemIndex = 1, DistanceFromCenter = 1 },
            new Memory() { InputItemIndex = 2, DistanceFromCenter = 1 },
            new Memory() { InputItemIndex = 3, DistanceFromCenter = 1 },
            new Memory() { InputItemIndex = 4, DistanceFromCenter = 1 },
            new Memory() { InputItemIndex = 5, DistanceFromCenter = 1 },
            new Memory() { InputItemIndex = 6, DistanceFromCenter = 1 },
            ];

        public int InputItemIndex;

        /// <summary>
        ///    Distance from center in ideal pinwheel in minicolumns
        /// </summary>
        public float DistanceFromCenter;

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(1))
            {
                writer.Write(InputItemIndex);
                writer.Write(DistanceFromCenter);
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
                        DistanceFromCenter = reader.ReadSingle();
                        break;
                }
            }
        }

        public static Memory FromInputItem(InputItem inputItem)
        {
            return new Memory()
            {
                InputItemIndex = inputItem.Index,
                DistanceFromCenter = inputItem.DistanceFromCenter
            };
        }
    }    
}
