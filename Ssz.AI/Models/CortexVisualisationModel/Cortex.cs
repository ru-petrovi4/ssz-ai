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
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;

namespace Ssz.AI.Models.CortexVisualisationModel;

public interface ICortexConstants
{
    int CotrexWidth_MiniColumns { get; }

    int CotrexHeight_MiniColumns { get; }

    float HyperColumnDiameter_Retina { get; }

    /// <summary>
    ///     Олпределенный зараниие радиус гиперколонки в миниколонках.
    /// </summary>
    int HyperColumnDefinedRadius_MiniColumns { get; }

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
    bool TotalEnergyThreshold { get; set; }

    /// <summary>
    ///     Режим с одним воспоминанием в миниколонке.
    /// </summary>
    bool SingleMemory { get; set; }
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

        MiniColumnX_Retina = constants.HyperColumnDiameter_Retina / (2.0f * constants.HyperColumnDefinedRadius_MiniColumns);
        MiniColumnY_Retina = constants.HyperColumnDiameter_Retina / (2.0f * constants.HyperColumnDefinedRadius_MiniColumns);
        HyperColumnDiameter_Retina2 = constants.HyperColumnDiameter_Retina * constants.HyperColumnDiameter_Retina;
    }

    #region public functions

    public readonly ICortexConstants Constants;

    public readonly ILogger Logger;

    public readonly float MiniColumnX_Retina;

    public readonly float MiniColumnY_Retina;

    public readonly float HyperColumnDiameter_Retina2;

    /// <summary>
    ///     Первое воспоминеие нулевое в идеальной вертушке. Следующие 6 воспоминаний вокруг нулевого в идеальной вертушке.
    /// </summary>
    public FastList<InputItem> InputItems { get; private set; } = null!;

    public FastList<Memory> IdealPinwheelMemories { get; private set; } = null!;

    //public static readonly Memory IdealPinwheelCenterMemory = new Memory()
    //{
    //    InputItemIndex = 0,
    //    DistanceFromCenter = 0
    //};

    //public static readonly Memory[] IdealPinwheelMemories = [
    //    new Memory() { InputItemIndex = 1, DistanceFromCenter = 1 },
    //        new Memory() { InputItemIndex = 2, DistanceFromCenter = 1 },
    //        new Memory() { InputItemIndex = 3, DistanceFromCenter = 1 },
    //        new Memory() { InputItemIndex = 4, DistanceFromCenter = 1 },
    //        new Memory() { InputItemIndex = 5, DistanceFromCenter = 1 },
    //        new Memory() { InputItemIndex = 6, DistanceFromCenter = 1 },
    //        ];

    public FastList<MiniColumn> MiniColumns { get; private set; } = null!;

    public FastList<int> HyperColumnCenters_MiniColumnIndices { get; private set; } = null!;

    public string Temp_InputCurrentDesc = null!;

    public void GenerateOwnedData(Random random, bool onlyCeneterHypercolumn)
    {
        MiniColumns = new FastList<MiniColumn>(Constants.CotrexWidth_MiniColumns * Constants.CotrexHeight_MiniColumns);

        float delta_MCX = 1.0f;
        float delta_MCY = MathF.Sqrt(1.0f - 0.5f * 0.5f);

        if (onlyCeneterHypercolumn)
        {
            float maxRadius = Constants.HyperColumnDefinedRadius_MiniColumns + 0.00001f;

            MiniColumn? global_CenterMiniColumn = null;

            for (int mcj = -(int)(Constants.HyperColumnDefinedRadius_MiniColumns / delta_MCY); mcj <= (int)(Constants.HyperColumnDefinedRadius_MiniColumns / delta_MCY); mcj += 1)
                for (int mci = -Constants.HyperColumnDefinedRadius_MiniColumns; mci <= Constants.HyperColumnDefinedRadius_MiniColumns; mci += 1)
                {
                    float mcx = mci + ((mcj % 2 == 0) ? 0.0f : 0.5f);
                    float mcy = mcj * delta_MCY;

                    float radius = MathF.Sqrt(mcx * mcx + mcy * mcy);
                    if (radius < maxRadius)
                    {
                        MiniColumn miniColumn = new MiniColumn(
                            Constants)
                        {
                            Index = MiniColumns.Count,
                            MCX = mcx,
                            MCY = mcy
                        };

                        miniColumn.GenerateOwnedData();

                        MiniColumns.Add(miniColumn);

                        if (radius < 0.00001f)
                            global_CenterMiniColumn = miniColumn;                        
                    }
                }

            HyperColumnCenters_MiniColumnIndices = new FastList<int>([ global_CenterMiniColumn!.Index ]);
        }
        else
        {
            float maxRadius = Math.Min(Constants.CotrexWidth_MiniColumns / 2.0f, Constants.CotrexHeight_MiniColumns / 2.0f) + 0.00001f;

            for (int mcj = -(int)(Constants.CotrexHeight_MiniColumns / (2.0f * delta_MCY)); mcj <= (int)(Constants.CotrexHeight_MiniColumns / (2.0f * delta_MCY)); mcj += 1)
                for (int mci = -(int)(Constants.CotrexWidth_MiniColumns / 2.0f); mci <= (int)(Constants.CotrexWidth_MiniColumns / 2.0f); mci += 1)
                {
                    float mcx = mci + ((mcj % 2 == 0) ? 0.0f : 0.5f);
                    float mcy = mcj * delta_MCY;

                    float radius = MathF.Sqrt(mcx * mcx + mcy * mcy);
                    if (radius < maxRadius)
                    {
                        MiniColumn miniColumn = new MiniColumn(
                            Constants)
                        {
                            Index = MiniColumns.Count,
                            MCX = mcx,
                            MCY = mcy
                        };

                        miniColumn.GenerateOwnedData();

                        MiniColumns.Add(miniColumn);
                    }
                }

            int x_Limit_MiniColumns = Constants.CotrexWidth_MiniColumns / 2 - 1;
            int y_Limit_MiniColumns = Constants.CotrexHeight_MiniColumns / 2 - 1;

            HyperColumnCenters_MiniColumnIndices = new FastList<int>(20);
            delta_MCX = Constants.HyperColumnDefinedRadius_MiniColumns * 2.0f;
            delta_MCY = MathF.Sqrt(3.0f * Constants.HyperColumnDefinedRadius_MiniColumns * Constants.HyperColumnDefinedRadius_MiniColumns);
            for (int mcj = -(int)(Constants.CotrexHeight_MiniColumns / (2.0f * delta_MCY)); mcj <= (int)(Constants.CotrexHeight_MiniColumns / (2.0f * delta_MCY)); mcj += 1)
                for (int mci = -(int)(Constants.CotrexWidth_MiniColumns / (2.0f * delta_MCX)); mci <= (int)(Constants.CotrexWidth_MiniColumns / (2.0f * delta_MCX)); mci += 1)
                {
                    bool r = mcj % 2 == 0;
                    if ((300 + mci + (r ? 0 : 2)) % 3 == 2)
                        continue;

                    float mcx = (mci + (r ? 0.0f : 0.5f)) * delta_MCX;
                    float mcy = mcj * delta_MCY;

                    if (mcx <= -x_Limit_MiniColumns || mcx >= x_Limit_MiniColumns ||
                            mcy <= -y_Limit_MiniColumns || mcy >= y_Limit_MiniColumns)
                        continue;

                    float min_abs_r2 = Single.MaxValue;
                    MiniColumn? nearestMiniColumn = null;
                    for (int mc_index = 0; mc_index < MiniColumns.Count; mc_index += 1)
                    {
                        MiniColumn mc = MiniColumns[mc_index];

                        float abs_r2 = (mcx - mc.MCX) * (mcx - mc.MCX) + (mcy - mc.MCY) * (mcy - mc.MCY);

                        if (abs_r2 < min_abs_r2)
                        {
                            min_abs_r2 = abs_r2;
                            nearestMiniColumn = mc;
                        }
                    }

                    if (nearestMiniColumn is not null)
                        HyperColumnCenters_MiniColumnIndices.Add(nearestMiniColumn.Index);
                }
        }

        IdealPinwheelMemories = new FastList<Memory>(HyperColumnCenters_MiniColumnIndices.Count * 7);

        foreach (int mc_index in HyperColumnCenters_MiniColumnIndices)
        {
            MiniColumn hyperColumnCenter_MiniColumn = MiniColumns[mc_index];

            // Воспоминания для оценки качества вертушки
            var inputItem = AddInputItem(random, hyperColumnCenter_MiniColumn, hyperColumnCenter_MiniColumn, hyperColumnCenter_MiniColumn);
            IdealPinwheelMemories.Add(Memory.FromInputItem(inputItem));

            FastList<MiniColumn> adjacentMiniColumns = new(6);
            for (int mc_index2 = 0; mc_index2 < MiniColumns.Count; mc_index2 += 1)
            {
                if (mc_index2 == mc_index)
                    continue;

                MiniColumn nearestMc = MiniColumns[mc_index2];

                float r2 = (nearestMc.MCX - hyperColumnCenter_MiniColumn.MCX) * (nearestMc.MCX - hyperColumnCenter_MiniColumn.MCX) + 
                    (nearestMc.MCY - hyperColumnCenter_MiniColumn.MCY) * (nearestMc.MCY - hyperColumnCenter_MiniColumn.MCY);
                float r = MathF.Sqrt(r2);                

                if (r < 1.00001f)
                    adjacentMiniColumns.Add(nearestMc);
            }

            foreach (var adjacentMiniColumn in adjacentMiniColumns
                .OrderBy(mc => MathF.Atan2(mc.MCY - hyperColumnCenter_MiniColumn.MCY, mc.MCX - hyperColumnCenter_MiniColumn.MCX)))
            {
                inputItem = AddInputItem(random, hyperColumnCenter_MiniColumn, adjacentMiniColumn, adjacentMiniColumn);
                IdealPinwheelMemories.Add(Memory.FromInputItem(inputItem));
            }
        }        
    }

    public void Prepare()
    {
        float hypercolumnDefinedRadius_MiniColumns = Constants.HyperColumnDefinedRadius_MiniColumns + 0.000001f;
        float _2hypercolumnDefinedRadius_MiniColumns = 2.0f * Constants.HyperColumnDefinedRadius_MiniColumns + 0.000001f;
        float sameFieldOfViewRadius_MiniColumns = Constants.HyperColumnDefinedRadius_MiniColumns / Constants.HyperColumnDiameter_Retina + 0.000001f;

        for (int mc_index = 0; mc_index < MiniColumns.Count; mc_index += 1)            
        {
            MiniColumn miniColumn = MiniColumns[mc_index];
            miniColumn.Prepare(sameFieldOfViewRadius_MiniColumns);

            MiniColumn nearest_HyperColumnCenter_MiniColumn = GetNearest_HyperColumnCenter_MiniColumn(miniColumn);
            if (nearest_HyperColumnCenter_MiniColumn.Temp_HyperColumnMiniColumns is null)
                nearest_HyperColumnCenter_MiniColumn.Temp_HyperColumnMiniColumns = new FastList<MiniColumn>((int)(Math.PI * 25.0f * Constants.HyperColumnDefinedRadius_MiniColumns * Constants.HyperColumnDefinedRadius_MiniColumns));
            nearest_HyperColumnCenter_MiniColumn.Temp_HyperColumnMiniColumns.Add(miniColumn);
        }

        // Находим ближайшие миниколонки для каждой миниколонки
        Parallel.For(
            fromInclusive: 0,
            toExclusive: MiniColumns.Count,
            (Action<int>)(mci =>
            {
                MiniColumn miniColumn = MiniColumns[mci];
                miniColumn.Temp_SameFieldOfViewMiniColumns.Add(miniColumn);

                for (int mc_index2 = 0; mc_index2 < MiniColumns.Count; mc_index2 += 1)                    
                {
                    if (mc_index2 == mci)                        
                        continue;

                    MiniColumn nearestMc = MiniColumns[mc_index2];                    

                    float r2 = (nearestMc.MCX - miniColumn.MCX) * (nearestMc.MCX - miniColumn.MCX) + (nearestMc.MCY - miniColumn.MCY) * (nearestMc.MCY - miniColumn.MCY);
                    float r = MathF.Sqrt(r2);

                    if (r < 2.00001f)
                        miniColumn.Temp_K_ForNearestMiniColumns.Add(
                            (MathHelper.GetInterpolatedValue(Constants.PositiveK, r),
                            MathHelper.GetInterpolatedValue(Constants.NegativeK, r),
                            nearestMc));

                    if (r < hypercolumnDefinedRadius_MiniColumns)
                        miniColumn.Temp_K_HyperColumnMiniColumns.Add((r, nearestMc));

                    if (r < _2hypercolumnDefinedRadius_MiniColumns)
                        miniColumn.Temp_K_2HyperColumnMiniColumns.Add((r2, nearestMc));        

                    if (r < sameFieldOfViewRadius_MiniColumns)                    
                        miniColumn.Temp_SameFieldOfViewMiniColumns.Add(nearestMc);

                    if (r < 1.00001f)
                        miniColumn.Temp_AdjacentMiniColumns.Add((r, nearestMc));
                }

                miniColumn.Temp_AdjacentMiniColumns = miniColumn.Temp_AdjacentMiniColumns
                    .OrderBy(it => MathF.Atan2(miniColumn.MCY - it.Item2.MCY, miniColumn.MCX - it.Item2.MCX))
                    .ToFastList();
            }));
    }

    public MiniColumn GetNearest_HyperColumnCenter_MiniColumn(MiniColumn miniColumn)
    {
        float min_abs_r2 = Single.MaxValue;
        MiniColumn? nearest_HyperColumnCenter_MiniColumn = null;
        for (int mc_index = 0; mc_index < HyperColumnCenters_MiniColumnIndices.Count; mc_index += 1)
        {
            MiniColumn mc = MiniColumns[HyperColumnCenters_MiniColumnIndices[mc_index]];

            float abs_r2 = (miniColumn.MCX - mc.MCX) * (miniColumn.MCX - mc.MCX) + (miniColumn.MCY - mc.MCY) * (miniColumn.MCY - mc.MCY);

            if (abs_r2 < min_abs_r2)
            {
                min_abs_r2 = abs_r2;
                nearest_HyperColumnCenter_MiniColumn = mc;
            }
        }
        return nearest_HyperColumnCenter_MiniColumn!;
    }

    /// <summary>
    ///     
    /// </summary>
    /// <param name="random"></param>
    /// <param name="hyperColumnCenter_MiniColumn"></param>
    /// <param name="idealAngleMagnitude_MiniColumn">For Angle and Magnitude</param>
    /// <param name="mainXY_MiniColumn">For X_Retina and Y_Retina</param>
    /// <returns></returns>
    public InputItem AddInputItem(
        Random random, 
        MiniColumn hyperColumnCenter_MiniColumn, 
        MiniColumn idealAngleMagnitude_MiniColumn, 
        MiniColumn mainXY_MiniColumn)
    {        
        float angle = MathHelper.NormalizeAngle(MathF.Atan2((idealAngleMagnitude_MiniColumn.MCY - hyperColumnCenter_MiniColumn.MCY), (idealAngleMagnitude_MiniColumn.MCX - hyperColumnCenter_MiniColumn.MCX)));
        if (MathF.Abs(hyperColumnCenter_MiniColumn.MCX) < 1.0f && MathF.Abs(hyperColumnCenter_MiniColumn.MCY) < 1.0f)
        {                 
        }
        else
        {
            float angleHypercolumn = MathHelper.NormalizeAngle(MathF.Atan2(hyperColumnCenter_MiniColumn.MCY, hyperColumnCenter_MiniColumn.MCX));
            float delta = angle - angleHypercolumn;
            angle = angleHypercolumn + MathF.PI - delta;
        }

        InputItem inputItem = new();
        inputItem.Index = InputItems.Count;
        inputItem.Angle = angle;
        inputItem.Magnitude = MathF.Sqrt((idealAngleMagnitude_MiniColumn.MCY - hyperColumnCenter_MiniColumn.MCY) * (idealAngleMagnitude_MiniColumn.MCY - hyperColumnCenter_MiniColumn.MCY)
            + (idealAngleMagnitude_MiniColumn.MCX - hyperColumnCenter_MiniColumn.MCX) * (idealAngleMagnitude_MiniColumn.MCX - hyperColumnCenter_MiniColumn.MCX));
        inputItem.MainXY_MiniColumnIndex = mainXY_MiniColumn.Index;
        inputItem.X_Retina = mainXY_MiniColumn.MCX * MiniColumnX_Retina;
        inputItem.Y_Retina = mainXY_MiniColumn.MCY * MiniColumnY_Retina;
              
        inputItem.HyperColumnCenter_MiniColumnIndex = hyperColumnCenter_MiniColumn!.Index;
        inputItem.X_HyperColumnCenter_Retina = hyperColumnCenter_MiniColumn.MCX * MiniColumnX_Retina;
        inputItem.Y_HyperColumnCenter_Retina = hyperColumnCenter_MiniColumn.MCY * MiniColumnY_Retina;        

        var distanceFromCenterNormalized = inputItem.Magnitude / (Constants.HyperColumnDefinedRadius_MiniColumns + 5);
        if (distanceFromCenterNormalized > 1.0f)
            distanceFromCenterNormalized = 1.0f;
        inputItem.ColorAngleMagnitude = Visualisation.ColorFromHSV((double)(inputItem.Angle + MathF.PI) / (2 * MathF.PI), distanceFromCenterNormalized, 1.0);

        float angleXY = MathHelper.NormalizeAngle(MathF.Atan2(inputItem.Y_HyperColumnCenter_Retina, inputItem.X_HyperColumnCenter_Retina));
        var sXY = MathF.Sqrt(hyperColumnCenter_MiniColumn.MCX * hyperColumnCenter_MiniColumn.MCX + hyperColumnCenter_MiniColumn.MCY * hyperColumnCenter_MiniColumn.MCY) * 2.0f / 
            MathF.Sqrt(Constants.CotrexWidth_MiniColumns * Constants.CotrexWidth_MiniColumns + Constants.CotrexHeight_MiniColumns * Constants.CotrexHeight_MiniColumns);
        inputItem.ColorXY = Visualisation.ColorFromHSV((double)(angleXY + MathF.PI) / (2 * MathF.PI), sXY, 1.0);

        inputItem.DistanceFromCenter = inputItem.Magnitude;        

        InputItems.Add(inputItem);
        return inputItem;
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {
            writer.WriteFastListOfOwnedDataSerializable(InputItems, context);
            writer.WriteFastListOfOwnedDataSerializable(IdealPinwheelMemories, context);
            writer.WriteFastListOfOwnedDataSerializable(MiniColumns, context);
            HyperColumnCenters_MiniColumnIndices.SerializeOwnedData(writer, context);            
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    InputItems = reader.ReadFastListOfOwnedDataSerializable(idx => new InputItem(), context);
                    IdealPinwheelMemories = reader.ReadFastListOfOwnedDataSerializable(idx => new Memory(), context);
                    MiniColumns = reader.ReadFastListOfOwnedDataSerializable(idx => new MiniColumn(Constants), context);
                    HyperColumnCenters_MiniColumnIndices = new FastList<int>();
                    HyperColumnCenters_MiniColumnIndices.DeserializeOwnedData(reader, context);
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

        public int Index;

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
        ///     Окружающие миниколонки в радиусе примерно 0.5 гиперколонки.
        ///     <para>(r, MiniColumn)</para>        
        /// </summary>
        public FastList<(float, MiniColumn)> Temp_K_HyperColumnMiniColumns = null!;

        /// <summary>
        ///     Окружающие миниколонки в радиусе примерно 1 гиперколонки.
        ///     <para>(r^2, MiniColumn)</para>        
        /// </summary>
        public FastList<(float, MiniColumn)> Temp_K_2HyperColumnMiniColumns = null!;

        /// <summary>
        ///     <para>!!! Сама миниколонка !!! и окружающие миниколонки в радиусе примерно 0.5 гиперколонки.</para>
        ///     <para>Может быть неровной формы.</para>
        ///     <para>Определено только для центров гиперколонок.</para>
        /// </summary>
        public FastList<MiniColumn> Temp_HyperColumnMiniColumns = null!;

        /// <summary>
        ///     !!! Сама миниколонка !!! и окружающие миниколонки в радиусе примерно 5 гиперколонок.
        /// </summary>
        public FastList<MiniColumn> Temp_SameFieldOfViewMiniColumns = null!;

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

        public void Prepare(float sameFieldOfViewRadius_MiniColumns)
        {
            Temp_K_ForNearestMiniColumns = new FastList<(float, float, MiniColumn)>(18);
            Temp_K_HyperColumnMiniColumns = new FastList<(float, MiniColumn)>((int)(Math.PI * Constants.HyperColumnDefinedRadius_MiniColumns * Constants.HyperColumnDefinedRadius_MiniColumns));
            Temp_K_2HyperColumnMiniColumns = new FastList<(float, MiniColumn)>((int)(Math.PI * 4.0f * Constants.HyperColumnDefinedRadius_MiniColumns * Constants.HyperColumnDefinedRadius_MiniColumns));            
            Temp_SameFieldOfViewMiniColumns = new FastList<MiniColumn>((int)(Math.PI * sameFieldOfViewRadius_MiniColumns * sameFieldOfViewRadius_MiniColumns + 5));
            Temp_AdjacentMiniColumns = new FastList<(double, MiniColumn)>(6);
            Temp_CortexMemories = new(10);
        }

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(1))
            {
                writer.Write(Index);
                writer.Write(MCX);
                writer.Write(MCY);
                writer.WriteFastListOfOwnedDataSerializable(CortexMemories, context);                
            }
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            using (Block block = reader.EnterBlock())
            {
                switch (block.Version)
                {
                    case 1:
                        Index = reader.ReadInt32();
                        MCX = reader.ReadSingle();
                        MCY = reader.ReadSingle();
                        CortexMemories = reader.ReadFastListOfOwnedDataSerializable(idx => (Memory?)new Memory(), context);
                        break;                    
                }
            }
        } 
    }

    public class Memory : IOwnedDataSerializable
    {   
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
