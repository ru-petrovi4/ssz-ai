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
using System.Threading.Tasks;

namespace Ssz.AI.Models.CortexVisualisationModel;

public partial class Cortex : ISerializableModelObject
{
    /// <summary>
    ///     Если задано SubAreaMiniColumnsCount, то генерируется только подмножество миниколонок с центром SubAreaCenter_Cx, SubAreaCenter_Cy и количеством SubAreaMiniColumnsCount
    /// </summary>
    /// <param name="constants"></param>        
    public Cortex(
        Model01.ModelConstants constants, 
        ILogger logger)
    {
        Constants = constants;
        Logger = logger;
        InputItems = new(10000);
    }

    #region public functions

    public readonly Model01.ModelConstants Constants;

    public readonly ILogger Logger;

    public List<InputItem> InputItems { get; private set; } = null!;

    public DenseMatrix<MiniColumn?> MiniColumns { get; private set; } = null!;    

    public string Temp_InputCurrentDesc = null!;

    public void GenerateOwnedData(Random random)
    {
        MiniColumns = new DenseMatrix<MiniColumn?>(Constants.CortexWidth_MiniColumns, Constants.CortexHeight_MiniColumns);

        int center_MCX = MiniColumns.Dimensions[0] / 2;
        int center_MCY = MiniColumns.Dimensions[1] / 2;
        float maxRadius = center_MCX;

        for (int mcy = 0; mcy < MiniColumns.Dimensions[1]; mcy += 1)
            for (int mcx = 0; mcx < MiniColumns.Dimensions[0]; mcx += 1)
            {
                float radius = MathF.Sqrt((center_MCX - mcx) * (center_MCX - mcx) + (center_MCY - mcy) * (center_MCY - mcy));
                if (radius <= maxRadius + 1)
                {
                    MiniColumn miniColumn = new MiniColumn(
                        Constants,
                        mcx,
                        mcy);

                    miniColumn.GenerateOwnedData();

                    MiniColumns[mcx, mcy] = miniColumn;
                }
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
                MiniColumn? miniColumn = MiniColumns.Data[mci];
                if (miniColumn is null)
                    return;
                miniColumn.Prepare();                                           

                for (int mcy = 0; mcy < MiniColumns.Dimensions[1]; mcy += 1)
                    for (int mcx = 0; mcx < MiniColumns.Dimensions[0]; mcx += 1)
                    {
                        if (mcx == miniColumn.MCX && mcy == miniColumn.MCY)
                            continue;

                        MiniColumn? nearestMc = MiniColumns[mcx, mcy];
                        if (nearestMc is null)
                            continue;

                        double k = MathF.Pow((mcx - miniColumn.MCX) * (mcx - miniColumn.MCX) + (mcy - miniColumn.MCY) * (mcy - miniColumn.MCY), 1.0f);
                        miniColumn.Temp_K_ForNearestMiniColumns.Add((k, nearestMc));

                        if (Math.Abs(mcx - miniColumn.MCX) <= 1 && Math.Abs(mcy - miniColumn.MCY) <= 1)
                            miniColumn.Temp_AdjacentMiniColumns.Add(nearestMc);
                    }
            });
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {
            writer.WriteListOfOwnedDataSerializable(InputItems, context);
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
                    InputItems = reader.ReadListOfOwnedDataSerializable(() => new InputItem(), context);
                    MiniColumns = Ssz.AI.Helpers.SerializationHelper.DeserializeOwnedData_DenseMatrix(reader, context, (mcx, mcy) => new MiniColumn(Constants, mcx, mcy));
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
        }

        public readonly Model01.ModelConstants Constants;

        /// <summary>
        ///     Индекс миниколонки в матрице по оси X (горизонтально вправо)
        /// </summary>
        public int MCX { get; }

        /// <summary>
        ///     Индекс миниколонки в матрице по оси Y (вертикально вниз)
        /// </summary>
        public int MCY { get; }

        /// <summary>
        ///     Окружающие миниколонки, для которых считается энергия.
        ///     <para>(r^2, MiniColumn)</para>        
        /// </summary>
        public FastList<(double, MiniColumn)> Temp_K_ForNearestMiniColumns = null!;

        /// <summary>
        ///     Смежные миниколонки
        ///     <para>(r^2, MiniColumn)</para>        
        /// </summary>
        public FastList<MiniColumn> Temp_AdjacentMiniColumns = null!;

        public double Temp_Energy;

        /// <summary>
        ///     Сохраненные хэш-коды
        /// </summary>
        public FastList<Memory?> CortexMemories = null!;        

        public void GenerateOwnedData()
        {
            CortexMemories = new(1);
        }

        public void Prepare()
        {            
            Temp_K_ForNearestMiniColumns = new FastList<(double, MiniColumn)>(Constants.CortexWidth_MiniColumns * Constants.CortexHeight_MiniColumns);
            Temp_AdjacentMiniColumns = new FastList<MiniColumn>(8);
        }

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(1))
            {
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
                        CortexMemories = Ssz.AI.Helpers.SerializationHelper.DeserializeOwnedData_FastList(reader, context, idx => (Memory?)new Memory());
                        break;                    
                }
            }
        } 
    }

    public class Memory : IOwnedDataSerializable
    {        
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
