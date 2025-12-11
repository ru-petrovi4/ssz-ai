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

public interface IModelConstants
{
    /// <summary>
    ///     Радиус зоны коры в миниколонках.
    /// </summary>
    int CortexRadius_MiniColumns { get; }
}

public partial class Cortex : ISerializableModelObject
{
    /// <summary>
    ///     Если задано SubAreaMiniColumnsCount, то генерируется только подмножество миниколонок с центром SubAreaCenter_Cx, SubAreaCenter_Cy и количеством SubAreaMiniColumnsCount
    /// </summary>
    /// <param name="constants"></param>        
    public Cortex(
        IModelConstants constants, 
        ILogger logger)
    {
        Constants = constants;
        Logger = logger;
        InputItems = new(10000);
    }

    #region public functions

    public readonly IModelConstants Constants;

    public readonly ILogger Logger;

    public List<InputItem> InputItems { get; private set; } = null!;

    public FastList<MiniColumn> MiniColumns { get; private set; } = null!;    

    public string Temp_InputCurrentDesc = null!;

    public void GenerateOwnedData(Random random)
    {
        MiniColumns = new FastList<MiniColumn>((int)(Math.PI * Constants.CortexRadius_MiniColumns * Constants.CortexRadius_MiniColumns));
        
        float delta_MCY = MathF.Sqrt(1.0f - 0.5f * 0.5f);
        float maxRadius = Constants.CortexRadius_MiniColumns + 0.00001f;

        for (int mcj = -(int)(Constants.CortexRadius_MiniColumns / delta_MCY); mcj <= (int)(Constants.CortexRadius_MiniColumns / delta_MCY); mcj += 1)
            for (int mci = -Constants.CortexRadius_MiniColumns; mci <= Constants.CortexRadius_MiniColumns; mci += 1)
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
                }
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

                    double k = (nearestMc.MCX - miniColumn.MCX) * (nearestMc.MCX - miniColumn.MCX) + (nearestMc.MCY - miniColumn.MCY) * (nearestMc.MCY - miniColumn.MCY);
                    double r = Math.Sqrt(k);

                    if (r < 3.00001)
                        miniColumn.Temp_NearestForEnergyMiniColumns.Add((k, nearestMc));
                    
                    //if (r < 1.00001)
                    miniColumn.Temp_CandidateForSwapMiniColumns.Add((r, nearestMc));

                    if (r < 1.00001)
                        miniColumn.Temp_AdjacentMiniColumns.Add((r, nearestMc));
                }
            });
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
        public MiniColumn(IModelConstants constants)
        {
            Constants = constants;
        }

        public readonly IModelConstants Constants;

        /// <summary>
        ///     Координата миниколонки по оси X (горизонтально вправо)
        /// </summary>
        public float MCX;

        /// <summary>
        ///     Координата миниколонки по оси Y (вертикально вниз)
        /// </summary>
        public float MCY;

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

        public double Temp_Energy;

        public double Temp_Distance;

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
            Temp_NearestForEnergyMiniColumns = new FastList<(double, MiniColumn)>((int)(Math.PI * Constants.CortexRadius_MiniColumns * Constants.CortexRadius_MiniColumns));
            Temp_CandidateForSwapMiniColumns = new FastList<(double, MiniColumn)>((int)(Math.PI * Constants.CortexRadius_MiniColumns * Constants.CortexRadius_MiniColumns));
            Temp_AdjacentMiniColumns = new FastList<(double, MiniColumn)>(6);
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
