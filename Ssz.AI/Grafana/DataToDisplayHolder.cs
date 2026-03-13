using Ssz.AI.Models;
using System;
using System.Drawing;
using static Ssz.AI.Models.Cortex_Simplified;

namespace Ssz.AI.Grafana;

public class DataToDisplayHolder
{
    public static DataToDisplayHolder Instance { get; } = new DataToDisplayHolder();

    public GradientDistribution GradientDistribution { get; set; } = new GradientDistribution();

    public ulong[] MiniColumsBitsCountInHashDistribution { get; set; } = new ulong[new Model05.ModelConstants().HashLength];

    /// <summary>
    ///     [mcx, mcy, bits count]
    /// </summary>
    public ulong[,,] WithCoordinate_MiniColumsBitsCountInHashDistribution { get; set; } = new ulong[0, 0, 0];        

    public MiniColumn? ContextSyncingMiniColumn { get; set; }
    //public MatrixFloat? ContextSyncingMatrixFloat { get; set; }
    //public Image? ContextSyncingImage { get; set; }

    /// <summary>
    ///     Распределение чего-либо
    /// </summary>
    public ulong[] Distribution { get; set; } = null!;

    public float DistributionXMin { get; set; } = 0.0f;
    public float DistributionXMax { get; set; } = 1.0f;
}

public class GradientDistribution //: IOwnedDataSerializable
{
    /// <summary>
    ///     Количество примеров в зависимости от модуля градиента
    /// </summary>
    public UInt64[] MagnitudeData = new UInt64[SobelOperator.MagnitudeMaxValue];

    /// <summary>
    ///     Количество примеров в зависимости от угла (в градусах) градиента
    /// </summary>
    public UInt64[] AngleData = new UInt64[360];

    //public void SerializeOwnedData(SerializationWriter writer, object? context)
    //{
    //    writer.WriteArrayOfUInt64(MagnitudeData);
    //    writer.WriteArrayOfUInt64(AngleData);
    //}

    //public void DeserializeOwnedData(SerializationReader reader, object? context)
    //{
    //    MagnitudeData = reader.ReadArrayOfUInt64();
    //    AngleData = reader.ReadArrayOfUInt64();
    //}
}