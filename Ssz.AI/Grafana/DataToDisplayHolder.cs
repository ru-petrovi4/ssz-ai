using Ssz.AI.Models;
using System;
using System.Drawing;
using static Ssz.AI.Models.Cortex;

namespace Ssz.AI.Grafana
{
    public class DataToDisplayHolder
    {
        public GradientDistribution GradientDistribution { get; set; } = new GradientDistribution();

        public ulong[] MiniColumsActivatedDetectorsCountDistribution { get; set; } = new ulong[new Model4.ModelConstants().MiniColumnVisibleDetectorsCount];

        public ulong[] MiniColumsBitsCountInHashDistribution { get; set; } = new ulong[new Model05.ModelConstants().HashLength];

        /// <summary>
        ///     [mcx, mcy, bits count]
        /// </summary>
        public ulong[,,] MiniColumsBitsCountInHashDistribution2 { get; set; } = new ulong[0, 0, 0];        

        public MiniColumn? ContextSyncingMiniColumn { get; set; }
        //public MatrixFloat? ContextSyncingMatrixFloat { get; set; }
        //public Image? ContextSyncingImage { get; set; }

        /// <summary>
        ///     Распределение чего-либо
        /// </summary>
        public ulong[] Distribution { get; set; } = null!;
    }

    public class GradientDistribution
    {
        /// <summary>
        ///     Количество примеров в зависимости от модуля градиента
        /// </summary>
        public readonly UInt64[] MagnitudeData = new UInt64[SobelOperator.MagnitudeMaxValue];

        /// <summary>
        ///     Количество примеров в зависимости от угла (в градусах) градиента
        /// </summary>
        public UInt64[] AngleData = new UInt64[360];
    }

    /// <summary>
    ///     Диапазоны для детекторов в зависимости от модуля градиента и угла градиента (в градусах).
    /// </summary>
    public class DetectorRanges
    {
        /// <summary>
        ///     Диапазоны модуля градиента в зависимости от модуля градиента и угла градиента (в градусах).
        /// </summary>
        public MatrixFloat GradientMagnitudeRanges = new();

        /// <summary>
        ///     Диапазоны угла градиента в зависимости от модуля градиента и угла градиента (в градусах).
        /// </summary>
        public MatrixFloat GradientAngleRanges = new();
    }
}
