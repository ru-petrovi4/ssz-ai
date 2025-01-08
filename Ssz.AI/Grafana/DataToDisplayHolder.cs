using Ssz.AI.Models;
using System;
using System.DrawingCore;
using static Ssz.AI.Models.Cortex;

namespace Ssz.AI.Grafana
{
    public class DataToDisplayHolder
    {
        public GradientDistribution GradientDistribution { get; set; } = new GradientDistribution();

        public ulong[] MiniColumsActivatedDetectorsCountDistribution { get; set; } = new ulong[new Model4.ModelConstants().MiniColumnVisibleDetectorsCount];

        public ulong[] MiniColumsBitsCountInHashDistribution { get; set; } = new ulong[new Model4.ModelConstants().HashLength];

        /// <summary>
        ///     [mcx, mcy, bits count]
        /// </summary>
        public ulong[,,] MiniColumsBitsCountInHashDistribution2 { get; set; } = new ulong[0, 0, 0];

        public MiniColumn? ContextSyncingMiniColumn { get; set; }
        //public MatrixFloat? ContextSyncingMatrixFloat { get; set; }
        //public Image? ContextSyncingImage { get; set; }
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
}
