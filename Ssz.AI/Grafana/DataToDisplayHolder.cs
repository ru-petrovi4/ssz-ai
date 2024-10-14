using Ssz.AI.Models;
using System;

namespace Ssz.AI.Grafana
{
    public class DataToDisplayHolder
    {
        public GradientDistribution GradientDistribution { get; set; } = new GradientDistribution();

        public ulong[] MiniColumsActiveBitsDistribution { get; set; } = new ulong[Model4.MiniColumnVisibleDetectorsCount];
    }

    public class GradientDistribution
    {
        /// <summary>
        ///     количество
        /// </summary>
        public UInt64[] MagnitudeData = Array.Empty<UInt64>();

        /// <summary>
        ///     количество
        /// </summary>
        public UInt64[] AngleData = Array.Empty<UInt64>();
    }
}
