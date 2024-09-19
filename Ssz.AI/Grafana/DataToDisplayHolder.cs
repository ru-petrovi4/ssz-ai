using System;

namespace Ssz.AI.Grafana
{
    public class DataToDisplayHolder
    {
        public GradientDistribution GradientDistribution { get; set; } = new GradientDistribution();
    }

    public class GradientDistribution
    {
        /// <summary>
        ///     количество
        /// </summary>
        public int[] Data = Array.Empty<int>();
    }
}
