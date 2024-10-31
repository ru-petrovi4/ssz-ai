using System.DrawingCore;

namespace Ssz.AI.Models
{
    public class VisualizationTableItem
    {
        public float[] Hash { get; init; } = null!;

        public Color Color { get; set; }

        public Bitmap Image { get; set; } = null!;
    }
}
